# app/services/candidate_matcher_service.py

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel
import logging

logger = logging.getLogger("candidate_matcher")

# === Config ===
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
SCORING_WEIGHTS = {
    "semantic": 0.45,
    "skill_overlap": 0.25,
    "keyword": 0.15,
    "recency": 0.15
}


# === Data Models ===
class ResumeProfile(BaseModel):
    candidate_id: str
    resume_text: str
    skills: List[str]
    created_at: Optional[str] = datetime.utcnow().isoformat()

class CandidateScore(BaseModel):
    candidate_id: str
    semantic_score: float
    skill_overlap: float
    keyword_match_score: float
    recency_score: float
    final_score: float
    matched_skills: List[str]
    explanation: Dict[str, float]

class CandidateMatchResponse(BaseModel):
    job_id: str
    matches: List[CandidateScore]


# === Candidate Matcher Service ===
class CandidateMatcherService:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.vectors: List[np.ndarray] = []
        self.store: Dict[str, dict] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}

    def _encode(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def _keyword_overlap(self, a: str, b: str) -> float:
        return len(set(a.lower().split()) & set(b.lower().split())) / max(len(b.lower().split()), 1)

    def _recency_score(self, created_at: Optional[str]) -> float:
        try:
            dt = datetime.fromisoformat(created_at)
            return max(0.0, 1.0 - (datetime.utcnow() - dt).days / 365.0)
        except Exception:
            return 0.5

    def index_candidate(self, resume: ResumeProfile):
        try:
            if resume.candidate_id in self.store:
                logger.info(f"[Update] Resume already exists: {resume.candidate_id}")
                self.delete_candidate(resume.candidate_id)

            vector = self._encode(resume.resume_text + " " + " ".join(resume.skills))
            idx = len(self.vectors)
            self.vectors.append(vector)
            self.index.add(np.array([vector]))

            self.store[resume.candidate_id] = resume.dict()
            self.id_to_index[resume.candidate_id] = idx
            self.index_to_id[idx] = resume.candidate_id

            logger.info(f"[Index] Candidate {resume.candidate_id} at {idx}")
        except Exception as e:
            logger.exception(f"[Indexing Error] {e}")
            raise

    def delete_candidate(self, candidate_id: str):
        if candidate_id in self.store:
            del self.store[candidate_id]
            self.rebuild_index()

    def rebuild_index(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.vectors.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        for resume in self.store.values():
            self.index_candidate(ResumeProfile(**resume))

    def clear_index(self):
        self.index.reset()
        self.vectors.clear()
        self.store.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()

    def reverse_match(
        self,
        job_id: str,
        job_title: str,
        job_description: str,
        required_skills: List[str],
        top_k: int = 5
    ) -> List[CandidateScore]:
        if not self.vectors:
            raise ValueError("No candidates indexed.")

        job_text = f"{job_title}. {job_description}. Skills: {'; '.join(required_skills)}"
        job_vector = self._encode(job_text)
        D, I = self.index.search(np.array([job_vector]), len(self.vectors))

        results = []
        for idx in I[0]:
            cand_id = self.index_to_id.get(idx)
            cand = self.store[cand_id]

            resume_vec = self.vectors[idx]
            semantic = float(np.dot(job_vector, resume_vec))
            matched_skills = list(set(required_skills) & set(cand["skills"]))
            skill_overlap = len(matched_skills) / max(len(required_skills), 1)
            keyword_score = self._keyword_overlap(job_description, cand["resume_text"])
            recency = self._recency_score(cand.get("created_at"))

            final_score = sum([
                SCORING_WEIGHTS["semantic"] * semantic,
                SCORING_WEIGHTS["skill_overlap"] * skill_overlap,
                SCORING_WEIGHTS["keyword"] * keyword_score,
                SCORING_WEIGHTS["recency"] * recency
            ])

            results.append(CandidateScore(
                candidate_id=cand_id,
                semantic_score=round(semantic, 4),
                skill_overlap=round(skill_overlap, 4),
                keyword_match_score=round(keyword_score, 4),
                recency_score=round(recency, 4),
                final_score=round(final_score, 4),
                matched_skills=matched_skills,
                explanation={
                    "semantic": semantic,
                    "skill_overlap": skill_overlap,
                    "keyword_match": keyword_score,
                    "recency": recency
                }
            ))

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]
