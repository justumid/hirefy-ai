# app/services/job_matcher_service.py

import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

logger = logging.getLogger("job_matcher_service")

# === Configuration ===
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
SCORING_WEIGHTS = {
    "semantic": 0.45,
    "skill_overlap": 0.25,
    "keyword": 0.15,
    "recency": 0.15
}

# === Pydantic Models ===

class JobPosting(BaseModel):
    job_id: str
    title: str
    description: str
    required_skills: List[str]
    location: Optional[str] = None
    created_at: Optional[str] = None


class ResumeProfile(BaseModel):
    candidate_id: str
    resume_text: str
    skills: List[str]


class MatchScore(BaseModel):
    job_id: str
    semantic_score: float
    skill_overlap: float
    keyword_match_score: float
    recency_score: float
    final_score: float
    matched_skills: List[str]
    explanation: Dict[str, float]


class MatchResponse(BaseModel):
    candidate_id: str
    matches: List[MatchScore]

# === Job Matching Engine ===

class JobMatcherService:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.job_vectors: List[np.ndarray] = []
        self.job_store: Dict[str, dict] = {}
        self.job_id_to_index: Dict[str, int] = {}
        self.index_to_job_id: Dict[int, str] = {}

    def encode(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(EMBEDDING_DIM)
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def _keyword_overlap(self, text_a: str, text_b: str) -> float:
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        return len(tokens_a & tokens_b) / max(len(tokens_b), 1)

    def _recency_score(self, created_at: Optional[str]) -> float:
        try:
            dt = datetime.fromisoformat(created_at)
            delta = (datetime.utcnow() - dt).days
            return max(0.0, 1.0 - delta / 365.0)
        except Exception:
            return 0.5

    # === Indexing ===

    def index_job(self, job: JobPosting):
        try:
            if job.job_id in self.job_store:
                logger.info(f"[Update] Job ID already exists: {job.job_id}. Overwriting.")
                self.delete_job(job.job_id)

            combined_text = f"{job.title}. {job.description}. Skills: {'; '.join(job.required_skills)}"
            vector = self.encode(combined_text)

            index_id = len(self.job_vectors)
            self.index.add(np.array([vector]))
            self.job_vectors.append(vector)

            self.job_store[job.job_id] = job.dict()
            self.job_id_to_index[job.job_id] = index_id
            self.index_to_job_id[index_id] = job.job_id

            logger.info(f"[Index] Job {job.job_id} indexed at position {index_id}")
        except Exception as e:
            logger.exception(f"[Error indexing job {job.job_id}] {e}")
            raise

    def delete_job(self, job_id: str):
        if job_id not in self.job_store:
            return
        del self.job_store[job_id]
        self.rebuild_index()

    def clear_index(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.job_vectors.clear()
        self.job_store.clear()
        self.job_id_to_index.clear()
        self.index_to_job_id.clear()
        logger.info("[Index] Cleared FAISS index and store")

    def rebuild_index(self):
        logger.info("[Index] Rebuilding FAISS from stored jobs")
        all_jobs = list(self.job_store.values())
        self.clear_index()
        for job_data in all_jobs:
            try:
                self.index_job(JobPosting(**job_data))
            except Exception as e:
                logger.warning(f"[Rebuild Skipped] Job {job_data.get('job_id')} failed: {e}")

    # === Matching ===

    def hybrid_match(
        self,
        resume: ResumeProfile,
        top_k: int = 5,
        filter_ids: Optional[List[str]] = None
    ) -> List[MatchScore]:
        if not self.job_vectors:
            raise ValueError("No jobs indexed yet.")

        resume_vector = self.encode(resume.resume_text + " " + " ".join(resume.skills))
        scores, indices = self.index.search(np.array([resume_vector]), len(self.job_vectors))

        results = []
        for idx in indices[0]:
            job_id = self.index_to_job_id.get(idx)
            if not job_id:
                continue
            if filter_ids and job_id not in filter_ids:
                continue

            job_data = self.job_store[job_id]
            job_vec = self.job_vectors[idx]
            semantic = float(np.dot(resume_vector, job_vec))

            required_skills = job_data.get("required_skills", [])
            matched_skills = list(set(resume.skills) & set(required_skills))
            skill_overlap = len(matched_skills) / max(len(required_skills), 1)

            keyword_score = self._keyword_overlap(resume.resume_text, job_data["description"])
            recency = self._recency_score(job_data.get("created_at"))

            final_score = sum([
                SCORING_WEIGHTS["semantic"] * semantic,
                SCORING_WEIGHTS["skill_overlap"] * skill_overlap,
                SCORING_WEIGHTS["keyword"] * keyword_score,
                SCORING_WEIGHTS["recency"] * recency
            ])

            results.append(MatchScore(
                job_id=job_id,
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

    # === Temporary Matching APIs ===

    def match_resume_to_job(self, resume_text: str, job_description: str) -> Dict:
        job = JobPosting(
            job_id=str(uuid.uuid4()),
            title="Temp",
            description=job_description,
            required_skills=[],
            created_at=datetime.utcnow().isoformat()
        )
        self.index_job(job)

        profile = ResumeProfile(candidate_id="temp", resume_text=resume_text, skills=[])
        matches = self.hybrid_match(profile, top_k=1, filter_ids=[job.job_id])

        self.delete_job(job.job_id)
        return matches[0].dict() if matches else {"score": 0.0, "explanation": "No match found"}

    def batch_match_resumes(self, resume_texts: List[str], job_description: str) -> List[Dict]:
        job = JobPosting(
            job_id=str(uuid.uuid4()),
            title="BatchTemp",
            description=job_description,
            required_skills=[],
            created_at=datetime.utcnow().isoformat()
        )
        self.index_job(job)

        results = []
        for idx, resume_text in enumerate(resume_texts):
            profile = ResumeProfile(candidate_id=f"cand_{idx}", resume_text=resume_text, skills=[])
            matches = self.hybrid_match(profile, top_k=1, filter_ids=[job.job_id])
            results.append(matches[0].dict() if matches else {"score": 0.0})

        self.delete_job(job.job_id)
        return results

    def match_job_to_candidates(self, job_description: str, candidate_resumes: List[Dict]) -> List[Dict]:
        job = JobPosting(
            job_id=str(uuid.uuid4()),
            title="ReverseMatch",
            description=job_description,
            required_skills=[],
            created_at=datetime.utcnow().isoformat()
        )
        self.index_job(job)

        results = []
        for c in candidate_resumes:
            profile = ResumeProfile(candidate_id=c["candidate_id"], resume_text=c["resume_text"], skills=[])
            matches = self.hybrid_match(profile, top_k=1, filter_ids=[job.job_id])
            result = matches[0].dict() if matches else {"score": 0.0}
            result["candidate_id"] = c["candidate_id"]
            results.append(result)

        self.delete_job(job.job_id)
        return results
