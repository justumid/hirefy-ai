# services/resume_generator_service.py

import os
import json
import logging
from typing import List, Dict

import openai
from openai import OpenAIError
from sentence_transformers import SentenceTransformer, util

from app.base.models import ResumeGenerationRequest, ResumeGenerationResult

logger = logging.getLogger("resume_generator_service")


class ResumeGeneratorService:
    def __init__(self):
        self.model_name = os.getenv("RESUME_MODEL_NAME", "gpt-4-turbo")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.skill_bank = self.load_skill_bank()

    def load_skill_bank(self, path: str = "skill_bank.json") -> List[str]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[ResumeGenerator] Failed to load skill bank: {e}")
        logger.warning("[ResumeGenerator] Skill bank not found or empty.")
        return []

    def expand_skills(self, skills: List[str], top_k: int = 5) -> List[str]:
        if not self.skill_bank:
            return skills

        try:
            base_embeds = self.sbert_model.encode(skills, convert_to_tensor=True)
            bank_embeds = self.sbert_model.encode(self.skill_bank, convert_to_tensor=True)
            expanded = set(skills)

            for emb in base_embeds:
                scores = util.cos_sim(emb, bank_embeds)[0]
                top_indices = scores.argsort(descending=True)[:top_k]
                for i in top_indices:
                    expanded.add(self.skill_bank[i])

            return sorted(expanded)
        except Exception as e:
            logger.warning(f"[ResumeGenerator] Skill expansion failed: {e}")
            return skills

    def _build_prompt(self, req: ResumeGenerationRequest, expanded_skills: List[str]) -> str:
        jd_prompt = f"Target Job Description:\n{req.job_description.strip()}\n\n" if req.job_description else ""
        lang_note = f"Language: {req.language or 'en'}"

        return f"""
You are a professional multilingual resume assistant. Generate a clean, structured resume for the candidate below.

{lang_note}
{jd_prompt}
Candidate Name: {req.name.strip()}

Skills:
{chr(10).join(f"- {s}" for s in expanded_skills)}

Experience:
{req.experience.strip()}

Output Format:
1. Summary
2. Skills
3. Work Experience
4. Education (if any)
5. Certifications (if any)
"""

    def generate_resume(self, req: ResumeGenerationRequest) -> ResumeGenerationResult:
        try:
            logger.info(f"[ResumeGenerator] Generating resume for: {req.name}")

            expanded_skills = self.expand_skills(req.skills)
            prompt = self._build_prompt(req, expanded_skills)

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI resume builder."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=1200,
            )

            content = response.choices[0].message["content"]
            logger.info(f"[ResumeGenerator] Resume generated successfully for: {req.name}")

            return ResumeGenerationResult(
                name=req.name,
                skills=expanded_skills,
                experience=req.experience,
                job_description=req.job_description,
                language=req.language or "en",
                generated_resume=content
            )

        except OpenAIError as e:
            logger.exception(f"[ResumeGenerator] OpenAI API error: {e}")
            return ResumeGenerationResult(
                name=req.name,
                skills=req.skills,
                experience=req.experience,
                job_description=req.job_description,
                error=f"OpenAI error: {str(e)}"
            )
        except Exception as ex:
            logger.exception(f"[ResumeGenerator] Internal failure: {ex}")
            return ResumeGenerationResult(
                name=req.name,
                skills=req.skills,
                experience=req.experience,
                job_description=req.job_description,
                error=f"Internal error: {str(ex)}"
            )
