import logging
import json
from typing import Dict, Any

from app.base.utils.interview_templates import (
    SESSION_SUMMARY_TEMPLATE,
    SKILL_EXTRACTION_TEMPLATE,
    PROFILE_EXTRACTION_TEMPLATE,
)

logger = logging.getLogger("gpt_scorer")


class GPTScorer:
    def __init__(self, model: str = "gpt-4"):
        self.llm = GPTWriter(model=model)

    def score_answer(self, question: str, answer: str, language: str = "en") -> Dict[str, Any]:
        prompt = self._build_score_prompt(question, answer, language)
        try:
            response = self.llm.write(prompt)
            return self._parse_score_response(response)
        except Exception as e:
            logger.warning(f"[ScoreFallback] {e}")
            return {
                "score": 0.0,
                "reasoning": "Scoring failed.",
                "tags": [],
                "model_used": "fallback"
            }

    def get_summary(self, answers_block: str, domain: str, level: str, language_name: str = "uz") -> str:
        try:
            summary_prompt = SESSION_SUMMARY_TEMPLATE.format(
                answers=answers_block,
                domain=domain,
                level=level,
                language_name=language_name
            )
            return self.llm.write(summary_prompt)
        except Exception as e:
            logger.warning(f"[SummaryFallback] {e}")
            return "Summary could not be generated."

    def extract_skills(self, full_text: str, job_context: str) -> Dict[str, Any]:
        try:
            prompt = SKILL_EXTRACTION_TEMPLATE.format(
                full_text=full_text,
                job_context=job_context
            )
            response = self.llm.write(prompt)
            return self._parse_skill_response(response)
        except Exception as e:
            logger.warning(f"[SkillExtractFallback] {e}")
            return {
                "skills": [],
                "traits": [],
                "intent": "unknown"
            }

    def extract_profile(self, full_text: str) -> Dict[str, Any]:
        try:
            prompt = PROFILE_EXTRACTION_TEMPLATE.format(full_text=full_text)
            response = self.llm.write(prompt)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"[ProfileExtractFallback] {e}")
            return {
                "expected_salary": None,
                "total_experience": None,
                "notable_projects": [],
                "preferred_stack": [],
                "relocation_interest": None,
                "current_role": None
            }

    # === Internal Prompt Builders ===

    def _build_score_prompt(self, question: str, answer: str, language: str) -> str:
        return f"""
You are a professional interview evaluator.

Question:
{question}

Candidate's Answer:
{answer}

Please rate the answer on a scale from 0.0 to 1.0 and explain why.
Return JSON:
{{
  "score": float,  # between 0 and 1
  "reasoning": "...",
  "tags": ["relevant", "clear", ...]
}}
"""

    def _parse_score_response(self, raw: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw)
            return {
                "score": float(parsed.get("score", 0.0)),
                "reasoning": parsed.get("reasoning", ""),
                "tags": parsed.get("tags", []),
                "model_used": "gpt"
            }
        except Exception as e:
            logger.warning(f"[ParseFail:Score] {e}")
            return {
                "score": 0.0,
                "reasoning": "Invalid JSON format",
                "tags": [],
                "model_used": "gpt"
            }

    def _parse_skill_response(self, raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"[ParseFail:Skills] {e}")
            return {
                "skills": [],
                "traits": [],
                "intent": "unknown"
            }
