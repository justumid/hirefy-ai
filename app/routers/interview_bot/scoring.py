import logging
from typing import Dict, Optional

from models.gpt_wrapper import GPTScorer

logger = logging.getLogger("interview_scoring")


class LLMAnswerScorer:
    def __init__(self, model_name: str = "gpt-4"):
        self.model = model_name
        self.scorer = GPTScorer(model=model_name)

    def score(self, question: str, answer: str, language: str = "uz") -> Dict:
        logger.info(f"[LLM] Scoring Q: {question[:60]}... A: {answer[:60]}... lang={language}")
        try:
            result = self.scorer.score_answer(question, answer, language=language)
            return {
                "score": result.get("score", 0),
                "reasoning": result.get("reasoning", "LLM provided no reasoning."),
                "tags": result.get("tags", []),
                "model_used": result.get("model_used", self.model),
                "source": "llm"
            }
        except Exception as e:
            logger.warning(f"[LLM Error] {e}")
            return {
                "score": 0,
                "reasoning": f"LLM scoring failed: {e}",
                "tags": [],
                "model_used": self.model,
                "source": "llm-fallback"
            }


class RuleBasedScorer:
    def __init__(self, keywords: Optional[Dict[str, float]] = None):
        self.keywords = keywords or {
            "python": 1.0,
            "django": 1.2,
            "fastapi": 1.1,
            "sql": 0.8,
            "docker": 1.0,
            "kubernetes": 1.3,
            "async": 0.9,
            "rest": 0.7,
            "microservice": 1.0
        }

    def score(self, question: str, answer: str, language: str = "uz") -> Dict:
        answer_lower = answer.lower()
        matched = []
        score = 0.0

        for kw, weight in self.keywords.items():
            if kw in answer_lower:
                score += weight
                matched.append(kw)

        max_possible = sum(self.keywords.values())
        normalized = round(min(score / max_possible * 10, 10), 2)

        logger.debug(f"[Rule] Matched: {matched} → Score: {normalized}")

        return {
            "score": normalized,
            "reasoning": f"Matched keywords: {', '.join(matched)}",
            "tags": matched,
            "model_used": "rule_based",
            "source": "rule"
        }


class InterviewAnswerScorer:
    def __init__(self, use_llm: bool = True, language: str = "uz"):
        self.language = language
        self.llm = LLMAnswerScorer() if use_llm else None
        self.rule = RuleBasedScorer()

    def score(self, question: str, answer: str) -> Dict:
        if self.llm:
            result = self.llm.score(question, answer, language=self.language)
            if result["score"] > 0:
                return result

        fallback_result = self.rule.score(question, answer, language=self.language)
        logger.info(f"[Scoring Fallback] Used rule-based scoring")
        return fallback_result


# === Optional high-level wrappers ===

def score_answer(question: str, answer: str, language: str = "uz") -> Dict:
    scorer = InterviewAnswerScorer(language=language)
    return scorer.score(question, answer)


def summarize_session(prompt: str) -> str:
    try:
        return GPTScorer().get_summary(prompt)
    except Exception as e:
        logger.warning(f"[Session Summary] Failed: {e}")
        return "Summary not available due to processing error."


def extract_skills(prompt: str) -> Dict:
    try:
        return GPTScorer().extract_skills(prompt)
    except Exception as e:
        logger.warning(f"[Skill Extraction] Failed: {e}")
        return {"skills": [], "traits": [], "intent": "unknown"}
