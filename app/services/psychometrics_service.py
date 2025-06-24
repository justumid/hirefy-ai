import logging
from typing import List, Dict, Any, Optional

from app.models.gpt_wrapper import GPTScorer
from app.base.models import (
    PsychometricsInput,
    PsychometricsResult,
    TraitScore,
    EmbeddingMetadata
)

logger = logging.getLogger("psychometrics_service")


# === Prompt Template ===
PSYCHOMETRIC_PROMPT_TEMPLATE = """
You are a psychometric evaluator based on the Big Five model (OCEAN).
Analyze the candidate’s transcript below and rate the following traits from 0 (very low) to 1 (very high):

- Openness
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism

Also provide:
- A short summary of the personality
- Relevant behavioral signals if available
- Return JSON with keys: openness, conscientiousness, extraversion, agreeableness, neuroticism, summary, traits_text

Transcript:
\"\"\"
{transcript}
\"\"\"
"""


class PsychometricsService:
    def __init__(self, model_name: str = "gpt-4-turbo"):
        self.llm = GPTScorer(model=model_name)

    def evaluate(self, input_data: PsychometricsInput) -> PsychometricsResult:
        logger.info(f"[PsychometricsService] Starting trait analysis for {input_data.candidate_id}")

        try:
            prompt = PSYCHOMETRIC_PROMPT_TEMPLATE.format(transcript=input_data.transcript)

            # === Query LLM for Personality Analysis ===
            response: Dict[str, Any] = self.llm.extract_traits(prompt, language=input_data.language or "en")

            logger.debug(f"[PsychometricsService] Raw LLM response: {response}")

            # === Normalize scores ===
            def to_score(value) -> float:
                try:
                    return round(float(value), 3)
                except Exception:
                    return 0.0

            trait_scores: List[TraitScore] = [
                TraitScore(name="openness", score=to_score(response.get("openness"))),
                TraitScore(name="conscientiousness", score=to_score(response.get("conscientiousness"))),
                TraitScore(name="extraversion", score=to_score(response.get("extraversion"))),
                TraitScore(name="agreeableness", score=to_score(response.get("agreeableness"))),
                TraitScore(name="neuroticism", score=to_score(response.get("neuroticism"))),
            ]

            # === Optional Embedding Metadata for Similarity Matching or SHAP ===
            embedding_metadata = EmbeddingMetadata(
                model_used=self.llm.model_name,
                source="llm_trait_inference",
                raw_text=input_data.transcript[:1000],
                input_language=input_data.language or "auto"
            )

            return PsychometricsResult(
                candidate_id=input_data.candidate_id,
                traits=trait_scores,
                summary=response.get("summary", "No summary provided."),
                traits_text=response.get("traits_text", ""),
                language_used=input_data.language or "auto",
                metadata=embedding_metadata
            )

        except Exception as e:
            logger.exception(f"[PsychometricsService Error] {e}")
            raise RuntimeError(f"Psychometric evaluation failed: {str(e)}")
