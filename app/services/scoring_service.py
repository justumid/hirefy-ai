import logging
import requests
import pandas as pd
from typing import Dict, Optional

from app.base.models import ResumeProfile, JobPosting
from app.base.config import settings
from app.base.scoring_utils import compute_final_score, normalize_score
from app.models.audit_fairness import FairnessExplainer
from app.services.job_matcher_service import JobMatcherService

logger = logging.getLogger("scoring_service")


class ScoringService:
    def __init__(self):
        self.matcher = JobMatcherService()
        self.psychometric_url = settings.PSYCHOMETRIC_API_URL

    def fetch_psychometric_score(self, candidate_id: str) -> float:
        """Fetch psychometric score from external service or fallback to neutral."""
        try:
            url = f"{self.psychometric_url}/score/{candidate_id}"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                score = float(response.json().get("score", 0.5))
                clamped = max(0.0, min(1.0, score))
                logger.info(f"[Psychometric] Score for {candidate_id}: {clamped}")
                return clamped
            else:
                logger.warning(f"[Psychometric] Non-200 response for {candidate_id}: {response.status_code}")
        except Exception as e:
            logger.warning(f"[Psychometric] Exception fetching for {candidate_id}: {e}")
        return 0.5  # fallback neutral

    def explain_score(self, features: Dict[str, float]) -> Dict:
        """Generate SHAP-style explanation dictionary."""
        try:
            dummy_model = lambda x: [sum(features.values())] * len(x)
            df = pd.DataFrame([features])
            explainer = FairnessExplainer(model=dummy_model, features=df)
            explanation = explainer.explain_instance(df.iloc[0].to_dict(), top_n=3)
            logger.debug(f"[Explain] SHAP explanation generated.")
            return explanation
        except Exception as e:
            logger.warning(f"[Explain] Failed: {e}")
            return {
                "shap_top_features": [],
                "reason_list": [],
                "summary": "Explanation unavailable due to error"
            }

    def score(self, request) -> Dict:
        """Main scoring method for candidate-job matching with fairness + psychometrics."""
        try:
            resume: ResumeProfile = request.resume
            job: JobPosting = request.job
            explain: bool = request.explain
            override_weights: Optional[Dict[str, float]] = request.override_weights
            psychometric_score: Optional[float] = request.psychometric_score

            logger.info(f"[Scoring] candidate={resume.candidate_id}, job={job.job_id}")

            # Step 1: Index job temporarily
            self.matcher.index_job(job)

            # Step 2: Run hybrid matcher
            matches = self.matcher.hybrid_match(resume, top_k=1, filter_ids=[job.job_id])
            if not matches:
                raise ValueError(f"No match found for job_id={job.job_id}")
            match = matches[0]

            # Step 3: Remove indexed job
            self.matcher.delete_job(job.job_id)

            # Step 4: Fetch psychometric score
            psychometric = psychometric_score or self.fetch_psychometric_score(resume.candidate_id)

            # Step 5: Normalize fairness-adjusted score
            fairness_score = normalize_score((match.final_score + psychometric) / 2)

            # Step 6: Weighted final score
            final_score = compute_final_score(
                semantic_score=match.semantic_score,
                skill_overlap=match.skill_overlap,
                psychometric_score=psychometric,
                fairness_adjusted_score=fairness_score,
                weights=override_weights
            )

            # Step 7: SHAP explanation
            explanation = None
            if explain:
                explanation = self.explain_score({
                    "semantic_score": match.semantic_score,
                    "skill_overlap": match.skill_overlap,
                    "psychometric_score": psychometric,
                    "fairness_adjusted_score": fairness_score
                })

            # Step 8: Return structured result
            result = {
                "candidate_id": resume.candidate_id,
                "job_id": job.job_id,
                "semantic_score": round(match.semantic_score, 4),
                "skill_overlap": round(match.skill_overlap, 4),
                "psychometric_score": round(psychometric, 4),
                "fairness_score": round(fairness_score, 4),
                "final_score": round(final_score, 4),
                "explanation": explanation
            }

            logger.info(f"[ScoringResult] {resume.candidate_id} → {job.job_id} = {result['final_score']}")
            return result

        except Exception as e:
            cid = getattr(request.resume, "candidate_id", "unknown")
            logger.exception(f"[ScoringService] Failed for candidate={cid}: {e}")
            raise RuntimeError(f"ScoringService failed: {e}")
