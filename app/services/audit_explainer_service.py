import logging
from typing import Dict

import shap
import pandas as pd
import numpy as np

from base.scoring_utils import compute_final_score
from app.base.models import AuditExplanationResult

logger = logging.getLogger("audit_explainer_service")


class AuditExplainerService:
    def __init__(self):
        """Initialize SHAP explainer with a dummy model."""
        self.model = self._wrap_model()

    def _wrap_model(self):
        def model_fn(X: pd.DataFrame) -> np.ndarray:
            return np.array([
                compute_final_score(
                    semantic_score=row.get("semantic_score", 0),
                    skill_overlap=row.get("skill_overlap", 0),
                    psychometric_score=row.get("psychometric_score", 0.5),
                    fairness_adjusted_score=row.get("fairness_score", 0.5)
                )
                for _, row in X.iterrows()
            ])
        return model_fn

    def explain_score(self, context: Dict) -> AuditExplanationResult:
        """
        Generate SHAP-based explanation for a final score decision.
        """
        try:
            candidate_id = context.get("candidate_id", "unknown")
            logger.info(f"[AuditExplainer] Explaining score for candidate {candidate_id}")

            feature_cols = ["semantic_score", "skill_overlap", "psychometric_score", "fairness_score"]
            features = {k: float(context.get(k, 0.0)) for k in feature_cols}
            df = pd.DataFrame([features])

            explainer = shap.Explainer(self.model, df)
            shap_values = explainer(df)

            top_features = self._extract_top_contributors(shap_values[0], df.columns)
            summary = self._summarize(top_features)

            return AuditExplanationResult(
                final_score=round(context.get("final_score", 0.0), 4),
                explanation=top_features,
                top_contributors=list(top_features.keys())[:3],
                fairness_flags=None,
                summary=summary
            )

        except Exception as e:
            logger.exception("[AuditExplainer] Explanation failed")
            return AuditExplanationResult(
                final_score=context.get("final_score", 0.0),
                explanation={},
                top_contributors=[],
                fairness_flags=None,
                summary="Explanation failed due to internal error"
            )

    def _extract_top_contributors(self, shap_row, feature_names, k: int = 10) -> Dict[str, float]:
        shap_vals = dict(zip(feature_names, shap_row.values))
        sorted_items = sorted(shap_vals.items(), key=lambda item: abs(item[1]), reverse=True)
        return {k: float(v) for k, v in sorted_items[:k]}

    def _summarize(self, top_features: Dict[str, float]) -> str:
        if not top_features:
            return "No dominant contributors identified."
        ranked = sorted(top_features.items(), key=lambda x: -abs(x[1]))
        return f"Top contributors: {', '.join([f'{k} ({round(v, 3)})' for k, v in ranked[:3]])}"
