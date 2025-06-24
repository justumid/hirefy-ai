# models/audit_fairness.py

import shap
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.base import BaseEstimator

logger = logging.getLogger("audit_fairness")


class FairnessExplainer:
    def __init__(self, model: BaseEstimator, features: pd.DataFrame, sensitive_features: Optional[List[str]] = None):
        self.model = model
        self.features = features
        self.sensitive_features = sensitive_features or []
        self.explainer = shap.Explainer(self.model.predict, features)
        logger.info(f"[FairnessExplainer] Initialized with {len(features.columns)} features")

    def explain_instance(self, instance: Union[pd.Series, Dict], top_n: int = 10) -> Dict:
        """
        Generate a SHAP explanation for a single candidate instance.
        """
        try:
            if isinstance(instance, dict):
                instance = pd.DataFrame([instance])
            elif isinstance(instance, pd.Series):
                instance = pd.DataFrame([instance.to_dict()])

            shap_values = self.explainer(instance)
            top_features = shap_values.abs.max(0).data.argsort()[::-1][:top_n]

            explanation = {
                "prediction": float(self.model.predict(instance)[0]),
                "shap_top_features": [
                    {
                        "feature": self.features.columns[i],
                        "shap_value": float(shap_values.values[0, i])
                    }
                    for i in top_features
                ]
            }
            return explanation

        except Exception as e:
            logger.exception(f"[SHAP Explain Error] {e}")
            return {"error": str(e)}

    def explain_global(self, top_n: int = 15) -> List[Dict]:
        """
        Generate a global importance summary for the entire dataset.
        """
        try:
            shap_values = self.explainer(self.features)
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            top_indices = np.argsort(mean_abs)[::-1][:top_n]

            summary = [
                {
                    "feature": self.features.columns[i],
                    "mean_abs_shap": float(mean_abs[i])
                }
                for i in top_indices
            ]
            return summary
        except Exception as e:
            logger.exception(f"[Global SHAP Error] {e}")
            return []

    def audit_bias(self, labels: List[int], sensitive_column: str, threshold: float = 0.8) -> Dict:
        """
        Perform bias audit using Fairlearn on a single sensitive feature.
        """
        try:
            sensitive_series = self.features[sensitive_column]
            preds = self.model.predict(self.features)

            frame = MetricFrame(
                metrics={"selection_rate": selection_rate},
                y_true=labels,
                y_pred=preds,
                sensitive_features=sensitive_series
            )

            disparity = demographic_parity_difference(y_true=labels, y_pred=preds, sensitive_features=sensitive_series)

            audit_result = {
                "fairness_metric": "selection_rate",
                "group_rates": frame.by_group.to_dict(),
                "overall_rate": frame.overall,
                "parity_difference": float(disparity),
                "is_fair": abs(disparity) <= (1 - threshold)
            }
            logger.info(f"[FairnessAudit] {audit_result}")
            return audit_result
        except Exception as e:
            logger.exception(f"[Fairness Audit Error] {e}")
            return {"error": str(e)}
