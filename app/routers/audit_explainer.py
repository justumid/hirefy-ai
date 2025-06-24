from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import logging

from app.services.audit_explainer_service import AuditExplainerService

router = APIRouter(tags=["Audit & Explainability"])
logger = logging.getLogger("audit_explainer")

audit_service = AuditExplainerService()

# === I/O Schemas ===

class ScoringContext(BaseModel):
    candidate_id: str = Field(..., example="cand_123")
    job_id: str = Field(..., example="job_456")
    semantic_score: float = Field(..., ge=0.0, le=1.0)
    skill_overlap: float = Field(..., ge=0.0, le=1.0)
    psychometric_score: float = Field(..., ge=0.0, le=1.0)
    fairness_score: float = Field(..., ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)
    override_weights: Optional[Dict[str, float]] = Field(None, example={"semantic_score": 0.4})
    language: Optional[str] = Field("en", example="en")

    @validator("override_weights")
    def validate_weights(cls, v):
        if v:
            total = sum(v.values())
            if not (0.95 <= total <= 1.05):
                raise ValueError("override_weights must sum approximately to 1.0")
        return v


class AuditExplanation(BaseModel):
    final_score: float
    explanation: Dict[str, float]  # SHAP-like or weighted contributions
    top_contributors: List[str]
    fairness_flags: Optional[Dict[str, bool]]
    summary: Optional[str] = Field(None, description="Natural language summary of explanation")


# === Endpoint ===

@router.post(
    "/audit/explain_score",
    response_model=AuditExplanation,
    summary="Explain Candidate Scoring Decision",
    description="Returns a breakdown of how final score was computed, including top contributing factors and fairness indicators."
)
def explain_candidate_score(context: ScoringContext):
    try:
        logger.info(f"[Audit] Explaining score for candidate={context.candidate_id} job={context.job_id}")
        result = audit_service.explain_score(context.dict())
        return AuditExplanation(**result)
    except Exception as e:
        logger.exception(f"[Audit] Failed to generate explanation for {context.candidate_id}")
        raise HTTPException(status_code=500, detail=f"Failed to explain score: {str(e)}")
