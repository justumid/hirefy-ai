# app/routers/scoring_engine.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict
import logging

from app.services.scoring_service import ScoringService
from app.base.models import ResumeProfile, JobPosting, ScoreExplanation

router = APIRouter(tags=["Scoring Engine"])
logger = logging.getLogger("scoring_engine")

# === Request / Response Models ===

class ScoringRequest(BaseModel):
    resume: ResumeProfile = Field(..., description="Candidate resume data")
    job: JobPosting = Field(..., description="Job posting data")
    psychometric_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional psychometric score [0-1]")
    explain: Optional[bool] = Field(False, description="Whether to include SHAP explanations")
    override_weights: Optional[Dict[str, float]] = Field(
        None,
        example={"semantic": 0.5, "skill_overlap": 0.3, "psychometric": 0.2},
        description="Override default scoring weights"
    )

class ScoringResponse(BaseModel):
    candidate_id: str
    job_id: str
    semantic_score: float
    skill_overlap: float
    psychometric_score: float
    fairness_score: float
    final_score: float
    explanation: Optional[ScoreExplanation]


# === Endpoint ===

@router.post(
    "/score",
    summary="Score a candidate for a job",
    response_model=ScoringResponse,
    response_description="Candidate-job match score with optional explanation"
)
def score_candidate(request: ScoringRequest):
    """
    Combines semantic similarity, skill overlap, psychometric score,
    and fairness-adjusted final score. Optionally returns SHAP-based explanations.
    """
    try:
        logger.info(f"[ScoreRequest] candidate_id={request.resume.candidate_id} job_id={request.job.job_id}")
        result = ScoringService.score(request)
        logger.info(f"[ScoreResult] final_score={result.final_score:.4f}")
        return result
    except ValueError as ve:
        logger.warning(f"[ValidationError] {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception(f"[ScoringError] candidate_id={request.resume.candidate_id} job_id={request.job.job_id}")
        raise HTTPException(status_code=500, detail="Internal scoring failure.")
