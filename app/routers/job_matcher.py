# app/routers/job_matcher.py

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import logging

from app.services.job_matcher_service import JobMatcherService

router = APIRouter(tags=["Job Matcher"])
matcher = JobMatcherService()
logger = logging.getLogger("job_matcher")

# === Request / Response Models ===

class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=30, description="Candidate resume text")
    job_description: str = Field(..., min_length=30, description="Job description")

class BatchMatchRequest(BaseModel):
    resume_texts: List[str] = Field(..., min_items=1, description="List of resumes")
    job_description: str

class ReverseMatchRequest(BaseModel):
    job_description: str
    candidate_resumes: List[Dict[str, str]] = Field(..., example=[
        {"candidate_id": "abc123", "resume_text": "Senior Python developer with 5 years..."}
    ])

class MatchResult(BaseModel):
    score: float
    matched: bool
    explanation: Optional[str] = None
    candidate_id: Optional[str] = None


# === Endpoints ===

@router.post("/match", summary="Match resume to job", response_model=MatchResult)
def match_resume_to_job(req: MatchRequest):
    try:
        logger.info(f"[JobMatch] Matching single resume.")
        result = matcher.match_resume_to_job(req.resume_text, req.job_description)
        return result
    except Exception as e:
        logger.exception("[JobMatch] Failed to match resume.")
        raise HTTPException(status_code=500, detail="Job matching failed. Please try again.")


@router.post("/batch_match", summary="Batch resume matching", response_model=List[MatchResult])
def batch_match_resumes(req: BatchMatchRequest):
    try:
        logger.info(f"[JobMatch] Matching {len(req.resume_texts)} resumes to one job.")
        results = matcher.batch_match_resumes(req.resume_texts, req.job_description)
        return results
    except Exception as e:
        logger.exception("[JobMatch] Batch match error.")
        raise HTTPException(status_code=500, detail="Batch matching failed.")


@router.post("/reverse_match", summary="Match job to multiple candidates", response_model=List[MatchResult])
def reverse_match_job_to_candidates(req: ReverseMatchRequest):
    try:
        logger.info(f"[JobMatch] Reverse matching job to {len(req.candidate_resumes)} candidates.")
        results = matcher.match_job_to_candidates(req.job_description, req.candidate_resumes)
        return results
    except Exception as e:
        logger.exception("[JobMatch] Reverse match error.")
        raise HTTPException(status_code=500, detail="Reverse matching failed.")
