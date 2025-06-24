# app/routers/resume_generator.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from app.services.resume_generator_service import ResumeGeneratorService
from app.base.models import ResumeGenerationResult

router = APIRouter(tags=["Resume Generator"])
logger = logging.getLogger("resume_generator")

# === Request / Response Schema ===

class ResumeGenRequest(BaseModel):
    name: str = Field(..., example="John Doe")
    skills: List[str] = Field(..., min_items=1, example=["Python", "FastAPI", "SQL"])
    experience: str = Field(..., example="3 years as backend engineer at Acme Inc.")
    job_description: Optional[str] = Field(None, example="Looking for a senior backend developer skilled in Python.")
    language: Optional[str] = Field("en", example="en", description="Output language for resume")
    expand_skills: Optional[bool] = Field(False, description="Whether to enrich skills using SBERT")

class ResumeGenResponse(BaseModel):
    name: str
    skills: List[str]
    experience: str
    job_description: Optional[str]
    language: str
    resume: str

# === Service Initialization ===
resume_gen_service = ResumeGeneratorService()

# === Endpoint ===

@router.post("/resume_generator/generate", response_model=ResumeGenResponse)
def generate_resume(request: ResumeGenRequest):
    """
    🎯 Generate a structured resume using LLM based on user inputs.
    Optionally expand skills using semantic similarity with SBERT.
    """
    try:
        logger.info(f"[ResumeGen] Start resume generation for: {request.name}")

        skills = request.skills
        if request.expand_skills:
            skills = resume_gen_service.expand_skills(skills)
            logger.info(f"[ResumeGen] Expanded skills: {skills}")

        result: ResumeGenerationResult = resume_gen_service.generate_resume(
            request=request
        )

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        logger.info(f"[ResumeGen] Resume successfully generated for {request.name}")
        return ResumeGenResponse(
            name=result.name,
            skills=result.skills,
            experience=result.experience,
            job_description=result.job_description,
            language=result.language or "en",
            resume=result.generated_resume
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[ResumeGen] Unexpected error during generation: {e}")
        raise HTTPException(status_code=500, detail="Resume generation failed. Please try again later.")
