# app/routers/copilot_service.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict

from app.services.copilot_service import CopilotService

router = APIRouter(tags=["Copilot Service"])
copilot = CopilotService()

# === Request Models ===

class JDRequest(BaseModel):
    title: str = Field(..., example="Senior Backend Engineer")
    company_name: str = Field(..., example="HirefyAI")
    responsibilities: str
    requirements: str
    soft_skills: Optional[str] = ""
    language: Optional[str] = "en"

class EmailRequest(BaseModel):
    candidate_name: str
    role: str
    company_name: str
    summary_points: str
    email_type: str = Field(..., example="rejection")  # options: interview_invite, offer, rejection
    language: Optional[str] = "en"

class OfferLetterRequest(BaseModel):
    candidate_name: str
    role: str
    company_name: str
    start_date: str
    salary: str
    benefits: Optional[str] = ""
    location: Optional[str] = ""
    language: Optional[str] = "en"

# === Response Model ===

class CopilotResponse(BaseModel):
    result: str

# === Endpoints ===

@router.post("/copilot/generate_jd", response_model=CopilotResponse, summary="Generate Job Description")
def generate_job_description(req: JDRequest):
    try:
        jd = copilot.generate_job_description(req)
        return CopilotResponse(result=jd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/copilot/email", response_model=CopilotResponse, summary="Generate Email to Candidate")
def generate_email(req: EmailRequest):
    try:
        email = copilot.generate_candidate_email(req)
        return CopilotResponse(result=email)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/copilot/offer_letter", response_model=CopilotResponse, summary="Generate Offer Letter")
def generate_offer_letter(req: OfferLetterRequest):
    try:
        letter = copilot.generate_offer_letter(req)
        return CopilotResponse(result=letter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
