from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from app.services.psychometrics_service import PsychometricsService

router = APIRouter(prefix="/psychometrics", tags=["Psychometrics"])
logger = logging.getLogger("psychometrics_router")

service = PsychometricsService()

# === Request / Response Models ===

class PsychometricsInput(BaseModel):
    candidate_id: str = Field(..., example="abc123")
    answers: List[str] = Field(..., example=["I prefer working alone", "I like taking the lead"])
    test_type: Optional[str] = Field(default="big5", example="big5")  # e.g., big5, mbti

class PsychometricsResult(BaseModel):
    candidate_id: str
    test_type: str
    traits: Dict[str, float]
    personality_type: Optional[str]
    interpretation: str

# === API Endpoint ===

@router.post("/analyze", response_model=PsychometricsResult)
def analyze_psychometrics(input_data: PsychometricsInput):
    try:
        result = service.analyze(input_data)
        return result
    except Exception as e:
        logger.exception(f"[Psychometrics] Failed to analyze: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze psychometrics")
