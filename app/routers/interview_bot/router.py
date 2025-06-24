import os
import uuid
import shutil
import logging
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from app.services.interview_service import InterviewService
from app.base.models import CandidateAudioInput, InterviewResult

router = APIRouter(prefix="/interview", tags=["Interview Bot"])
logger = logging.getLogger("interview_bot")

interview_service = InterviewService()


# === API Input Schema ===
class InterviewInputMetadata(BaseModel):
    candidate_id: str
    domain: str
    difficulty: Optional[str] = "mixed"
    interview_type: Optional[str] = "mixed"
    language: Optional[str] = "uz"  # default is Uzbek


# === POST: Upload Audio + Process ===
@router.post("/submit", response_model=InterviewResult)
async def submit_interview(
    candidate_id: str = Form(...),
    domain: str = Form(...),
    difficulty: Optional[str] = Form("mixed"),
    interview_type: Optional[str] = Form("mixed"),
    language: Optional[str] = Form("uz"),
    audio_files: List[UploadFile] = File(...)
):
    session_id = str(uuid.uuid4())
    session_dir = f"/tmp/interviews/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    audio_paths = []
    try:
        for i, uploaded in enumerate(audio_files):
            file_path = os.path.join(session_dir, f"q{i+1}.wav")
            with open(file_path, "wb") as out_f:
                shutil.copyfileobj(uploaded.file, out_f)
            audio_paths.append(file_path)

        input_data = CandidateAudioInput(
            candidate_id=candidate_id,
            domain=domain,
            difficulty=difficulty,
            interview_type=interview_type,
            language=language,
            audio_paths=audio_paths
        )

        logger.info(f"[InterviewBot] Received interview for {candidate_id} with {len(audio_paths)} audio files")
        result = await interview_service.process_interview(input_data)
        return result

    except Exception as e:
        logger.exception(f"[InterviewBot] Failed processing interview: {e}")
        raise HTTPException(status_code=500, detail="Interview processing failed")

    finally:
        try:
            shutil.rmtree(session_dir)
        except Exception as cleanup_err:
            logger.warning(f"[InterviewBot] Cleanup failed: {cleanup_err}")
