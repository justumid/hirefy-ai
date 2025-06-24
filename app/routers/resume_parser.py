# app/routers/resume_parser.py

import os
import uuid
import time
import logging
from typing import Dict

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.resume_parser_service import ResumeParserService
from app.base.models import ParsedResume

router = APIRouter(tags=["Resume Parser"])
logger = logging.getLogger("resume_parser")

# === Config ===
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 10
TMP_DIR = "/tmp/resumes"
os.makedirs(TMP_DIR, exist_ok=True)

# === Service ===
parser_service = ResumeParserService()

@router.post("/resume_parser/parse", response_model=ParsedResume)
async def parse_resume(file: UploadFile = File(...)) -> ParsedResume:
    """
    Upload a resume file (PDF/DOCX/IMG) and receive structured JSON.
    """
    try:
        start_time = time.time()
        ext = file.filename.split(".")[-1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=400, detail="File too large (>10MB)")

        tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.{ext}")
        with open(tmp_path, "wb") as f:
            f.write(contents)

        logger.info(f"[ResumeParser] File={file.filename}, Size={file_size_mb:.2f}MB, Type={ext}")

        parsed_data: Dict = parser_service.parse_resume(tmp_path)
        os.remove(tmp_path)

        duration = time.time() - start_time
        logger.info(f"[ResumeParser] Parsed successfully in {duration:.2f}s")

        return ParsedResume(**parsed_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[ResumeParser] Failed to parse {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Resume parsing failed. Please try another file.")
