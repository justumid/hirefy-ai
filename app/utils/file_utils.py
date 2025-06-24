import os
import uuid
import shutil
import base64
import tempfile
import logging
from typing import Optional, Tuple, List
from pathlib import Path
from mimetypes import guess_type

from fastapi import UploadFile

logger = logging.getLogger("file_utils")


# === General File Utilities ===

def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def get_mime_type(file_path: str) -> Optional[str]:
    mime, _ = guess_type(file_path)
    return mime


def is_audio_file(filename: str) -> bool:
    return get_file_extension(filename) in [".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"]


def is_resume_file(filename: str) -> bool:
    return get_file_extension(filename) in [".pdf", ".docx", ".doc", ".txt", ".rtf"]


def generate_temp_dir(prefix: str = "tmp") -> str:
    path = tempfile.mkdtemp(prefix=f"{prefix}_")
    logger.debug(f"[TempDir] Created temporary directory: {path}")
    return path


def cleanup_temp_dir(path: str):
    try:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
            logger.info(f"[Cleanup] Deleted temp directory: {path}")
    except Exception as e:
        logger.warning(f"[Cleanup] Failed to delete: {path}, Error: {e}")


# === File Save / Decode ===

def save_temp_file(uploaded_file: UploadFile, suffix: Optional[str] = None) -> str:
    try:
        suffix = suffix or get_file_extension(uploaded_file.filename)
        tmp_dir = generate_temp_dir("upload")
        tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}{suffix}")
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(uploaded_file.file, f)
        logger.info(f"[TempFile] Saved file to: {tmp_path}")
        return tmp_path
    except Exception as e:
        logger.exception(f"[TempFile] Failed to save uploaded file: {e}")
        raise


def decode_base64_file(base64_str: str, suffix: str = ".txt") -> str:
    try:
        tmp_dir = generate_temp_dir("base64")
        file_path = os.path.join(tmp_dir, f"{uuid.uuid4()}{suffix}")
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(base64_str))
        logger.info(f"[Base64] Decoded file to: {file_path}")
        return file_path
    except Exception as e:
        logger.exception(f"[Base64] Decode failed: {e}")
        raise


# === Resume Extraction Helpers ===

def is_valid_resume(file_path: str) -> bool:
    ext = get_file_extension(file_path)
    return ext in [".pdf", ".docx", ".txt", ".rtf", ".doc"] and os.path.exists(file_path)


def list_resume_files(folder: str) -> List[str]:
    resume_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if is_resume_file(file):
                resume_files.append(os.path.join(root, file))
    return resume_files
