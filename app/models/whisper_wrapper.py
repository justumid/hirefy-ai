"""
Whisper Wrapper for HirefyAI

High-level abstraction for audio transcription using Whisper and FasterWhisper.
Supports byte stream or file input, multilingual handling, and confidence scoring.
"""

import os
import hashlib
import torch
import logging
import tempfile
from typing import Tuple, Optional

from app.models.whisper_loader import WhisperModelRegistry
from app.routers.interview_bot.config import config

logger = logging.getLogger("whisper_wrapper")


class WhisperTranscriber:
    def __init__(self, language: str = "auto"):
        """
        Initializes the Whisper model using the registry.
        Automatically resolves the backend and device.
        """
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_wrapper = WhisperModelRegistry.get_model(language=language)

    def _compute_checksum(self, file_path: str) -> str:
        """
        Calculate a SHA256 hash of the audio file to avoid redundant work.
        """
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def transcribe(self, file_path: str, language: Optional[str] = None) -> Tuple[str, float]:
        """
        Transcribe a full audio file using the selected Whisper model.

        Args:
            file_path: Path to the input audio file
            language: Language code or "auto"

        Returns:
            (transcript_text, average_confidence)
        """
        lang = language or self.language or "auto"
        try:
            logger.info(f"[WhisperTranscriber] Transcribing {file_path} | lang={lang}")
            checksum = self._compute_checksum(file_path)
            cache_path = f"/tmp/whisper_cache/{checksum}.txt"

            # Check cache
            if os.path.exists(cache_path):
                logger.info(f"[WhisperTranscriber] Using cached result for {file_path}")
                with open(cache_path, "r", encoding="utf-8") as f:
                    return f.read(), 1.0

            text, segments = self.model_wrapper.transcribe(file_path, language=lang)
            confidences = []

            for segment in segments:
                if hasattr(segment, "avg_logprob"):
                    confidence = min(1.0, max(0.0, 1.0 + float(segment.avg_logprob)))
                else:
                    confidence = 1.0
                confidences.append(confidence)

            avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

            os.makedirs("/tmp/whisper_cache", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)

            return text, avg_conf

        except Exception as e:
            logger.exception(f"[WhisperTranscriber] Error during transcription: {e}")
            return "", 0.0

    def transcribe_from_bytes(self, audio_bytes: bytes, language: Optional[str] = None) -> Tuple[str, float]:
        """
        Transcribe audio directly from byte stream using a temporary file.

        Args:
            audio_bytes: Raw audio file as bytes
            language: Optional language override

        Returns:
            (transcript_text, average_confidence)
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            return self.transcribe(tmp_path, language=language)
        finally:
            os.remove(tmp_path)
