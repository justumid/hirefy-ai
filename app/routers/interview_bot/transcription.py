import os
import tempfile
import logging
import hashlib
from typing import Tuple, Optional, Dict

from app.models.whisper_wrapper import WhisperTranscriber
from interview_bot.config import config

logger = logging.getLogger("interview_transcription")


class TranscriptionService:
    def __init__(self, model_size: str = None, language: Optional[str] = None):
        self.model_size = model_size or config.WHISPER_MODEL_SIZE
        self.language = language or config.DEFAULT_LANGUAGE
        self.transcriber = WhisperTranscriber(model_size=self.model_size)

        # In-memory transcript cache
        self.checksum_cache: Dict[str, Tuple[str, float]] = {}

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def transcribe_file(self, audio_path: str) -> Tuple[str, float]:
        """
        Transcribe a full audio file. Uses checksum caching.
        """
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            checksum = self.compute_checksum(audio_data)
            if checksum in self.checksum_cache:
                logger.info(f"[Cache Hit] {audio_path} checksum={checksum[:8]}")
                return self.checksum_cache[checksum]

            logger.info(f"[Transcribe File] {audio_path} checksum={checksum[:8]}")
            transcript, confidence = self.transcriber.transcribe(audio_path, language=self.language)

            self.checksum_cache[checksum] = (transcript, confidence)
            logger.info(f"[File Done] {len(transcript)} chars, conf: {confidence:.2f}")
            return transcript, confidence

        except Exception as e:
            logger.exception(f"[Transcription Error] file={audio_path} → {e}")
            return "", 0.0

    def transcribe_bytes(self, audio_bytes: bytes) -> Tuple[str, float]:
        """
        Transcribe audio provided as raw bytes. Uses checksum caching.
        """
        try:
            checksum = self.compute_checksum(audio_bytes)
            if checksum in self.checksum_cache:
                logger.debug(f"[Cache Hit] chunk checksum={checksum[:8]}")
                return self.checksum_cache[checksum]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            transcript, confidence = self.transcriber.transcribe(tmp_path, language=self.language)
            os.remove(tmp_path)

            self.checksum_cache[checksum] = (transcript, confidence)
            logger.info(f"[Stream Chunk] → {len(transcript)} chars, conf: {confidence:.2f}")
            return transcript, confidence

        except Exception as e:
            logger.exception(f"[Transcription Bytes Error] {e}")
            return "", 0.0

    def transcribe_from_microphone(self, duration_sec: int = 10) -> Tuple[str, float]:
        """
        Transcribe local microphone input (for dev testing only).
        """
        try:
            import sounddevice as sd
            import numpy as np
            from scipy.io.wavfile import write

            samplerate = config.AUDIO_SAMPLE_RATE
            logger.info(f"[Mic] Recording {duration_sec}s @ {samplerate}Hz")
            recording = sd.rec(int(duration_sec * samplerate), samplerate=samplerate, channels=1)
            sd.wait()
            audio_np = np.squeeze(recording)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                write(tmp_file.name, samplerate, audio_np)
                tmp_path = tmp_file.name

            transcript, confidence = self.transcriber.transcribe(tmp_path, language=self.language)
            os.remove(tmp_path)
            logger.info(f"[Mic] Transcribed {len(transcript)} chars, conf: {confidence:.2f}")
            return transcript, confidence

        except ImportError:
            logger.warning("[Mic] sounddevice/scipy not installed")
            return "", 0.0
        except Exception as e:
            logger.exception(f"[Mic Transcription Error] {e}")
            return "", 0.0
