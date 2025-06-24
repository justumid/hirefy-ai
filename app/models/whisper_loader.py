"""
Whisper Loader for HirefyAI

This module abstracts over OpenAI Whisper and FasterWhisper implementations.
Supports multilingual transcription for Uzbek, Russian, English, and fallback.
"""

import logging
from typing import Optional, Union, TYPE_CHECKING, Any
from threading import Lock

try:
    import whisper
except ImportError:
    whisper = None

from faster_whisper import WhisperModel as FasterWhisperModel

if TYPE_CHECKING:
    from whisper import Whisper

logger = logging.getLogger("whisper_loader")


class WhisperModelWrapper:
    """
    Unified wrapper around Whisper and FasterWhisper backends.
    """
    def __init__(self, model: Union["Whisper", FasterWhisperModel, Any], backend: str):
        self.model = model
        self.backend = backend

    def transcribe(self, audio_path: str, language: Optional[str] = "auto") -> tuple[str, list[Any]]:
        if self.backend == "openai":
            result = self.model.transcribe(audio_path, language=language)
            return result.get("text", ""), result.get("segments", [])
        elif self.backend == "faster":
            segments, _ = self.model.transcribe(audio_path, language=language, beam_size=5)
            full_text = " ".join([seg.text for seg in segments])
            return full_text, segments
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")


class WhisperModelRegistry:
    """
    Global singleton registry for Whisper models with lazy loading and caching.
    Supports:
        - 🇺🇿 Uzbek: aisha-org/Whisper-Uzbek (FasterWhisper)
        - 🇷🇺 Russian: OpenAI whisper-large-v3
        - 🇬🇧 English: OpenAI whisper-medium.en
        - Auto fallback: OpenAI whisper-base
    """
    _instances: dict[str, WhisperModelWrapper] = {}
    _lock = Lock()

    @classmethod
    def get_model(cls, language: Optional[str] = "auto") -> WhisperModelWrapper:
        key = cls._resolve_model_key(language)

        with cls._lock:
            if key not in cls._instances:
                logger.info(f"[WhisperLoader] Loading model for language='{language}' → resolved='{key}'")
                try:
                    cls._instances[key] = cls._load_model_for_key(key)
                except Exception as e:
                    logger.exception(f"[WhisperLoader] Failed to load model for key='{key}': {e}")
                    raise RuntimeError(f"Whisper model loading failed for language='{language}'") from e

            return cls._instances[key]

    @staticmethod
    def _resolve_model_key(lang: Optional[str]) -> str:
        if not lang:
            return "auto"
        lang = lang.lower()
        if lang in {"uz", "uzb", "uzbek"}:
            return "uz"
        elif lang in {"ru", "rus", "russian"}:
            return "ru"
        elif lang in {"en", "eng", "english"}:
            return "en"
        return "auto"

    @staticmethod
    def _load_model_for_key(key: str) -> WhisperModelWrapper:
        if key == "uz":
            model = FasterWhisperModel(
                model_size_or_path="aisha-org/Whisper-Uzbek",
                compute_type="float16",
                device="cuda" if FasterWhisperModel.is_cuda_available() else "cpu"
            )
            return WhisperModelWrapper(model, backend="faster")

        if not whisper:
            raise ImportError("OpenAI Whisper not installed")

        if key == "ru":
            return WhisperModelWrapper(whisper.load_model("large-v3"), backend="openai")
        elif key == "en":
            return WhisperModelWrapper(whisper.load_model("medium.en"), backend="openai")
        else:
            return WhisperModelWrapper(whisper.load_model("base"), backend="openai")

    @classmethod
    def preload_all(cls):
        """
        Warm up all known models to avoid latency on first request.
        Run at app startup.
        """
        for lang in ["uz", "ru", "en"]:
            try:
                cls.get_model(lang)
            except Exception as e:
                logger.warning(f"[WhisperLoader] Failed to preload model '{lang}': {e}")
