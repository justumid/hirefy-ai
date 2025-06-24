import os
from pydantic import BaseSettings, validator, root_validator


class InterviewBotConfig(BaseSettings):
    # === Audio & Whisper Configuration ===
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_BUFFER_SECONDS: int = 5  # Process every 5 seconds
    AUDIO_CHANNELS: int = 1  # Mono audio
    AUDIO_SAMPLE_WIDTH: int = 2  # 16-bit (2 bytes)
    
    # === Buffer Size in Bytes ===
    AUDIO_BUFFER_BYTES: int = None

    # === Streaming Timeouts ===
    STREAMING_TIMEOUT_SECONDS: int = int(os.getenv("STREAMING_TIMEOUT_SECONDS", 120))
    MAX_STREAM_DURATION_MINUTES: int = int(os.getenv("MAX_STREAM_DURATION_MINUTES", 15))

    # === Multilingual LLM Configuration ===
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "uz")  # Default to Uzbek
    SUPPORTED_LANGUAGES: tuple = ("uz", "ru", "en")

    # Whisper model mapping
    WHISPER_MODEL_MAP: dict = {
        "uz": "aisha-org/Whisper-Uzbek",
        "ru": "openai/whisper-large-v3",
        "en": "openai/whisper-medium.en"
    }

    # === LLM Model Preferences ===
    SCORING_MODEL: str = os.getenv("SCORING_MODEL", "gpt-4")
    SUMMARY_MODEL: str = os.getenv("SUMMARY_MODEL", "gpt-4")
    EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "gpt-4")

    # === Thresholds & Validation ===
    MIN_TRANSCRIPTION_CONFIDENCE: float = 0.6
    MIN_ANSWER_LENGTH_CHARS: int = 10
    MAX_ANSWER_LENGTH_CHARS: int = 1000

    # === Logging and Debug Controls ===
    ENABLE_TRANSCRIPTION_LOGGING: bool = True
    ENABLE_STREAM_DEBUG_LOGGING: bool = False

    # === Load from .env file ===
    class Config:
        env_file = ".env"

    # === Auto-calculate derived fields ===
    @root_validator
    def compute_audio_buffer_bytes(cls, values):
        sample_rate = values.get("AUDIO_SAMPLE_RATE", 16000)
        seconds = values.get("AUDIO_BUFFER_SECONDS", 5)
        width = values.get("AUDIO_SAMPLE_WIDTH", 2)
        channels = values.get("AUDIO_CHANNELS", 1)
        values["AUDIO_BUFFER_BYTES"] = sample_rate * seconds * width * channels
        return values

    @validator("DEFAULT_LANGUAGE")
    def validate_language(cls, lang):
        if lang not in ("uz", "ru", "en", "auto"):
            raise ValueError("DEFAULT_LANGUAGE must be one of: uz, ru, en, auto")
        return lang


# === Global Config Instance ===
config = InterviewBotConfig()
