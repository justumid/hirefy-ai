import os
from functools import lru_cache
from pydantic import BaseSettings, Field


class AppConfig(BaseSettings):
    # === App Metadata ===
    PROJECT_NAME: str = "HirelyAI"
    ENVIRONMENT: str = Field("dev", env="ENVIRONMENT")  # dev, staging, prod
    DEBUG_MODE: bool = Field(True, env="DEBUG_MODE")
    API_VERSION: str = "v1"

    # === Security ===
    API_KEY: str = Field("super-secret-key", env="API_KEY")
    RATE_LIMIT_PER_MINUTE: int = Field(100, env="RATE_LIMIT")
    ENABLE_API_KEY_SECURITY: bool = Field(True, env="ENABLE_API_KEY_SECURITY")

    # === Logging ===
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    ENABLE_JSON_LOGS: bool = Field(False, env="ENABLE_JSON_LOGS")

    @property
    def LOG_LEVEL_NUMERIC(self) -> int:
        import logging
        return getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)

    # === Database (PostgreSQL or SQLite fallback) ===
    DB_HOST: str = Field("localhost", env="DB_HOST")
    DB_PORT: int = Field(5432, env="DB_PORT")
    DB_USER: str = Field("postgres", env="DB_USER")
    DB_PASSWORD: str = Field("postgres", env="DB_PASSWORD")
    DB_NAME: str = Field("hirely_db", env="DB_NAME")

    @property
    def DATABASE_URL(self) -> str:
        if self.DB_HOST == "sqlite":
            return "sqlite:///./test.db"
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # === Redis ===
    REDIS_URL: str = Field("redis://localhost:6379", env="REDIS_URL")

    # === MinIO / S3 ===
    MINIO_ENDPOINT: str = Field("localhost:9000", env="MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = Field("minio", env="MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = Field("minio123", env="MINIO_SECRET_KEY")
    MINIO_BUCKET_NAME: str = Field("uploads", env="MINIO_BUCKET_NAME")

    # === AI/ML Models ===
    DEFAULT_LLM_MODEL: str = Field("gpt-4", env="DEFAULT_LLM_MODEL")
    WHISPER_MODEL: str = Field("openai/whisper-large-v3", env="WHISPER_MODEL")
    UZBEK_MODEL: str = Field("aisha-org/Whisper-Uzbek", env="UZBEK_MODEL")
    SENTENCE_BERT_MODEL: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", env="SBERT_MODEL"
    )

    # === External Services ===
    PSYCHOMETRIC_API_URL: str = Field("http://localhost:8010", env="PSYCHOMETRIC_API_URL")
    FRAUD_SCORING_URL: str = Field("http://localhost:8011", env="FRAUD_SCORING_URL")
    LIMIT_ENGINE_URL: str = Field("http://localhost:8012", env="LIMIT_ENGINE_URL")
    GPT_PROVIDER: str = Field("openai", env="GPT_PROVIDER")

    # === Email SMTP ===
    SMTP_SERVER: str = Field("smtp.mailtrap.io", env="SMTP_SERVER")
    SMTP_PORT: int = Field(587, env="SMTP_PORT")
    SMTP_USER: str = Field("", env="SMTP_USER")
    SMTP_PASSWORD: str = Field("", env="SMTP_PASSWORD")
    DEFAULT_SENDER: str = Field("noreply@hirely.ai", env="DEFAULT_SENDER")

    @property
    def SMTP_ENABLED(self) -> bool:
        return all([self.SMTP_SERVER, self.SMTP_USER, self.SMTP_PASSWORD])

    # === Feature Flags ===
    ENABLE_INTERVIEW_BOT: bool = Field(True, env="ENABLE_INTERVIEW_BOT")
    ENABLE_VOICE_TO_RESUME: bool = Field(True, env="ENABLE_VOICE_TO_RESUME")
    ENABLE_AUDIT_EXPLAINER: bool = Field(True, env="ENABLE_AUDIT_EXPLAINER")
    ENABLE_MULTILINGUAL: bool = Field(True, env="ENABLE_MULTILINGUAL")
    ENABLE_BATCH_SCORING: bool = Field(True, env="ENABLE_BATCH_SCORING")
    ENABLE_PROMETHEUS: bool = Field(True, env="ENABLE_PROMETHEUS")

    # === Environment Shortcuts ===
    @property
    def IS_PROD(self) -> bool:
        return self.ENVIRONMENT.lower() == "prod"

    @property
    def IS_DEV(self) -> bool:
        return self.ENVIRONMENT.lower() == "dev"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> AppConfig:
    return AppConfig()


# Global config instance
settings = get_settings()
