from fastapi import FastAPI, Request, Security, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import JSONResponse

from app.base.config import settings
from app.base.logging_config import app_logger as logger

from app.routers import (
    resume_parser,
    resume_generator,
    job_matcher,
    psychometrics,
    scoring_engine,
    copilot_service,
    audit_explainer,
    embedding_store,
    interview_scheduler,
)
from app.routers.interview_bot import router as interview_bot_router

# --- API key header config ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if settings.ENABLE_API_KEY_SECURITY and key != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key")

# --- FastAPI app instance ---
app = FastAPI(
    title="HirelyAI Unified API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# --- CORS config ---
origins = [
    "http://localhost:3000",     # Local React dev
    "https://hirely.ai",         # Production frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prometheus metrics ---
Instrumentator().instrument(app).expose(app)

# --- Logging middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"📥 {request.method} request to {request.url}")
    response = await call_next(request)
    logger.info(f"📤 Response: {response.status_code} for {request.url}")
    return response

# --- Global exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"🔥 Unhandled exception during request: {request.method} {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# --- API Routers ---
app.include_router(resume_parser.router, prefix="/resume", tags=["Resume"])
app.include_router(resume_generator.router, prefix="/resume", tags=["Resume"])
app.include_router(job_matcher.router, prefix="/match", tags=["Matching"])
app.include_router(psychometrics.router, prefix="/psychometrics", tags=["Psychometrics"])
app.include_router(scoring_engine.router, prefix="/score", tags=["Scoring"])
app.include_router(copilot_service.router, prefix="/copilot", tags=["Copilot"])
app.include_router(audit_explainer.router, prefix="/explain", tags=["Explainability"])
app.include_router(embedding_store.router, prefix="/embeddings", tags=["Embeddings"])
app.include_router(interview_scheduler.router, prefix="/calendar", tags=["Calendar"])
app.include_router(interview_bot_router, prefix="/interview", tags=["Interview"])

# --- System endpoints ---
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}

@app.get("/version", tags=["System"])
def version_check():
    return {
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "llm_model": getattr(settings, "DEFAULT_LLM_MODEL", "unknown"),
        "whisper_model": getattr(settings, "WHISPER_MODEL", "unknown")
    }
