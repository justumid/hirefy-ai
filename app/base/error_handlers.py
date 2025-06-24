from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback

from base.logging_config import setup_logger
from base.metrics import api_exception_counter

logger = setup_logger("error_handler", log_file="errors.log")

def register_exception_handlers(app):
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(f"[HTTPException] {exc.detail} | Path={request.url.path}")
        api_exception_counter.labels(type="http").inc()
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status_code": exc.status_code},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"[ValidationError] Path={request.url.path} | {exc.errors()}")
        api_exception_counter.labels(type="validation").inc()
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid request parameters", "details": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"[UnhandledError] {str(exc)}\n{traceback.format_exc()}")
        api_exception_counter.labels(type="unhandled").inc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(exc)},
        )
