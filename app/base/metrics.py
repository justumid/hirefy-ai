from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time


# === Global Metrics ===

http_request_count = Counter(
    "http_requests_total", "Total number of HTTP requests",
    ["method", "endpoint", "status_code"]
)

http_request_duration = Histogram(
    "http_request_duration_seconds", "Duration of HTTP requests in seconds",
    ["method", "endpoint"]
)

in_progress_requests = Gauge(
    "http_requests_in_progress", "Number of HTTP requests currently being processed"
)

api_exception_counter = Counter(
    "api_exception_count", "Total API exceptions by type",
    ["type"]
)


# === FastAPI Middleware for Metrics ===

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        endpoint = request.url.path

        start_time = time.time()
        in_progress_requests.inc()

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time
            http_request_count.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
            http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
            in_progress_requests.dec()

        return response


# === Metrics Endpoint ===

def metrics_endpoint():
    from fastapi import APIRouter
    router = APIRouter()

    @router.get("/metrics")
    async def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

    return router
