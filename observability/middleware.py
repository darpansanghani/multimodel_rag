from starlette.middleware.base import BaseHTTPMiddleware
from observability.tracer import span

class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        skip_paths = {"/health", "/docs", "/openapi.json", "/observability/status"}
        if request.url.path in skip_paths:
            return await call_next(request)

        attributes = {
            "http.method": request.method,
            "http.path": request.url.path,
        }
        with span(f"http {request.method} {request.url.path}", attributes) as s:
            response = await call_next(request)
            s.set_attribute("http.status_code", str(response.status_code))
            return response
