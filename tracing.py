"""
Distributed tracing middleware for API
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
import logging
from typing import Optional, Dict, Any
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
import os


logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """Distributed tracing middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.tracer = trace.get_tracer(__name__)
        self._setup_opentelemetry()
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing"""
        try:
            # Configure Jaeger exporter
            jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")
            
            exporter = JaegerExporter(
                endpoint=jaeger_endpoint,
                collector_endpoint=jaeger_endpoint,
            )
            
            # Set up trace provider
            trace.set_tracer_provider(TracerProvider())
            tracer_provider = trace.get_tracer_provider()
            
            # Add span processor
            span_processor = BatchSpanProcessor(exporter)
            tracer_provider.add_span_processor(span_processor)
            
            logger.info("OpenTelemetry tracing initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry: {e}")
    
    async def dispatch(self, request: Request, call_next):
        """Process request with tracing"""
        
        # Start span
        span_name = f"{request.method} {request.url.path}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            # Add span attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.scheme", request.url.scheme)
            span.set_attribute("http.host", request.url.hostname)
            span.set_attribute("http.target", request.url.path)
            
            # Add client IP
            if request.client:
                span.set_attribute("http.client_ip", request.client.host)
            
            # Add user agent
            user_agent = request.headers.get("user-agent")
            if user_agent:
                span.set_attribute("http.user_agent", user_agent)
            
            # Add request ID if available
            if hasattr(request.state, 'request_id'):
                span.set_attribute("request.id", request.state.request_id)
            
            # Add user info if available
            if hasattr(request.state, 'user_id'):
                span.set_attribute("user.id", request.state.user_id)
            
            # Record start time
            start_time = time.time()
            
            try:
                # Process request
                response = await call_next(request)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Add response attributes
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.response_time_ms", round(duration * 1000, 2))
                
                # Add trace headers to response
                self._add_trace_headers(response, span)
                
                return response
                
            except Exception as e:
                # Record error in span
                duration = time.time() - start_time
                span.set_attribute("http.response_time_ms", round(duration * 1000, 2))
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
    
    def _add_trace_headers(self, response: Response, span: trace.Span):
        """Add tracing headers to response"""
        try:
            span_context = span.get_span_context()
            
            # Add traceparent header
            traceparent = f"00-{span_context.trace_id:032x}-{span_context.span_id:016x}-01"
            response.headers["traceparent"] = traceparent
            
            # Add tracestate header if available
            if hasattr(span_context, 'trace_state') and span_context.trace_state:
                response.headers["tracestate"] = str(span_context.trace_state)
                
        except Exception as e:
            logger.warning(f"Failed to add trace headers: {e}")


def setup_tracing(app):
    """Setup tracing for FastAPI application"""
    try:
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        
        # Instrument HTTPX (for outgoing requests)
        HTTPXClientInstrumentor.instrument()
        
        logger.info("FastAPI tracing instrumentation completed")
        
    except Exception as e:
        logger.warning(f"Failed to setup FastAPI tracing: {e}")


class TraceContext:
    """Helper class for managing trace context"""
    
    @staticmethod
    def get_current_span() -> Optional[trace.Span]:
        """Get current active span"""
        return trace.get_current_span()
    
    @staticmethod
    def add_event(name: str, attributes: Dict[str, Any] = None):
        """Add event to current span"""
        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes or {})
    
    @staticmethod
    def set_attribute(key: str, value: Any):
        """Set attribute on current span"""
        span = trace.get_current_span()
        if span:
            span.set_attribute(key, value)
    
    @staticmethod
    def set_baggage(key: str, value: str):
        """Set baggage item"""
        try:
            from opentelemetry.baggage import set_baggage
            set_baggage(key, value)
        except ImportError:
            logger.warning("Baggage not available")
