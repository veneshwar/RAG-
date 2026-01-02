"""
Logging middleware for API
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
import json
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware"""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details"""
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            await self._log_response(request, response, request_id, duration)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            await self._log_error(request, e, request_id, duration)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request"""
        log_data = {
            "event": "request_start",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "timestamp": time.time()
        }
        
        # Add user info if available
        if hasattr(request.state, 'user_id'):
            log_data["user_id"] = request.state.user_id
        
        logger.info(f"Request started: {json.dumps(log_data)}")
    
    async def _log_response(self, request: Request, response: Response, request_id: str, duration: float):
        """Log outgoing response"""
        log_data = {
            "event": "request_complete",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "response_headers": dict(response.headers),
            "timestamp": time.time()
        }
        
        # Add user info if available
        if hasattr(request.state, 'user_id'):
            log_data["user_id"] = request.state.user_id
        
        logger.info(f"Request completed: {json.dumps(log_data)}")
    
    async def _log_error(self, request: Request, error: Exception, request_id: str, duration: float):
        """Log request error"""
        log_data = {
            "event": "request_error",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration_ms": round(duration * 1000, 2),
            "timestamp": time.time()
        }
        
        # Add user info if available
        if hasattr(request.state, 'user_id'):
            log_data["user_id"] = request.state.user_id
        
        logger.error(f"Request failed: {json.dumps(log_data)}")
