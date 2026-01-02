"""
Authentication middleware for API
"""

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import jwt
import os
from typing import Optional


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.security = HTTPBearer(auto_error=False)
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key")
        self.algorithm = "HS256"
    
    async def dispatch(self, request: Request, call_next):
        """Process request and add authentication"""
        
        # Skip auth for health checks and metrics
        if request.url.path in ["/", "/api/v1/health", "/api/v1/metrics"]:
            return await call_next(request)
        
        # Get authorization header
        credentials: Optional[HTTPAuthorizationCredentials] = await self.security(request)
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        try:
            # Verify JWT token
            payload = jwt.decode(
                credentials.credentials,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Add user info to request state
            request.state.user_id = payload.get("sub")
            request.state.user_role = payload.get("role", "user")
            request.state.permissions = payload.get("permissions", [])
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        response = await call_next(request)
        return response


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            if not hasattr(request.state, 'permissions'):
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            if permission not in request.state.permissions:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_role(role: str):
    """Decorator to require specific role"""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            if not hasattr(request.state, 'user_role'):
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            if request.state.user_role != role and request.state.user_role != "admin":
                raise HTTPException(status_code=403, detail="Insufficient role")
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
