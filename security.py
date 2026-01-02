"""
Security API Routes
Authentication, authorization, and security monitoring endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import jwt
from pydantic import BaseModel, EmailStr

from app.observability.metrics.prometheus import PrometheusMetrics
from app.observability.logging.logger import setup_logging

router = APIRouter(prefix="/security", tags=["security"])
logger = setup_logging(__name__)
security = HTTPBearer()

# Pydantic models
class UserLogin(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False

class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "user"

@router.post("/login")
async def login(
    request: Request,
    login_data: UserLogin,
    metrics: PrometheusMetrics = Depends(lambda: PrometheusMetrics())
):
    """User authentication endpoint"""
    try:
        client_ip = request.client.host
        logger.info(f"Login attempt for {login_data.email} from {client_ip}")
        
        # Demo authentication
        if login_data.email == "admin@example.com" and login_data.password == "password":
            token = jwt.encode(
                {
                    "sub": login_data.email,
                    "role": "admin",
                    "exp": datetime.utcnow() + timedelta(hours=24)
                },
                "your-secret-key",
                algorithm="HS256"
            )
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "expires_in": 86400,
                "user": {
                    "email": login_data.email,
                    "name": "Admin User",
                    "role": "admin"
                }
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

@router.get("/profile")
async def get_profile(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    metrics: PrometheusMetrics = Depends(lambda: PrometheusMetrics())
):
    """Get user profile"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
        
        return {
            "email": payload["sub"],
            "role": payload["role"],
            "name": "User Name",
            "last_login": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Profile error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile"
        )
