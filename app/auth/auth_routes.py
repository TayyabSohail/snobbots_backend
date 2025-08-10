"""Authentication routes for the FastAPI application."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import logging
from .models import (
    RegisterRequest, 
    LoginRequest, 
    ResetPasswordRequest,
    AuthResponse,
    ErrorResponse
)
from .auth_service import (
    register_user,
    login_user,
    reset_user_password
)

logger = logging.getLogger(__name__)

# Create router for auth endpoints
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Register a new user with email, password, and name"
)
async def register(user_data: RegisterRequest):
    """Register a new user."""
    try:
        result = await register_user(user_data)
        
        if 'error' in result:
            logger.warning(f"Registration failed for {user_data.email}: {result['error']}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
        
        logger.info(f"User {user_data.email} registered successfully")
        return AuthResponse(
            success=True,
            message="User registered successfully. You can now log in."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration"
        )


@auth_router.post(
    "/login",
    response_model=AuthResponse,
    summary="Login user",
    description="Login user with email and password"
)
async def login(user_data: LoginRequest):
    """Login a user."""
    try:
        result = await login_user(user_data)
        
        if 'error' in result:
            logger.warning(f"Login failed for {user_data.email}: {result['error']}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result['error']
            )
        
        logger.info(f"User {user_data.email} logged in successfully")
        return AuthResponse(
            success=True,
            message="Login successful"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )


@auth_router.post(
    "/reset-password",
    response_model=AuthResponse,
    summary="Reset user password",
    description="Send password reset email to user"
)
async def reset_password(reset_data: ResetPasswordRequest):
    """Reset user password."""
    try:
        result = await reset_user_password(reset_data.email)
        
        if 'error' in result:
            logger.warning(f"Password reset failed for {reset_data.email}: {result['error']}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
        
        logger.info(f"Password reset email sent to {reset_data.email}")
        return AuthResponse(
            success=True,
            message="Password reset email sent successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during password reset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during password reset"
        )


@auth_router.get(
    "/health",
    summary="Health check",
    description="Check if the auth service is running"
)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "auth"}