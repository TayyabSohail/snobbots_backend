"""Authentication routes for the FastAPI application."""

from fastapi import APIRouter, HTTPException, status, Request, Depends
from fastapi.responses import JSONResponse, RedirectResponse
import logging
import requests
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
from app.supabase.supabase_client import get_supabase_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create router for auth endpoints
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

# Get Supabase client
supabase = get_supabase_client()

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

# Google OAuth Authentication Routes

@auth_router.get(
    "/login/google",
    summary="Start Google OAuth login",
    description="Redirects user to Google OAuth for authentication"
)
async def login_google():
    """Start Google OAuth login process."""
    try:
        redirect_url = f"{settings.frontend_url}/auth/callback" if hasattr(settings, 'frontend_url') else "http://localhost:8000/auth/callback"
        res = supabase.auth.sign_in_with_oauth(
            {
                "provider": "google",
                "options": {"redirect_to": redirect_url},
            }
        )
        return RedirectResponse(res.url)
    except Exception as e:
        logger.error(f"Google OAuth login failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate Google OAuth login"
        )

@auth_router.get(
    "/callback",
    summary="OAuth callback handler",
    description="Handles OAuth callback from Google with authorization code"
)
async def auth_callback(request: Request):
    """Handle OAuth callback from Google."""
    try:
        code = request.query_params.get("code")
        if not code:
            logger.warning("OAuth callback received without authorization code")
            return JSONResponse(
                {"error": "No authorization code in callback"}, 
                status_code=status.HTTP_400_BAD_REQUEST
            )

        session = supabase.auth.exchange_code_for_session({"auth_code": code})
        if not session:
            logger.error("Failed to exchange authorization code for session")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Authentication failed"
            )

        logger.info("OAuth authentication successful")
        return JSONResponse(session)  # Contains access_token, user info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during OAuth callback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during OAuth callback"
        )

@auth_router.get(
    "/me",
    summary="Get current user info",
    description="Protected endpoint to get current authenticated user information"
)
async def me(authorization: str = Depends(lambda request: request.headers.get("Authorization"))):
    """Get current authenticated user information."""
    try:
        if not authorization:
            logger.warning("Protected endpoint accessed without authorization header")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Missing authorization token"
            )

        token = authorization.replace("Bearer ", "")
        resp = requests.get(
            f"{settings.supabase_url}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}", 
                "apikey": settings.supabase_service_role_key
            },
        )
        
        if resp.status_code != 200:
            logger.warning(f"Invalid token provided: {resp.status_code}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid or expired token"
            )

        user_data = resp.json()
        logger.info(f"User info retrieved successfully for user ID: {user_data.get('id')}")
        return user_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error while retrieving user info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving user information"
        )