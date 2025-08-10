"""Pydantic models for authentication request/response validation."""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class RegisterRequest(BaseModel):
    """User registration request model."""
    email: EmailStr
    password: str = Field(..., min_length=6)
    name: str = Field(..., min_length=1, max_length=100)


class LoginRequest(BaseModel):
    """User login request model."""
    email: EmailStr
    password: str


class ResetPasswordRequest(BaseModel):
    """Password reset request model."""
    email: EmailStr


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    name: str
    approved: bool
    created_at: Optional[datetime] = None


class AuthResponse(BaseModel):
    """Authentication response model."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    user: Optional[UserResponse] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None