"""Authentication service using Supabase."""

from typing import Dict, Any, Optional
import logging
from app.supabase import get_supabase_client, get_admin_supabase_client
from app.core.config import settings
from .models import RegisterRequest, LoginRequest, UserResponse

logger = logging.getLogger(__name__)


async def ensure_user_in_database(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure user exists in the registered_users table.
    
    Args:
        user_data: Dictionary containing user information
        
    Returns:
        Dictionary with insertion status and user data
    """
    supabase = get_admin_supabase_client()
    
    try:
        # Check if user already exists in registered_users
        response = supabase.table('registered_users') \
            .select('id') \
            .eq('id', user_data['id']) \
            .execute()
        
        # If user doesn't exist, insert them
        if not response.data:
            user_to_insert = {
                'id': user_data['id'],
                'email': user_data['email'],
                'name': user_data['name'],
                'approved': user_data.get('approved', True)
            }
            
            insert_response = supabase.table('registered_users') \
                .insert([user_to_insert]) \
                .execute()
            
            if insert_response.data:
                logger.info(f"User {user_data['email']} inserted into registered_users")
                return {'inserted': True, 'user': insert_response.data[0]}
            else:
                logger.error(f"Failed to insert user {user_data['email']}")
                raise Exception('Failed to add user to database')
        
        logger.info(f"User {user_data['email']} already exists in registered_users")
        return {'inserted': False, 'user': response.data[0]}
        
    except Exception as e:
        logger.error(f"Error ensuring user in database: {str(e)}")
        raise


async def register_user(register_data: RegisterRequest) -> Dict[str, Any]:
    """
    Register a new user with Supabase Auth and add to registered_users table.
    
    Args:
        register_data: User registration data
        
    Returns:
        Dictionary with registration result
    """
    supabase = get_supabase_client()
    try:
        # Sign up user with Supabase Auth
        auth_response = supabase.auth.sign_up({
            'email': register_data.email,
            'password': register_data.password,
            'options': {
                'data': {'name': register_data.name},
            }
        })

        # Handle error if user already exists
        if hasattr(auth_response, 'error') and auth_response.error:
            error_msg = auth_response.error.message.lower()
            if ("already registered" in error_msg or "already exists" in error_msg or "duplicate" in error_msg or "user already" in error_msg):
                logger.info(f"Duplicate registration attempt for {register_data.email}")
                return {'error': 'User with this email already exists. Please log in or reset your password.'}
            logger.error(f"Registration error for {register_data.email}: {error_msg}")
            return {'error': error_msg}

        if auth_response.user is None:
            logger.error(f"Signup failed for {register_data.email}: Unknown error.")
            return {'error': "Signup failed. Unknown error."}

        # Add user to registered_users table
        try:
            # Check if user already exists in registered_users
            supabase_admin = get_admin_supabase_client()
            existing_user_response = (
                supabase_admin.table("registered_users")
                .select("id")
                .eq("email", register_data.email)
                .execute()
            )

            if existing_user_response.data and len(existing_user_response.data) > 0:
                logger.info(f"Duplicate registration attempt for {register_data.email} (already in registered_users)")
                return {"error": "User with this email already exists. Please log in or reset your password."}

            user_result = await ensure_user_in_database({
                'id': auth_response.user.id,
                'email': register_data.email,
                'name': register_data.name,
                'approved': True
            })
            return {
                'success': True,
                'user': user_result['user']
            }
        except Exception as e:
            logger.error(f"User created but failed to add to whitelist: {str(e)}")
            return {'error': 'User created but failed to add to whitelist.'}
    except Exception as e:
        error_msg = str(e).lower()

        # ✅ Case 1: user already exists
        if any(x in error_msg for x in [
            "already registered", "already exists", "duplicate", "user already"
        ]):
            logger.info(f"Duplicate registration attempt for {register_data.email} (exception)")
            return {"error": "User with this email already exists. Please log in or reset your password."}

        # ✅ Case 2: any other unexpected error → log + return real message
        logger.error(f"Unexpected registration error for {register_data.email}: {repr(e)}")
        return {"error": str(e)}

async def login_user(login_data: LoginRequest) -> Dict[str, Any]:
    """
    Login user with Supabase Auth and verify they're in registered_users.
    
    Args:
        login_data: User login data
        
    Returns:
        Dictionary with login result and user data
    """
    supabase = get_supabase_client()
    
    try:
        # Sign in with Supabase Auth
        auth_response = supabase.auth.sign_in_with_password({
            'email': login_data.email,
            'password': login_data.password
        })
        
        if hasattr(auth_response, 'error') and auth_response.error:
            return {'error': auth_response.error.message}
        
        # Check if user exists in registered_users table and get their data
        user_response = (
            supabase.table("registered_users")
            .select("*")
            .eq("email", login_data.email)
            .single()
            .execute()
        )
        
        if not user_response.data:
            return {'error': 'You are not authorized to log in'}
        
        return {
            'success': True,
            'user': user_response.data
        }
    
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return {'error': str(e)}


async def reset_user_password(email: str) -> Dict[str, Any]:
    """
    Reset user password using Supabase Auth.
    
    Args:
        email: User email address
        
    Returns:
        Dictionary with reset result
    """
    supabase = get_supabase_client()
    
    try:
        response = supabase.auth.reset_password_for_email(
            email,
            {
                "redirect_to": f"{settings.frontend_url}/reset-password"
            }
        )
        
        # Supabase client returns a `PostgrestResponse` or AuthResponse type
        if hasattr(response, 'error') and response.error:
            return {'error': response.error.message}
        return {'success': True, 'message': 'Password reset email sent'}
    
    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        return {'error': str(e)}



async def get_user_profile(user_id: str) -> Optional[UserResponse]:
    """
    Get user profile from registered_users table.
    
    Args:
        user_id: User ID
        
    Returns:
        User profile data or None
    """
    supabase = get_admin_supabase_client()
    
    try:
        response = supabase.table('registered_users') \
            .select('*') \
            .eq('id', user_id) \
            .single() \
            .execute()
        
        if response.data:
            return UserResponse(**response.data)
        
        return None
    
    except Exception as e:
        logger.error(f"Error fetching user profile: {str(e)}")
        return None

async def update_user_password(access_token: str, refresh_token: str, new_password: str) -> Dict[str, Any]:
    """
    Update user's password using the Supabase session tokens.
    """
    supabase = get_supabase_client()

    try:
        # Set the authenticated session using both tokens
        supabase.auth.set_session(
            access_token=access_token,
            refresh_token=refresh_token
        )

        # Update the password
        response = supabase.auth.update_user({"password": new_password})

        if getattr(response, "error", None):
            return {"error": response.error.message}

        return {"success": True, "message": "Password updated successfully"}

    except Exception as e:
        return {"error": str(e)}