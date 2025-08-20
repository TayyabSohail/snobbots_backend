"""Authentication service using Supabase."""

from typing import Dict, Any, Optional
import logging
from app.supabase import get_supabase_client, get_admin_supabase_client
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
        # Sign up user with Supabase Auth (no email verification)
        auth_response = supabase.auth.sign_up({
            'email': register_data.email,
            'password': register_data.password,
            'options': {
                'data': {'name': register_data.name},
                'email_confirm': False  # Disable email verification
            }
        })
        
        if auth_response.user is None:
            error_msg = "Signup failed"
            if hasattr(auth_response, 'error') and auth_response.error:
                error_msg = auth_response.error.message
            return {'error': error_msg}
        
        # Add user to registered_users table
        try:
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
        logger.error(f"Registration error: {str(e)}")
        return {'error': str(e)}


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
        user_response = supabase.table('registered_users') \
            .select('*') \
            .eq('email', login_data.email) \
            .maybe_single() \
            .execute()
        
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
        response = supabase.auth.reset_password_email(email)
        
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