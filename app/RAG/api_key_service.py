from typing import Optional
from app.supabase import get_supabase_client

def validate_api_key(api_key: str) -> Optional[dict]:
    """Validate API key and return user_id and chatbot_title."""
    supabase = get_supabase_client()
    
    result = supabase.table('chatbot_configs').select('user_id, chatbot_title, is_active').eq('api_key', api_key).eq('is_active', True).execute()
    
    if not result.data:
        return None
    
    return {
        'user_id': result.data[0]['user_id'],
        'chatbot_title': result.data[0]['chatbot_title']
    }