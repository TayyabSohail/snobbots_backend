from typing import Any, Dict, Optional

def success_response(data: Optional[Dict[str, Any]] = None, message: str = "Success") -> Dict[str, Any]:
    return {
        "success": True,
        "data": data or {},
        "error": None,
        "message": message,
    }

def error_response(error: str, message: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "success": False,
        "data": data or {},
        "error": error,
        "message": message or error,
    }
