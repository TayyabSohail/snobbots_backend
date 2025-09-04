from fastapi import Header, HTTPException
from app.supabase import get_supabase_client
import requests

def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.split(" ")[1]
    supabase = get_supabase_client()

    resp = requests.get(
        f"{supabase.supabase_url}/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": supabase.supabase_key}
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid token")

    return resp.json()