from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.s3.s3_helper import upload_file_to_s3
from app.RAG.auth_utils import get_current_user, get_api_key

s3_router = APIRouter(prefix="/s3", tags=["S3"])

# ------------------ MODELS ------------------ #
class RawTextRequest(BaseModel):
    chatbot_title: str
    raw_text: str

class QARequest(BaseModel):
    chatbot_title: str
    qa_pairs: list[dict]  # [{"question": "...", "answer": "..."}]

class CrawlRequest(BaseModel):
    chatbot_title: str
    url: str

# ------------------ FILE UPLOAD ------------------ #
@s3_router.post("/upload/file")
async def upload_file_to_s3_api(
    file: UploadFile = File(...),
    chatbot_title: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a document file to S3 (stored under user_id/chatbot_title/files/).
    Requires valid Supabase access token and active chatbot API key.
    """
    try:
        user_id = current_user["id"]
        chatbot_title = chatbot_title.lower()

        # Ensure chatbot has an API key
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(
                status_code=403,
                detail=f"No active API key found for chatbot '{chatbot_title}'"
            )

        file_bytes = await file.read()
        s3_key = f"{user_id}/{chatbot_title}/files/{file.filename}"

        result = upload_file_to_s3(file_bytes, s3_key, file.content_type)

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return {
            "url": result["url"],
            "filename": file.filename,
            "uploaded_by": user_id,
            "chatbot_title": chatbot_title,
            "source": "file"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ RAW TEXT UPLOAD ------------------ #
@s3_router.post("/upload/raw")
async def upload_raw_to_s3_api(
    request: RawTextRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Upload raw text to S3 (stored under user_id/chatbot_title/raw/).
    Requires valid Supabase access token and active chatbot API key.
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # Ensure chatbot has an API key
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(
                status_code=403,
                detail=f"No active API key found for chatbot '{chatbot_title}'"
            )

        s3_key = f"{user_id}/{chatbot_title}/raw/{chatbot_title}.txt"
        file_bytes = request.raw_text.encode("utf-8")

        result = upload_file_to_s3(file_bytes, s3_key, "text/plain")

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return {
            "url": result["url"],
            "chatbot_title": chatbot_title,
            "uploaded_by": user_id,
            "source": "raw_text"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ QA PAIRS UPLOAD ------------------ #
@s3_router.post("/upload/qa")
async def upload_qa_to_s3_api(
    request: QARequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Upload QA pairs JSON to S3 (stored under user_id/chatbot_title/qa/).
    Requires valid Supabase access token and active chatbot API key.
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # Ensure chatbot has an API key
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(
                status_code=403,
                detail=f"No active API key found for chatbot '{chatbot_title}'"
            )

        s3_key = f"{user_id}/{chatbot_title}/qa/{chatbot_title}.json"
        file_bytes = str(request.qa_pairs).encode("utf-8")

        result = upload_file_to_s3(file_bytes, s3_key, "application/json")

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return {
            "url": result["url"],
            "chatbot_title": chatbot_title,
            "uploaded_by": user_id,
            "source": "qa_pairs"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ WEB CRAWLING UPLOAD ------------------ #
@s3_router.post("/upload/crawl")
async def upload_crawl_to_s3_api(
    request: CrawlRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Save crawled URL reference to S3 (stored under user_id/chatbot_title/crawls/).
    Does NOT store full page content.
    Requires valid Supabase access token and active chatbot API key.
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # Ensure chatbot has an API key
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(
                status_code=403,
                detail=f"No active API key found for chatbot '{chatbot_title}'"
            )

        s3_key = f"{user_id}/{chatbot_title}/crawls/{chatbot_title}.txt"
        file_bytes = request.url.encode("utf-8")

        result = upload_file_to_s3(file_bytes, s3_key, "text/plain")

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return {
            "url": result["url"],
            "chatbot_title": chatbot_title,
            "uploaded_by": user_id,
            "saved_url": request.url,
            "source": "web_crawling"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))