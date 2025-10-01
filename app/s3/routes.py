from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
import json
from app.s3.s3_helper import upload_file_to_s3, list_files_in_s3, get_file_from_s3, generate_presigned_url, delete_file_from_s3
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
    
class FetchRequest(BaseModel):
    chatbot_title: str

class RemoveRequest(BaseModel):
    chatbot_title: str
    filename: str


# ------------------------------------------------------------------------------------------- #
# =========================================================================================== #
# -------------------------------------- UPLOAD APIs ---------------------------------------- #
# =========================================================================================== #
# ------------------------------------------------------------------------------------------- #

# ------------------ FILE UPLOAD ------------------ #
@s3_router.post("/upload/file")
async def upload_file_to_s3_api(
    file: UploadFile = File(...),
    chatbot_title: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        file_bytes = await file.read()
        s3_key = f"{user_id}/{chatbot_title}/files/{file.filename}"

        result = upload_file_to_s3(file_bytes, s3_key, file.content_type)
        if result["status"] == "error":
            raise HTTPException(500, result["message"])

        return {
            "url": result["url"],
            "filename": file.filename,
            "uploaded_by": user_id,
            "chatbot_title": chatbot_title,
            "source": "file"
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ RAW TEXT UPLOAD ------------------ #
@s3_router.post("/upload/raw")
async def upload_raw_to_s3_api(
    request: RawTextRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        s3_key = f"{user_id}/{chatbot_title}/raw/{chatbot_title}.txt"
        file_bytes = request.raw_text.encode("utf-8")

        result = upload_file_to_s3(file_bytes, s3_key, "text/plain")
        if result["status"] == "error":
            raise HTTPException(500, result["message"])

        return {
            "url": result["url"],
            "chatbot_title": chatbot_title,
            "uploaded_by": user_id,
            "source": "raw_text"
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ QA PAIRS UPLOAD ------------------ #
@s3_router.post("/upload/qa")
async def upload_qa_to_s3_api(
    request: QARequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        s3_key = f"{user_id}/{chatbot_title}/qa/{chatbot_title}.json"
        file_bytes = json.dumps(request.qa_pairs, indent=2).encode("utf-8")

        result = upload_file_to_s3(file_bytes, s3_key, "application/json")
        if result["status"] == "error":
            raise HTTPException(500, result["message"])

        return {
            "url": result["url"],
            "chatbot_title": chatbot_title,
            "uploaded_by": user_id,
            "source": "qa_pairs"
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ WEB CRAWLING UPLOAD ------------------ #
@s3_router.post("/upload/crawl")
async def upload_crawl_to_s3_api(
    request: CrawlRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        s3_key = f"{user_id}/{chatbot_title}/crawls/{chatbot_title}.txt"
        file_bytes = request.url.encode("utf-8")

        result = upload_file_to_s3(file_bytes, s3_key, "text/plain")
        if result["status"] == "error":
            raise HTTPException(500, result["message"])

        return {
            "url": result["url"],
            "chatbot_title": chatbot_title,
            "uploaded_by": user_id,
            "saved_url": request.url,
            "source": "web_crawling"
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    
# ------------------------------------------------------------------------------------------- #
# =========================================================================================== #
# -------------------------------------- FETCH APIs ----------------------------------------- #
# =========================================================================================== #
# ------------------------------------------------------------------------------------------- #
    
# ------------------ FETCH FILES ------------------ #
@s3_router.post("/fetch/files")
async def fetch_files_api(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # 🔐 Check API key before fetching
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        prefix = f"{user_id}/{chatbot_title}/files/"
        objects = list_files_in_s3(prefix)
        files = []

        for obj in objects:
            key = obj["key"]
            presigned_url = generate_presigned_url(key, expires_in=3600)
            files.append({
                "filename": key.split("/")[-1],
                "url": presigned_url
            })

        return {"chatbot_title": chatbot_title, "files": files}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ FETCH RAW TEXTS ------------------ #
@s3_router.post("/fetch/raw")
async def fetch_raw_api(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # 🔐 Check API key before fetching
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        prefix = f"{user_id}/{chatbot_title}/raw/"
        objects = list_files_in_s3(prefix)

        raws = []
        for obj in objects:
            key = obj["key"]
            content = get_file_from_s3(key).decode("utf-8")
            raws.append({"filename": key.split("/")[-1], "content": content})

        return {"chatbot_title": chatbot_title, "raw_texts": raws}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ FETCH QA PAIRS ------------------ #
@s3_router.post("/fetch/qa")
async def fetch_qa_api(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # 🔐 Check API key before fetching
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        prefix = f"{user_id}/{chatbot_title}/qa/"
        objects = list_files_in_s3(prefix)

        qa_files = []
        for obj in objects:
            key = obj["key"]
            content = get_file_from_s3(key).decode("utf-8")
            try:
                qa_pairs = json.loads(content)
            except Exception:
                qa_pairs = []
            qa_files.append({"filename": key.split("/")[-1], "qa_pairs": qa_pairs})

        return {"chatbot_title": chatbot_title, "qa_data": qa_files}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ FETCH CRAWLED URLS ------------------ #
@s3_router.post("/fetch/crawl")
async def fetch_crawl_api(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # 🔐 Check API key before fetching
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        prefix = f"{user_id}/{chatbot_title}/crawls/"
        objects = list_files_in_s3(prefix)

        crawls = []
        for obj in objects:
            key = obj["key"]
            content = get_file_from_s3(key).decode("utf-8")
            crawls.append({"filename": key.split("/")[-1], "url": content})

        return {"chatbot_title": chatbot_title, "crawls": crawls}
    except Exception as e:
        raise HTTPException(500, str(e))
    
# ------------------------------------------------------------------------------------------- #
# =========================================================================================== #
# -------------------------------------- REMOVE APIs ---------------------------------------- #
# =========================================================================================== #
# ------------------------------------------------------------------------------------------- #

# ------------------ REMOVE FILE ------------------ #
@s3_router.post("/remove/file")
async def remove_file_api(
    request: RemoveRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # 🔐 Check API key before removing
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        key = f"{user_id}/{chatbot_title}/files/{request.filename}"
        result = delete_file_from_s3(key)

        if result["status"] == "error":
            raise HTTPException(404, result["message"])

        return {"success": True, "removed_file": request.filename}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ REMOVE RAW ------------------ #
@s3_router.post("/remove/raw")
async def remove_raw_api(
    request: RemoveRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        key = f"{user_id}/{chatbot_title}/raw/{request.filename}"
        result = delete_file_from_s3(key)

        if result["status"] == "error":
            raise HTTPException(404, result["message"])

        return {"success": True, "removed_file": request.filename}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ REMOVE QA ------------------ #
@s3_router.post("/remove/qa")
async def remove_qa_api(
    request: RemoveRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        key = f"{user_id}/{chatbot_title}/qa/{request.filename}"
        result = delete_file_from_s3(key)

        if result["status"] == "error":
            raise HTTPException(404, result["message"])

        return {"success": True, "removed_file": request.filename}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ REMOVE CRAWL ------------------ #
@s3_router.post("/remove/crawl")
async def remove_crawl_api(
    request: RemoveRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        key = f"{user_id}/{chatbot_title}/crawls/{request.filename}"
        result = delete_file_from_s3(key)

        if result["status"] == "error":
            raise HTTPException(404, result["message"])

        return {"success": True, "removed_file": request.filename}
    except Exception as e:
        raise HTTPException(500, str(e))