import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from typing import Optional, List
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import secrets
import string

from app.RAG.rag_helper import generate_response
from app.RAG.pdf_processor import process_and_index_data
from app.RAG.auth_utils import get_current_user, validate_api_key, get_api_key
from app.RAG.link_finder import get_internal_links
from app.RAG.enums import Theme, Position
from app.RAG.token_tracker import update_tokens, get_user_total_tokens
import asyncio
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright


import certifi
import os

_browser = None
_playwright = None


rag_router = APIRouter(prefix="/rag", tags=["RAG"])
os.environ["SSL_CERT_FILE"] = certifi.where()



# ------------------ MODELS ------------------ #

class CreateChatbotRequest(BaseModel):
    chatbot_title: str



class CreateChatbotRequest(BaseModel):
    chatbot_title: str
    category: str = Field(..., min_length=1, max_length=100)
    language: Optional[str] = None
    description: Optional[str] = None


class UpdateChatbotRequest(BaseModel):
    chatbot_title: str
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None


class AppearanceRequest(BaseModel):
    chatbot_title: str
    bot_title: Optional[str] = None
    theme: Optional[Theme] = None
    primary_color_rgb: Optional[str] = Field(None, pattern=r'^rgb\(\d{1,3},\s*\d{1,3},\s*\d{1,3}\)$|^#[0-9A-Fa-f]{6}$')
    border_radius_px: Optional[int] = Field(None, ge=0, le=50)
    position: Optional[Position] = None


class AppearanceResponse(BaseModel):
    id: str
    user_id: str
    chatbot_title: str
    bot_avatar_url: Optional[str]
    theme: Optional[str]
    primary_color_rgb: Optional[str]
    border_radius_px: Optional[int]
    position: Optional[str]
    created_at: str
    updated_at: str

class QueryRequest(BaseModel):
    query: str
    api_key: str

class ApiKeyRequest(BaseModel):
    api_key: str

class QAPair(BaseModel):
    question: str
    answer: str

class RawTextRequest(BaseModel):
    chatbot_title: str
    raw_text: str

class QARequest(BaseModel):
    chatbot_title: str
    qa_pairs: List[QAPair]

class FileRequest(BaseModel):
    chatbot_title: str
    filename: str
    file_bytes: str
    
class DiscoverRequest(BaseModel):
    url: str

class FetchRequest(BaseModel):
    base_url: str
    endpoint: str
    chatbot_title: str
    
class FlushRequest(BaseModel):
    chatbot_title: str
    
    



# ------------------ CREATE CHATBOT ------------------ #

@rag_router.post("/create-chatbot")
async def create_chatbot_api(
    chatbot_title: str = Form(...),
    category: str = Form(...),
    description: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    avatar: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user),
):
    """Create (or return existing) API key for a chatbot with category and description."""
    user_id = current_user["id"]
    chatbot_title_lower = chatbot_title.lower()

    try:
        from app.supabase import get_admin_supabase_client
        supabase = get_admin_supabase_client()

        # Check if chatbot config exists
        existing_config = (
            supabase.table("chatbot_configs")
            .select("api_key, category, description")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title_lower)
            .execute()
        )

        if existing_config.data:
            # If config exists, fetch appearance data and return
            appearance_data = (
                supabase.table("chatbot_appearance")
                .select("language, bot_avatar_url")
                .eq("user_id", user_id)
                .eq("chatbot_title", chatbot_title_lower)
                .execute()
            )
            
            appearance = appearance_data.data[0] if appearance_data.data else {}

            return {
                "api_key": existing_config.data[0]["api_key"],
                "message": "API key already exists",
                "category": existing_config.data[0].get("category"),
                "description": existing_config.data[0].get("description"),
                "language": appearance.get("language"),
                "bot_avatar_url": appearance.get("bot_avatar_url"),
            }

        # Check if user already has 5 bots
        all_user_bots = (
            supabase.table("chatbot_configs")
            .select("chatbot_title")
            .eq("user_id", user_id)
            .execute()
        )

        current_bot_count = len(all_user_bots.data)
        if current_bot_count >= 5:
            raise HTTPException(
                status_code=403,
                detail=f"You already have {current_bot_count} chatbots. Maximum limit is 5. Please delete a chatbot before creating a new one."
            )

        api_key = "snb_" + "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(32)
        )

        # Handle avatar upload
        bot_avatar_url = None
        if avatar:
            if not avatar.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Avatar must be an image file")
            
            file_content = await avatar.read()
            if len(file_content) > 2 * 1024 * 1024:  # 2MB limit
                raise HTTPException(status_code=400, detail="Avatar file too large. Maximum size is 2MB.")
            
            import base64
            file_extension = avatar.filename.split('.')[-1] if '.' in avatar.filename else 'png'
            base64_data = base64.b64encode(file_content).decode('utf-8')
            bot_avatar_url = f"data:image/{file_extension};base64,{base64_data}"

        # Insert into chatbot_configs
        config_data = {
            "user_id": user_id,
            "chatbot_title": chatbot_title_lower,
            "api_key": api_key,
            "is_active": True,
            "category": category,
            "description": description,
        }
        supabase.table("chatbot_configs").insert(config_data).execute()

        # Insert into chatbot_appearance
        appearance_data = {
            "user_id": user_id,
            "chatbot_title": chatbot_title_lower,
            "language": language,
            "bot_avatar_url": bot_avatar_url,
        }
        supabase.table("chatbot_appearance").insert(appearance_data).execute()

        return {
            "api_key": api_key,
            "message": "API key created successfully",
            "category": category,
            "description": description,
            "language": language,
            "bot_avatar_url": bot_avatar_url,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API key creation failed: {str(e)}")


@rag_router.put("/update-chatbot")
def update_chatbot_api(
    request: UpdateChatbotRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update chatbot category and description."""
    user_id = current_user["id"]
    chatbot_title = request.chatbot_title.lower()

    try:
        from app.supabase import get_admin_supabase_client
        supabase = get_admin_supabase_client()

        # Check if chatbot exists and get current data
        existing = (
            supabase.table("chatbot_configs")
            .select("id, category, description, api_key")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        if not existing.data:
            raise HTTPException(status_code=404, detail="Chatbot not found")

        # Prepare update data - only include fields that are provided
        update_data = {}
        if request.category is not None:
            update_data["category"] = request.category
        if request.description is not None:
            update_data["description"] = request.description

        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Update the chatbot
        result = (
            supabase.table("chatbot_configs")
            .update(update_data)
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        # Get updated data
        updated = (
            supabase.table("chatbot_configs")
            .select("category, description, api_key")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        return {
            "message": "Chatbot updated successfully",
            "api_key": updated.data[0]["api_key"],
            "category": updated.data[0]["category"],
            "description": updated.data[0]["description"],
            "updated_fields": list(update_data.keys())
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot update failed: {str(e)}")


# ------------------ APPEARANCE MANAGEMENT ------------------ #
# DEPRECATED: /create-appearance merged with /create-chatbot (appearance is created during chatbot creation)
# DEPRECATED: /update-appearance - no longer needed

# @rag_router.post("/create-appearance")
# async def create_appearance(
#     chatbot_title: str = Form(...),
#     theme: Optional[Theme] = Form(None),
#     primary_color_rgb: Optional[str] = Form(None),
#     border_radius_px: Optional[int] = Form(None),
#     position: Optional[Position] = Form(None),
#     current_user: dict = Depends(get_current_user),
# ):
#     """Sets non-avatar appearance settings. If settings exist, they are updated. If not, they are created."""
#     user_id = current_user["id"]
#     chatbot_title = chatbot_title.lower()
#
#     try:
#         from app.supabase import get_admin_supabase_client
#         supabase = get_admin_supabase_client()
#
#         # Check if chatbot config exists first
#         chatbot_exists = supabase.table("chatbot_configs").select("id").eq("user_id", user_id).eq("chatbot_title", chatbot_title).execute()
#         if not chatbot_exists.data:
#             raise HTTPException(status_code=404, detail="Chatbot not found")
#
#         # Prepare data for update/insert
#         update_data = {}
#         if theme is not None:
#             update_data["theme"] = theme.value
#         if primary_color_rgb is not None:
#             update_data["primary_color_rgb"] = primary_color_rgb
#         if border_radius_px is not None:
#             update_data["border_radius_px"] = border_radius_px
#         if position is not None:
#             update_data["position"] = position.value
#
#         if not update_data:
#             raise HTTPException(status_code=400, detail="No fields to update")
#
#         # Check if appearance record exists
#         existing_appearance = supabase.table("chatbot_appearance").select("id").eq("user_id", user_id).eq("chatbot_title", chatbot_title).execute()
#
#         if existing_appearance.data:
#             # Update existing appearance record
#             (supabase.table("chatbot_appearance").update(update_data).eq("user_id", user_id).eq("chatbot_title", chatbot_title).execute())
#             message = "Appearance settings updated successfully."
#         else:
#             # This case is for older bots made before the logic change
#             update_data["user_id"] = user_id
#             update_data["chatbot_title"] = chatbot_title
#             (supabase.table("chatbot_appearance").insert(update_data).execute())
#             message = "Appearance settings created successfully."
#
#         return {
#             "message": message,
#             "updated_fields": list(update_data.keys())
#         }
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Appearance update failed: {str(e)}")


# @rag_router.put("/update-appearance")
# async def update_appearance(
#     chatbot_title: str = Form(...),
#     avatar: Optional[UploadFile] = File(None),
#     theme: Optional[Theme] = Form(None),
#     primary_color_rgb: Optional[str] = Form(None),
#     border_radius_px: Optional[int] = Form(None),
#     position: Optional[Position] = Form(None),
#     current_user: dict = Depends(get_current_user),
# ):
#     """Update chatbot appearance settings."""
#     user_id = current_user["id"]
#     chatbot_title = chatbot_title.lower()
#
#     try:
#         from app.supabase import get_admin_supabase_client
#         supabase = get_admin_supabase_client()
#
#         # Check if chatbot exists
#         chatbot_exists = (
#             supabase.table("chatbot_configs")
#             .select("id")
#             .eq("user_id", user_id)
#             .eq("chatbot_title", chatbot_title)
#             .execute()
#         )
#
#         if not chatbot_exists.data:
#             raise HTTPException(status_code=404, detail="Chatbot not found")
#
#         # Check if appearance exists
#         existing = (
#             supabase.table("chatbot_appearance")
#             .select("id")
#             .eq("user_id", user_id)
#             .eq("chatbot_title", chatbot_title)
#             .execute()
#         )
#
#         if not existing.data:
#             raise HTTPException(status_code=404, detail="Appearance settings not found. Use create-appearance first.")
#
#         # Handle avatar upload if provided
#         bot_avatar_url = None
#         if avatar:
#             # Validate file type
#             if not avatar.content_type.startswith('image/'):
#                 raise HTTPException(status_code=400, detail="Avatar must be an image file")
#             
#             # Validate file size (max 2MB)
#             file_content = await avatar.read()
#             if len(file_content) > 2 * 1024 * 1024:  # 2MB limit
#                 raise HTTPException(status_code=400, detail="Avatar file too large. Maximum size is 2MB.")
#             
#             # Convert to base64 and store in database
#             import base64
#             file_extension = avatar.filename.split('.')[-1] if '.' in avatar.filename else 'png'
#             base64_data = base64.b64encode(file_content).decode('utf-8')
#             bot_avatar_url = f"data:image/{file_extension};base64,{base64_data}"
#
#         # Prepare update data - only include fields that are provided
#         update_data = {}
#         
#         if bot_avatar_url is not None:
#             update_data["bot_avatar_url"] = bot_avatar_url
#         if theme is not None:
#             update_data["theme"] = theme.value
#         if primary_color_rgb is not None:
#             update_data["primary_color_rgb"] = primary_color_rgb
#         if border_radius_px is not None:
#             update_data["border_radius_px"] = border_radius_px
#         if position is not None:
#             update_data["position"] = position.value
#
#         if not update_data:
#             raise HTTPException(status_code=400, detail="No fields to update")
#
#         # Update appearance
#         result = (
#             supabase.table("chatbot_appearance")
#             .update(update_data)
#             .eq("user_id", user_id)
#             .eq("chatbot_title", chatbot_title)
#             .execute()
#         )
#
#         return {
#             "message": "Appearance updated successfully",
#             "updated_fields": list(update_data.keys())
#         }
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Appearance update failed: {str(e)}")


# @rag_router.get("/appearance/{chatbot_title}")
# def get_appearance(
#     chatbot_title: str,
#     current_user: dict = Depends(get_current_user),
# ):
#     """Get current chatbot appearance settings."""
#     user_id = current_user["id"]
#     chatbot_title = chatbot_title.lower()

#     try:
#         from app.supabase import get_admin_supabase_client
#         supabase = get_admin_supabase_client()

#         result = (
#             supabase.table("chatbot_appearance")
#             .select("*")
#             .eq("user_id", user_id)
#             .eq("chatbot_title", chatbot_title)
#             .execute()
#         )

#         if not result.data:
#             # Return default values if no appearance settings exist
#             return {
#                 "chatbot_title": chatbot_title,
#                 "bot_avatar_url": None,
#                 "theme": None,
#                 "primary_color_rgb": None,
#                 "border_radius_px": None,
#                 "position": None,
#                 "message": "No appearance settings found - using defaults"
#             }

#         return AppearanceResponse(**result.data[0])

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch appearance: {str(e)}")


@rag_router.post("/get-appearance")
def get_appearance_public(request: ApiKeyRequest):
    """Get chatbot appearance settings using API key (no authentication required)."""
    api_data = validate_api_key(request.api_key)
    if not api_data:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    user_id = api_data["user_id"]
    chatbot_title = api_data["chatbot_title"].lower()

    try:
        from app.supabase import get_admin_supabase_client
        supabase = get_admin_supabase_client()

        result = (
            supabase.table("chatbot_appearance")
            .select("*")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        if not result.data:
            # Return default values if no appearance settings exist
            return {
                "chatbot_title": chatbot_title,
                "bot_avatar_url": None,
                "theme": None,
                "primary_color_rgb": None,
                "border_radius_px": None,
                "position": None,
                "message": "No appearance settings found - using defaults"
            }

        # Return all fields from the table
        return result.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch appearance: {str(e)}")


# ------------------ DOCS SEPARATED ------------------ #
# DEPRECATED: These endpoints have been replaced by S3 upload endpoints
# /rag/docs/file → /s3/upload/file
# /rag/docs/raw → /s3/upload/raw
# /rag/docs/qa → /s3/upload/qa

# @rag_router.post("/docs/file")
# def docs_file(
#     file: UploadFile = File(...),
#     chatbot_title: str = Form(...),
#     current_user: dict = Depends(get_current_user),
# ):
#     """Upload a document file (.pdf/.docx/.txt) and index it into the chatbot."""
#     user_id = current_user["id"]
#
#     chatbot_title = chatbot_title.lower()
#     api_key = get_api_key(user_id, chatbot_title)
#     if not api_key:
#         raise HTTPException(
#             status_code=403,
#             detail=f"No active API key found for chatbot '{chatbot_title}'"
#         )
#
#     if not file.filename.lower().endswith((".pdf", ".docx", ".txt")):
#         raise HTTPException(
#             status_code=400, detail="Only .pdf, .docx, and .txt files are supported"
#         )
#
#     file_bytes = file.file.read()
#     filename = file.filename
#
#     result = process_and_index_data(
#         user_id=user_id,
#         filename=filename,
#         file_bytes=file_bytes,
#         chatbot_title=chatbot_title,
#     )
#
#     # Save tokens to database (we already have the value from result)
#     update_tokens(
#         user_id=user_id,
#         chatbot_title=chatbot_title,
#         operation_type="file_upload",
#         tokens_used=result["tokens_used"]
#     )
#
#     return {
#         "message": f"File '{filename}' processed successfully",
#         "chunks_indexed": result["chunks_indexed"],
#         "tokens_used": result["tokens_used"],
#         "api_key": api_key,
#     }


# @rag_router.post("/docs/raw")
# def upload_raw_text(request: RawTextRequest, current_user: dict = Depends(get_current_user)):
#     """Upload and index raw text input."""
#     user_id = current_user["id"]
#     chatbot_title = request.chatbot_title.lower()
#
#     api_key = get_api_key(user_id, chatbot_title)
#     if not api_key:
#         raise HTTPException(status_code=403, detail=f"No active API key found for chatbot '{chatbot_title}'")
#
#     result = process_and_index_data(
#         user_id=user_id,
#         raw_text=request.raw_text,
#         chatbot_title=chatbot_title,
#     )
#
#     # Save tokens to database (we already have the value from result)
#     update_tokens(
#         user_id=user_id,
#         chatbot_title=chatbot_title,
#         operation_type="raw_text",
#         tokens_used=result["tokens_used"]
#     )
#
#     return result


# @rag_router.post("/docs/qa")
# def upload_qa_pairs(request: QARequest, current_user: dict = Depends(get_current_user)):
#     """Upload and index QA pairs."""
#     user_id = current_user["id"]
#     chatbot_title = request.chatbot_title.lower()
#
#     api_key = get_api_key(user_id, chatbot_title)
#     if not api_key:
#         raise HTTPException(status_code=403, detail=f"No active API key found for chatbot '{chatbot_title}'")
#
#     qa_data = [{"question": qa.question, "answer": qa.answer} for qa in request.qa_pairs]
#
#     result = process_and_index_data(
#         user_id=user_id,
#         qa_json=qa_data,
#         chatbot_title=chatbot_title,
#     )
#
#     # Save tokens to database (we already have the value from result)
#     update_tokens(
#         user_id=user_id,
#         chatbot_title=chatbot_title,
#         operation_type="qa_pairs",
#         tokens_used=result["tokens_used"]
#     )
#
#     return result


# ------------------ WEB CRAWLING ------------------ #

@rag_router.post("/crawl/discover")
def discover_links(request: DiscoverRequest, current_user: dict = Depends(get_current_user)):
    """Discover all internal endpoints from the given website."""
    if not current_user or "id" not in current_user:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized user")

    endpoints = get_internal_links(request.url)
    return {"base_url": request.url, "endpoints": endpoints}



@rag_router.on_event("startup")
async def startup_event():
    """Launch a single global Playwright browser asynchronously."""
    global _playwright, _browser
    os.environ["SSL_CERT_FILE"] = certifi.where()
    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(headless=True)
    print("✅ Playwright browser started globally.")


@rag_router.on_event("shutdown")
async def shutdown_event():
    """Close the global browser on app shutdown."""
    global _playwright, _browser
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()
    print("🧹 Playwright browser closed.")


# DEPRECATED: Public endpoint removed - use /s3/upload/crawl instead for batch processing
# This function is still used internally by /s3/upload/crawl
async def fetch_and_index(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Fetch a JS-rendered webpage using global async Playwright instance
    → Extract structured text
    → Batch embed + index once (optimized for performance)
    
    NOTE: This is now an internal function used by /s3/upload/crawl.
    Use /s3/upload/crawl endpoint for crawling multiple URLs.
    """
    global _browser
    if not _browser:
        raise HTTPException(status_code=500, detail="Playwright browser not initialized.")

    os.environ["SSL_CERT_FILE"] = certifi.where()

    user_id = current_user["id"]
    chatbot_title = request.chatbot_title.lower()

    # ✅ Validate API key
    api_key = get_api_key(user_id, chatbot_title)
    if not api_key:
        raise HTTPException(
            status_code=403,
            detail=f"No active API key found for chatbot '{chatbot_title}'"
        )

    full_url = urljoin(request.base_url, request.endpoint)

    try:
        page = await _browser.new_page()
        await page.goto(full_url, wait_until="domcontentloaded", timeout=90000)
        await page.wait_for_selector("body", timeout=15000)
        html_content = await page.content()
        await page.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or render {full_url}: {str(e)}")

    # 🧩 Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    grouped_chunks = []
    current_heading = None
    current_block = []

    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ["h1", "h2", "h3", "h4"]:
            if current_heading or current_block:
                grouped_chunks.append({
                    "heading": current_heading,
                    "content": " ".join(current_block).strip()
                })
                current_block = []
            current_heading = text
        else:
            current_block.append(text)

    if current_heading or current_block:
        grouped_chunks.append({
            "heading": current_heading,
            "content": " ".join(current_block).strip()
        })

    if not grouped_chunks:
        raise HTTPException(status_code=400, detail=f"No meaningful structured text found on {full_url}")

    # ✅ Combine all chunks before embedding/indexing
    texts_to_index = []
    previews = []

    for block in grouped_chunks:
        combined_text = (
            f"{block['heading']}\n{block['content']}" if block["heading"] else block["content"]
        )
        texts_to_index.append(combined_text)
        previews.append({
            "heading": block["heading"],
            "preview": combined_text[:120],
        })

    # ✅ Single embedding/indexing call for all chunks
    try:
        result = process_and_index_data(
            user_id=user_id,
            raw_text="\n\n".join(texts_to_index),
            filename=request.endpoint.strip("/"),
            source_type="web_crawling",
            chatbot_title=chatbot_title,
        )

        update_tokens(
            user_id=user_id,
            chatbot_title=chatbot_title,
            operation_type="web_crawl",
            tokens_used=result["tokens_used"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    return {
        "base_url": request.base_url,
        "endpoint": request.endpoint,
        "blocks_extracted": len(grouped_chunks),
        "chunks_indexed": result["chunks_indexed"],
        "tokens_used": result["tokens_used"],
        "indexed_blocks": previews,
        "message": "✅ Crawled and indexed successfully in a single pass.",
    }
    
    
# ------------------ ASK ------------------ #

@rag_router.post("/ask")
async def ask(request: QueryRequest):
    """Ask questions using API key (no authentication required)."""
    api_data = validate_api_key(request.api_key)
    if not api_data:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    user_id = api_data["user_id"]
    chatbot_title = api_data["chatbot_title"].lower()

    # Call rag_helper
    full_text, usage = generate_response(request.query, user_id, chatbot_title)

    # We already have the token values from the LLM response
    total_tokens = usage.get("total_tokens", 0)

    # Save tokens to database (we already have the value from usage)
    update_tokens(
        user_id=user_id,
        chatbot_title=chatbot_title,
        operation_type="ask_query",
        tokens_used=total_tokens
    )

    return JSONResponse({
        "answer": full_text,
        "tokens_used": total_tokens,
    })


# ------------------ TOKEN TRACKING ------------------ #

@rag_router.get("/tokens")
def get_all_user_tokens(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    
    summary = get_user_total_tokens(user_id)
    
    if "error" in summary:
        raise HTTPException(status_code=500, detail=summary["error"])
    
    return summary


# ------------------ FLUSH ------------------ #

@rag_router.post("/flush")
def flush_namespace(
    request: FlushRequest,
    current_user: dict = Depends(get_current_user)
):
    """Flush all vectors for a chatbot's namespace."""
    user_id = current_user["id"]
    chatbot_title = request.chatbot_title.lower()
    namespace = chatbot_title.strip().replace(" ", "_")

    INDEX_NAME = f"snobbots-{user_id.lower().replace(' ', '_')}"

    try:
        from app.RAG.pdf_processor import pc  # reuse Pinecone client

        if INDEX_NAME not in pc.list_indexes().names():
            raise HTTPException(status_code=404, detail=f"Index '{INDEX_NAME}' not found")

        index = pc.Index(INDEX_NAME)

        # delete all vectors in namespace
        index.delete(delete_all=True, namespace=namespace)

        return {
            "message": f"Namespace '{namespace}' flushed successfully from index '{INDEX_NAME}'",
            "namespace": namespace,
            "index_name": INDEX_NAME
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flush failed: {str(e)}")
    

# ------------------ Get All Chatbots ------------------ #
@rag_router.get("/all-chatbots")
def get_user_chatbots(current_user: dict = Depends(get_current_user)):
    """Get all chatbots of the current user with their details, token usage, and query counts."""
    user_id = current_user["id"]

    try:
        from app.supabase import get_admin_supabase_client
        supabase = get_admin_supabase_client()

        # Fetch all chatbots for this user
        chatbots_response = (
            supabase.table("chatbot_configs")
            .select("chatbot_title, api_key, is_active, category, description, created_at, updated_at")
            .eq("user_id", user_id)
            .execute()
        )
        
        chatbots_data = chatbots_response.data

        if not chatbots_data:
            return {
                "total_count": 0,
                "total_queries_all_bots": 0,
                "chatbots": []
            }

        chatbot_titles = [bot["chatbot_title"] for bot in chatbots_data]

        # Fetch appearance data for all chatbots
        appearance_response = (
            supabase.table("chatbot_appearance")
            .select("chatbot_title, language, bot_avatar_url")
            .in_("chatbot_title", chatbot_titles)
            .eq("user_id", user_id)
            .execute()
        )
        
        appearance_map = {item["chatbot_title"]: item for item in appearance_response.data}

        # Fetch token usage and query count summary for each bot
        from app.RAG.token_tracker import get_user_total_tokens
        token_summary = get_user_total_tokens(user_id)
        
        if "error" in token_summary:
            # If there's an error fetching token data, proceed without it
            token_data = {}
            total_queries_all_bots = 0
        else:
            token_data = token_summary.get("bots", {})
            total_queries_all_bots = token_summary.get("total_queries_all_bots", 0)

        # Attach token count, query count, and appearance data per bot
        chatbot_list = []
        for bot in chatbots_data:
            chatbot_title = bot["chatbot_title"]
            appearance = appearance_map.get(chatbot_title, {})
            bot_token_data = token_data.get(chatbot_title, {})
            
            chatbot_list.append({
                **bot,
                "language": appearance.get("language"),
                "bot_avatar_url": appearance.get("bot_avatar_url"),
                "total_tokens_used": bot_token_data.get("total_tokens", 0),
                "query_count": bot_token_data.get("query_count", 0),
                "token_breakdown": bot_token_data.get("operations", {})
            })

        return {
            "total_count": len(chatbot_list),
            "total_queries_all_bots": total_queries_all_bots,
            "chatbots": chatbot_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chatbots: {str(e)}")
