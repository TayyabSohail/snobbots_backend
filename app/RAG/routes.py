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

rag_router = APIRouter(prefix="/rag", tags=["RAG"])



# ------------------ MODELS ------------------ #

class CreateChatbotRequest(BaseModel):
    chatbot_title: str



class CreateChatbotRequest(BaseModel):
    chatbot_title: str
    category: str = Field(..., min_length=1, max_length=100)
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
def create_chatbot_api(
    request: CreateChatbotRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create (or return existing) API key for a chatbot with category and description."""
    user_id = current_user["id"]
    chatbot_title = request.chatbot_title.lower()

    try:
        from app.supabase import get_admin_supabase_client
        supabase = get_admin_supabase_client()

        existing = (
            supabase.table("chatbot_configs")
            .select("api_key, category, description")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        if existing.data:
            return {
                "api_key": existing.data[0]["api_key"], 
                "message": "API key already exists",
                "category": existing.data[0].get("category"),
                "description": existing.data[0].get("description")
            }

        # Check if user already has 5 bots (allow 1-5, block at 6+)
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

        supabase.table("chatbot_configs").insert({
            "user_id": user_id,
            "chatbot_title": chatbot_title,
            "api_key": api_key,
            "is_active": True,
            "category": request.category,
            "description": request.description,
        }).execute()

        return {
            "api_key": api_key, 
            "message": "API key created successfully",
            "category": request.category,
            "description": request.description
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

@rag_router.post("/create-appearance")
async def create_appearance(
    chatbot_title: str = Form(...),
    avatar: Optional[UploadFile] = File(None),
    theme: Optional[Theme] = Form(None),
    primary_color_rgb: Optional[str] = Form(None),
    border_radius_px: Optional[int] = Form(None),
    position: Optional[Position] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    """Create chatbot appearance settings."""
    user_id = current_user["id"]
    chatbot_title = chatbot_title.lower()

    try:
        from app.supabase import get_admin_supabase_client
        supabase = get_admin_supabase_client()

        # Check if chatbot exists
        chatbot_exists = (
            supabase.table("chatbot_configs")
            .select("id")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        if not chatbot_exists.data:
            raise HTTPException(status_code=404, detail="Chatbot not found")

        # Check if appearance already exists
        existing = (
            supabase.table("chatbot_appearance")
            .select("id")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        if existing.data:
            raise HTTPException(status_code=409, detail="Appearance settings already exist. Use update-appearance instead.")

        # Handle avatar upload if provided
        bot_avatar_url = None
        if avatar:
            # Validate file type
            if not avatar.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Avatar must be an image file")
            
            # Validate file size (max 2MB)
            file_content = await avatar.read()
            if len(file_content) > 2 * 1024 * 1024:  # 2MB limit
                raise HTTPException(status_code=400, detail="Avatar file too large. Maximum size is 2MB.")
            
            # Convert to base64 and store in database
            import base64
            file_extension = avatar.filename.split('.')[-1] if '.' in avatar.filename else 'png'
            base64_data = base64.b64encode(file_content).decode('utf-8')
            bot_avatar_url = f"data:image/{file_extension};base64,{base64_data}"

        # Prepare appearance data
        appearance_data = {
            "user_id": user_id,
            "chatbot_title": chatbot_title,
        }
        
        if bot_avatar_url is not None:
            appearance_data["bot_avatar_url"] = bot_avatar_url
        if theme is not None:
            appearance_data["theme"] = theme.value
        if primary_color_rgb is not None:
            appearance_data["primary_color_rgb"] = primary_color_rgb
        if border_radius_px is not None:
            appearance_data["border_radius_px"] = border_radius_px
        if position is not None:
            appearance_data["position"] = position.value

        # Create new appearance
        result = (
            supabase.table("chatbot_appearance")
            .insert(appearance_data)
            .execute()
        )

        return {
            "message": "Appearance created successfully",
            "appearance_data": appearance_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Appearance creation failed: {str(e)}")


@rag_router.put("/update-appearance")
async def update_appearance(
    chatbot_title: str = Form(...),
    avatar: Optional[UploadFile] = File(None),
    theme: Optional[Theme] = Form(None),
    primary_color_rgb: Optional[str] = Form(None),
    border_radius_px: Optional[int] = Form(None),
    position: Optional[Position] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    """Update chatbot appearance settings."""
    user_id = current_user["id"]
    chatbot_title = chatbot_title.lower()

    try:
        from app.supabase import get_admin_supabase_client
        supabase = get_admin_supabase_client()

        # Check if chatbot exists
        chatbot_exists = (
            supabase.table("chatbot_configs")
            .select("id")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        if not chatbot_exists.data:
            raise HTTPException(status_code=404, detail="Chatbot not found")

        # Check if appearance exists
        existing = (
            supabase.table("chatbot_appearance")
            .select("id")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        if not existing.data:
            raise HTTPException(status_code=404, detail="Appearance settings not found. Use create-appearance first.")

        # Handle avatar upload if provided
        bot_avatar_url = None
        if avatar:
            # Validate file type
            if not avatar.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Avatar must be an image file")
            
            # Validate file size (max 2MB)
            file_content = await avatar.read()
            if len(file_content) > 2 * 1024 * 1024:  # 2MB limit
                raise HTTPException(status_code=400, detail="Avatar file too large. Maximum size is 2MB.")
            
            # Convert to base64 and store in database
            import base64
            file_extension = avatar.filename.split('.')[-1] if '.' in avatar.filename else 'png'
            base64_data = base64.b64encode(file_content).decode('utf-8')
            bot_avatar_url = f"data:image/{file_extension};base64,{base64_data}"

        # Prepare update data - only include fields that are provided
        update_data = {}
        
        if bot_avatar_url is not None:
            update_data["bot_avatar_url"] = bot_avatar_url
        if theme is not None:
            update_data["theme"] = theme.value
        if primary_color_rgb is not None:
            update_data["primary_color_rgb"] = primary_color_rgb
        if border_radius_px is not None:
            update_data["border_radius_px"] = border_radius_px
        if position is not None:
            update_data["position"] = position.value

        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Update appearance
        result = (
            supabase.table("chatbot_appearance")
            .update(update_data)
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        return {
            "message": "Appearance updated successfully",
            "updated_fields": list(update_data.keys())
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Appearance update failed: {str(e)}")


@rag_router.get("/appearance/{chatbot_title}")
def get_appearance(
    chatbot_title: str,
    current_user: dict = Depends(get_current_user),
):
    """Get current chatbot appearance settings."""
    user_id = current_user["id"]
    chatbot_title = chatbot_title.lower()

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

        return AppearanceResponse(**result.data[0])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch appearance: {str(e)}")


# ------------------ DOCS SEPARATED ------------------ #

@rag_router.post("/docs/file")
def docs_file(
    file: UploadFile = File(...),
    chatbot_title: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a document file (.pdf/.docx/.txt) and index it into the chatbot."""
    user_id = current_user["id"]

    chatbot_title = chatbot_title.lower()
    api_key = get_api_key(user_id, chatbot_title)
    if not api_key:
        raise HTTPException(
            status_code=403,
            detail=f"No active API key found for chatbot '{chatbot_title}'"
        )

    if not file.filename.lower().endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(
            status_code=400, detail="Only .pdf, .docx, and .txt files are supported"
        )

    file_bytes = file.file.read()
    filename = file.filename

    result = process_and_index_data(
        user_id=user_id,
        filename=filename,
        file_bytes=file_bytes,
        chatbot_title=chatbot_title,
    )

    # Save tokens to database (we already have the value from result)
    update_tokens(
        user_id=user_id,
        chatbot_title=chatbot_title,
        operation_type="file_upload",
        tokens_used=result["tokens_used"]
    )

    return {
        "message": f"File '{filename}' processed successfully",
        "chunks_indexed": result["chunks_indexed"],
        "tokens_used": result["tokens_used"],
        "api_key": api_key,
    }


@rag_router.post("/docs/raw")
def upload_raw_text(request: RawTextRequest, current_user: dict = Depends(get_current_user)):
    """Upload and index raw text input."""
    user_id = current_user["id"]
    chatbot_title = request.chatbot_title.lower()

    api_key = get_api_key(user_id, chatbot_title)
    if not api_key:
        raise HTTPException(status_code=403, detail=f"No active API key found for chatbot '{chatbot_title}'")

    result = process_and_index_data(
        user_id=user_id,
        raw_text=request.raw_text,
        chatbot_title=chatbot_title,
    )

    # Save tokens to database (we already have the value from result)
    update_tokens(
        user_id=user_id,
        chatbot_title=chatbot_title,
        operation_type="raw_text",
        tokens_used=result["tokens_used"]
    )

    return result


@rag_router.post("/docs/qa")
def upload_qa_pairs(request: QARequest, current_user: dict = Depends(get_current_user)):
    """Upload and index QA pairs."""
    user_id = current_user["id"]
    chatbot_title = request.chatbot_title.lower()

    api_key = get_api_key(user_id, chatbot_title)
    if not api_key:
        raise HTTPException(status_code=403, detail=f"No active API key found for chatbot '{chatbot_title}'")

    qa_data = [{"question": qa.question, "answer": qa.answer} for qa in request.qa_pairs]

    result = process_and_index_data(
        user_id=user_id,
        qa_json=qa_data,
        chatbot_title=chatbot_title,
    )

    # Save tokens to database (we already have the value from result)
    update_tokens(
        user_id=user_id,
        chatbot_title=chatbot_title,
        operation_type="qa_pairs",
        tokens_used=result["tokens_used"]
    )

    return result


# ------------------ WEB CRAWLING ------------------ #

@rag_router.post("/crawl/discover")
def discover_links(request: DiscoverRequest, current_user: dict = Depends(get_current_user)):
    """Discover all internal endpoints from the given website."""
    if not current_user or "id" not in current_user:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized user")

    endpoints = get_internal_links(request.url)
    return {"base_url": request.url, "endpoints": endpoints}


@rag_router.post("/crawl/fetch")
def fetch_and_index(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    """Fetch a specific endpoint and index its content into RAG pipeline with heading + body grouping."""
    user_id = current_user["id"]
    chatbot_title = request.chatbot_title.lower()

    api_key = get_api_key(user_id, chatbot_title)
    if not api_key:
        raise HTTPException(status_code=403, detail=f"No active API key found for chatbot '{chatbot_title}'")

    full_url = urljoin(request.base_url, request.endpoint)

    try:
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {full_url}: {str(e)}")

    from app.RAG.pdf_processor import process_and_index_data

    soup = BeautifulSoup(response.text, "html.parser")

    grouped_chunks = []
    current_heading = None
    current_block = []

    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ["h1", "h2", "h3", "h4"]:
            if current_heading or current_block:
                grouped_chunks.append({"heading": current_heading, "content": " ".join(current_block).strip()})
                current_block = []
            current_heading = text
        else:
            current_block.append(text)

    if current_heading or current_block:
        grouped_chunks.append({"heading": current_heading, "content": " ".join(current_block).strip()})

    if not grouped_chunks:
        raise HTTPException(status_code=400, detail=f"No meaningful structured text found on {full_url}")

    results = []
    for block in grouped_chunks:
        combined_text = f"{block['heading']}\n{block['content']}" if block["heading"] else block["content"]
        result = process_and_index_data(
            user_id=user_id,
            raw_text=combined_text,
            filename=request.endpoint.strip("/"),
            source_type="web_crawling",
            chatbot_title=chatbot_title,
        )
        # Save tokens to database (we already have the value from result)
        update_tokens(
            user_id=user_id,
            chatbot_title=chatbot_title,
            operation_type="web_crawl",
            tokens_used=result["tokens_used"]
        )
        
        results.append({
            "heading": block["heading"],
            "preview": combined_text[:120],
            "chunks_indexed": result["chunks_indexed"],
            "tokens_used": result["tokens_used"],
        })

    return {
        "base_url": request.base_url,
        "endpoint": request.endpoint,
        "blocks_extracted": len(grouped_chunks),
        "indexed_blocks": results
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
    """Get all chatbots of the current user with their details and total count."""
    user_id = current_user["id"]

    try:
        from app.supabase import get_admin_supabase_client
        supabase = get_admin_supabase_client()

        # Fetch all chatbots for this user
        chatbots = (
            supabase.table("chatbot_configs")
            .select("chatbot_title, api_key, is_active, category, description, created_at, updated_at")
            .eq("user_id", user_id)
            .execute()
        )

        if not chatbots.data:
            return {
                "total_count": 0,
                "chatbots": []
            }

        # Optionally, fetch token usage summary for each bot
        from app.RAG.token_tracker import get_user_total_tokens
        token_summary = get_user_total_tokens(user_id)
        token_data = token_summary.get("details", []) if isinstance(token_summary, dict) else []

        # Map chatbot_title → total_tokens_used
        token_map = {item["chatbot_title"]: item["total_tokens"] for item in token_data}

        # Attach token count per bot
        chatbot_list = []
        for bot in chatbots.data:
            chatbot_list.append({
                **bot,
                "total_tokens_used": token_map.get(bot["chatbot_title"], 0)
            })

        return {
            "total_count": len(chatbot_list),
            "chatbots": chatbot_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chatbots: {str(e)}")
