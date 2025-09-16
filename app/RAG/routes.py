import requests
from fastapi import APIRouter, Depends, Query, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import secrets
import string
import json

from app.RAG.rag_helper import generate_response
from app.RAG.pdf_processor import process_and_index_data
from app.RAG.auth_utils import get_current_user, validate_api_key, get_api_key
from app.RAG.link_finder import get_internal_links

rag_router = APIRouter(prefix="/rag", tags=["RAG"])


# ------------------ MODELS ------------------ #

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


# ------------------ CREATE CHATBOT ------------------ #

@rag_router.post("/create-chatbot")
def create_chatbot_api(
    chatbot_title: str,
    current_user: dict = Depends(get_current_user),
):
    """Create (or return existing) API key for a chatbot."""
    user_id = current_user["id"]
    chatbot_title = chatbot_title.lower()

    try:
        from app.supabase import get_admin_supabase_client
        supabase = get_admin_supabase_client()

        existing = (
            supabase.table("chatbot_configs")
            .select("api_key")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        if existing.data:
            return {"api_key": existing.data[0]["api_key"], "message": "API key already exists"}

        api_key = "snb_" + "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(32)
        )

        supabase.table("chatbot_configs").insert({
            "user_id": user_id,
            "chatbot_title": chatbot_title,
            "api_key": api_key,
            "is_active": True,
        }).execute()

        return {"api_key": api_key, "message": "API key created successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API key creation failed: {str(e)}")


# ------------------ DOCS SEPARATED ------------------ #

@rag_router.post("/docs/file")
def docs_file(
    file: UploadFile = File(...),
    chatbot_title: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a document file (.pdf/.docx/.txt) and index it into the chatbot."""
    user_id = current_user["id"]

    # ✅ normalize chatbot title
    chatbot_title = chatbot_title.lower()

    # ✅ ensure API key exists for this chatbot
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

    return {
        "message": f"File '{filename}' processed successfully",
        "chunks_indexed": result["chunks_indexed"],
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
        results.append({
            "heading": block["heading"],
            "preview": combined_text[:120],
            "chunks_indexed": result["chunks_indexed"]
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

    full_text = "".join([chunk for chunk in generate_response(request.query, user_id, chatbot_title)])
    return JSONResponse({"answer": full_text})