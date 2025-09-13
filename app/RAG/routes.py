import requests
from fastapi import APIRouter
from pydantic import BaseModel
from app.RAG.rag_helper import generate_response
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, Form, Query, HTTPException
from app.RAG.pdf_processor import process_and_index_data
from app.RAG.auth_utils import get_current_user
from app.RAG.link_finder import get_internal_links
from fastapi import Depends
from typing import Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from app.RAG.api_key_service import validate_api_key
import secrets
import string

rag_router = APIRouter(prefix="/rag", tags=["RAG"])


class QueryRequest(BaseModel):
    query: str
    api_key: str


@rag_router.post("/ask")
async def ask(request: QueryRequest):
    """Ask questions using API key (no authentication required)."""
    # Validate API key
    api_data = validate_api_key(request.api_key)
    if not api_data:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    user_id = api_data["user_id"]
    chatbot_title = api_data["chatbot_title"].lower()
    full_text = "".join(
        [chunk for chunk in generate_response(request.query, user_id, chatbot_title)]
    )
    return JSONResponse({"answer": full_text})


class QAPair(BaseModel):
    question: str
    answer: str


@rag_router.post("/docs")
def docs(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    qa_json: Optional[str] = Form(None),
    chatbot_title: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """Unified endpoint: accepts file (.pdf/.docx/.txt),
    raw text, and/or QA JSON (question-answer pairs).
    """
    user_id = current_user["id"]

    # ✅ normalize chatbot title to lowercase
    chatbot_title = chatbot_title.lower()

    file_bytes = None
    filename = None
    qa_data = None

    # Case 1: File Upload
    if file:
        if not file.filename.lower().endswith((".pdf", ".docx", ".txt")):
            raise HTTPException(
                status_code=400, detail="Only .pdf, .docx, and .txt files are supported"
            )
        file_bytes = file.file.read()
        filename = file.filename

    # Case 2: QA JSON
    if qa_json:
        try:
            qa_data = json.loads(qa_json)
            if not isinstance(qa_data, list):
                raise ValueError("qa_json must be a list of objects")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid qa_json format: {e}")

    # If nothing provided
    if not (file_bytes or raw_text or qa_data):
        raise HTTPException(status_code=400, detail="No valid input provided")

    # Process and index the documents
    result = process_and_index_data(
        user_id=user_id,
        filename=filename,
        file_bytes=file_bytes,
        raw_text=raw_text,
        qa_json=qa_data,
        chatbot_title=chatbot_title,
    )

    # Automatically create API key for this chatbot if it doesn't exist
    try:
        from app.supabase import get_admin_supabase_client

        supabase = get_admin_supabase_client()

        # Check if API key already exists for this user+chatbot
        existing = (
            supabase.table("chatbot_configs")
            .select("api_key")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        if not existing.data:
            # Generate API key
            api_key = "snb_" + "".join(
                secrets.choice(string.ascii_letters + string.digits) for _ in range(32)
            )

            # Insert API key
            supabase.table("chatbot_configs").insert(
                {
                    "user_id": user_id,
                    "chatbot_title": chatbot_title,
                    "api_key": api_key,
                    "is_active": True,
                }
            ).execute()

            result["api_key"] = api_key
            result["message"] = f"Documents processed and API key created: {api_key}"
        else:
            result["api_key"] = existing.data[0]["api_key"]
            result["message"] = "Documents processed (API key already exists)"

    except Exception as e:
        result["api_key"] = None
        result["message"] = f"Documents processed but API key creation failed: {str(e)}"

    return result


@rag_router.get("/crawl/discover")
def discover_links(
    url: str = Query(..., description="Base website URL"),
    current_user: dict = Depends(get_current_user),
):
    """Discover all internal endpoints from the given website."""
    if not current_user or "id" not in current_user:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized user")

    endpoints = get_internal_links(url)

    return {"base_url": url, "endpoints": endpoints}


@rag_router.post("/crawl/fetch")
def fetch_and_index(
    base_url: str = Query(..., description="Base website URL"),
    endpoint: str = Query(..., description="Specific endpoint path (e.g., /faq)"),
    chatbot_title: str = Query(..., description="Unique chatbot title"),
    current_user: dict = Depends(get_current_user),
):
    """Fetch a specific endpoint and index its content into RAG pipeline with heading + body grouping."""
    user_id = current_user["id"]

    # ✅ normalize chatbot title to lowercase
    chatbot_title = chatbot_title.lower()

    full_url = urljoin(base_url, endpoint)

    try:
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to fetch {full_url}: {str(e)}"
        )

    soup = BeautifulSoup(response.text, "html.parser")

    # Collect structured blocks
    grouped_chunks = []
    current_heading = None
    current_block = []

    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue

        if el.name in ["h1", "h2", "h3", "h4"]:
            if current_heading or current_block:
                grouped_chunks.append(
                    {"heading": current_heading, "content": " ".join(current_block).strip()}
                )
                current_block = []
            current_heading = text
        else:
            current_block.append(text)

    if current_heading or current_block:
        grouped_chunks.append(
            {"heading": current_heading, "content": " ".join(current_block).strip()}
        )

    if not grouped_chunks:
        raise HTTPException(
            status_code=400, detail=f"No meaningful structured text found on {full_url}"
        )

    # ✅ Process and index each heading+content pair
    results = []
    for block in grouped_chunks:
        combined_text = (
            f"{block['heading']}\n{block['content']}"
            if block["heading"]
            else block["content"]
        )

        result = process_and_index_data(
            user_id=user_id,
            raw_text=combined_text,
            filename=endpoint.strip("/"),
            source_type="web_crawling",
            chatbot_title=chatbot_title,
        )

        results.append(
            {
                "heading": block["heading"],
                "preview": combined_text[:120],
                "chunks_indexed": result["chunks_indexed"],
            }
        )

    # ✅ API key creation (same as in /docs)
    try:
        from app.supabase import get_admin_supabase_client

        supabase = get_admin_supabase_client()

        # Check if API key already exists for this user+chatbot
        existing = (
            supabase.table("chatbot_configs")
            .select("api_key")
            .eq("user_id", user_id)
            .eq("chatbot_title", chatbot_title)
            .execute()
        )

        if not existing.data:
            # Generate API key
            api_key = "snb_" + "".join(
                secrets.choice(string.ascii_letters + string.digits) for _ in range(32)
            )

            # Insert API key
            supabase.table("chatbot_configs").insert(
                {
                    "user_id": user_id,
                    "chatbot_title": chatbot_title,
                    "api_key": api_key,
                    "is_active": True,
                }
            ).execute()

            api_message = f"API key created: {api_key}"
        else:
            api_key = existing.data[0]["api_key"]
            api_message = "API key already exists"
    except Exception as e:
        api_key = None
        api_message = f"API key creation failed: {str(e)}"

    return {
        "base_url": base_url,
        "endpoint": endpoint,
        "blocks_extracted": len(grouped_chunks),
        "indexed_blocks": results,
        "api_key": api_key,
        "message": api_message,
    }