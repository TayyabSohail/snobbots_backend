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

rag_router = APIRouter(prefix="/rag", tags=["RAG"])

class QueryRequest(BaseModel):
    query: str


@rag_router.post("/ask")
async def ask(
    request: QueryRequest,
    current_user:dict = Depends(get_current_user)
    ):
    
    user_id = current_user["id"]
    full_text = "".join([chunk for chunk in generate_response(request.query,user_id)])
    return JSONResponse({"answer": full_text})



class QAPair(BaseModel):
    question: str
    answer: str


@rag_router.post("/docs")
def docs(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    qa_json: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Unified endpoint: accepts file (.pdf/.docx/.txt),
    raw text, and/or QA JSON (question-answer pairs).
    Can handle multiple inputs at once.
    """
    user_id = current_user["id"]

    file_bytes = None
    filename = None
    qa_data = None

    # Case 1: File Upload
    if file:
        if not file.filename.lower().endswith((".pdf", ".docx", ".txt")):
            raise HTTPException(status_code=400, detail="Only .pdf, .docx, and .txt files are supported")
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

    # Pass all collected inputs
    return process_and_index_data(
        user_id=user_id,
        filename=filename,
        file_bytes=file_bytes,
        raw_text=raw_text,
        qa_json=qa_data
    )

@rag_router.get("/crawl/discover")
def discover_links(
    url: str = Query(..., description="Base website URL"),
    current_user: dict = Depends(get_current_user)
):
    """Discover all internal endpoints from the given website."""

    # Explicitly validate user
    if not current_user or "id" not in current_user:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized user")

    endpoints = get_internal_links(url)

    return {
        "base_url": url,
        "endpoints": endpoints,
        "user_id": current_user["id"]
    }

@rag_router.post("/crawl/fetch")
def fetch_and_index(
    base_url: str = Query(..., description="Base website URL"),
    endpoint: str = Query(..., description="Specific endpoint path (e.g., /faq)"),
    current_user: dict = Depends(get_current_user)
):
    """Fetch a specific endpoint and index its content into RAG pipeline."""
    user_id = current_user["id"]
    full_url = urljoin(base_url, endpoint)

    try:
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {full_url}: {str(e)}")

    soup = BeautifulSoup(response.text, "html.parser")
    # Extract visible text only
    text = " ".join([t.get_text(" ", strip=True) for t in soup.find_all(["p", "li", "h1", "h2", "h3", "h4"]) if t.get_text(strip=True)])

    if not text:
        raise HTTPException(status_code=400, detail=f"No text found on {full_url}")

    result = process_and_index_data(
        user_id=user_id,
        raw_text=text,
        filename=endpoint.strip("/")
    )

    return {
        "base_url": base_url,
        "endpoint": endpoint,
        "indexed_chars": len(text),
        "result": result
    }