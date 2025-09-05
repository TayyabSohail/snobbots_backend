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
    Accepts either:
    - File upload (.pdf, .docx, .txt)
    - Raw text (string)
    - QA JSON (stringified JSON)
    """
    user_id = current_user["id"]

    if file:
        # --- File mode ---
        filename = file.filename.lower()
        if not (filename.endswith(".pdf") or filename.endswith(".docx") or filename.endswith(".txt")):
            raise HTTPException(status_code=400, detail="Only .pdf, .docx, .txt supported")
        file_bytes = file.file.read()
        result = process_and_index_data(user_id=user_id, filename=filename, file_bytes=file_bytes)
        return {"mode": "file", "result": result}

    elif raw_text:
        # --- Raw text mode ---
        result = process_and_index_data(user_id=user_id, raw_text=raw_text)
        return {"mode": "raw_text", "result": result}

    elif qa_json:
        # --- QA JSON mode ---
        try:
            qa_data = json.loads(qa_json)  # ensure it's parsed
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for qa_json")
        result = process_and_index_data(user_id=user_id, qa_json=qa_data)
        return {"mode": "qa_json", "result": result}

    else:
        raise HTTPException(status_code=400, detail="Provide a file, raw_text, or qa_json")
    
@rag_router.get("/crawl/discover")
def discover_links(
    url: str = Query(..., description="Base website URL"),
    current_user: dict = Depends(get_current_user)
):
    """Discover all internal endpoints from the given website."""
    endpoints = get_internal_links(url)
    return {"base_url": url, "endpoints": endpoints}


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