"""Main FastAPI application with Supabase authentication."""
import os
import sys
import json
import logging
from contextlib import asynccontextmanager

import requests
import numpy as np
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pinecone import Pinecone
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.auth import auth_router
from app.helpers.response_helper import error_response

from fastapi import FastAPI, UploadFile, File, Query, Header, HTTPException
from app.RAG.RAG_pipeline import process_and_index_pdf

from app.supabase import get_supabase_client
# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------------------------
# Environment + OpenAI + Pinecone Setup
# ---------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "snobbots-index"


# Connect to existing index
index = pc.Index(INDEX_NAME)


# ---------------------------
# FastAPI Lifespan
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting Snobbots Backend API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    yield
    logger.info("Shutting down Snobbots Backend API")


# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(
    title="Snobbots Backend API",
    description="Backend API for Snobbots with Supabase Authentication + OpenAI RAG",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["http://localhost:3000"] if frontend runs there
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# ✅ Global Exception Handlers
# -------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic validation errors in request body/query/params."""
    return JSONResponse(
        status_code=422,
        content=error_response(
            message="Validation failed",
            code="VALIDATION_ERROR",
            errors=exc.errors(),
        ),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handles typical HTTP exceptions (e.g. 404, 401)."""
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            message=str(exc.detail),
            code="HTTP_ERROR",
        ),
    )


# ---------------------------
# Exception Handler
# ---------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handles all other uncaught exceptions."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=error_response(
            message="Internal server error",
            code="SERVER_ERROR",
        ),
    )


# ---------------------------
# OpenAI RAG Helper with Pinecone
# ---------------------------
class QueryRequest(BaseModel):
    query: str
    
def generate_response(query: str):
    # 1. Get embeddings from OpenAI
    embed_resp = client.embeddings.create(
        model="text-embedding-3-large",  # or "text-embedding-3-small"
        input=query
    )
    query_embedding = embed_resp.data[0].embedding

    # 2. Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    top_chunks = [match["metadata"]["chunk_text"] for match in results["matches"]]
    context = "\n\n".join(top_chunks)

    # 3. Prompt
    prompt = f"""You are a helpful chatbot assistant who replies to all queries using the provided context. 
If the answer cannot be found in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:"""

    # 4. Stream response from OpenAI Chat API
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
# ---------------------------
# Endpoints
# ---------------------------


@app.post("/ask")
async def ask(request: QueryRequest):
    full_text = "".join([chunk for chunk in generate_response(request.query)])
    return JSONResponse({"answer": full_text})

def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.split(" ")[1]
    supabase = get_supabase_client()

    # Call Supabase auth API directly
    resp = requests.get(
        f"{supabase.supabase_url}/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": supabase.supabase_key}
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid token")

    return resp.json()  # contains user_id, email, etc.


@app.post("/docs")
def docs(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")
    
    user_id = current_user["id"]

    file_bytes = file.file.read()
    result = process_and_index_pdf(file_bytes, file.filename, user_id)

    return result


# Supabase Auth Router
app.include_router(auth_router, prefix=settings.api_prefix)



# Root endpoint
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "message": "Snobbots Backend API",
        "version": "1.0.0",
        "status": "running",
    }


# Health
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "debug": settings.debug,
    }


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
