import os
import requests
import numpy as np
import json
import logging
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from mistralai import Mistral

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import settings
from app.auth import auth_router
from pinecone import Pinecone
from pydantic import BaseModel
from fastapi.responses import JSONResponse



# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------
# Environment + Mistral + Pinecone Setup
# ---------------------------
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

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
    description="Backend API for Snobbots with Supabase Authentication + Mistral RAG",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["http://localhost:3000"] if frontend runs there
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Exception Handler
# ---------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )


# ---------------------------
# Mistral RAG Helper with Pinecone
# ---------------------------
class QueryRequest(BaseModel):
    query: str
    
def generate_response(query: str):
    # 1. Get embeddings
    embed_resp = client.embeddings.create(model="mistral-embed", inputs=[query])
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
    prompt = f"""You are a helpful chatbot assistant who replies to all the queries related to the context provided. 
Use the context provided to answer their queries.

Context:
{context}

Question:
{query}

Answer:"""

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }

    with requests.post(url, headers=headers, json=data, stream=True) as r:
        for line in r.iter_lines():
            if line:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue

                payload = decoded[len("data: "):]
                if payload == "[DONE]":
                    break

                obj = json.loads(payload)
                delta = obj["choices"][0]["delta"].get("content")
                if delta:
                    yield delta
# ---------------------------
# Endpoints
# ---------------------------


@app.post("/ask")
async def ask(request: QueryRequest):
    full_text = "".join([chunk for chunk in generate_response(request.query)])
    return JSONResponse({"answer": full_text})


# Supabase Auth Router
app.include_router(auth_router, prefix=settings.api_prefix)


# Root
@app.get("/")
async def root():
    return {
        "message": "Snobbots Backend API",
        "version": "1.0.0",
        "status": "running"
    }


# Health
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "debug": settings.debug
    }


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )

