import os
import faiss
import pickle
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
# Environment + Mistral Setup
# ---------------------------
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY)

# Load FAISS index and chunks
index = faiss.read_index("output_data/faiss.index")
with open("output_data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


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
    allow_origins=settings.cors_origins,
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
# Mistral RAG Helper
# ---------------------------
def generate_response(query: str):
    # 1. Get embeddings
    embed_resp = client.embeddings.create(model="mistral-embed", inputs=[query])
    query_embedding = np.array(embed_resp.data[0].embedding, dtype="float32").reshape(1, -1)

    # 2. Search FAISS
    _, indices = index.search(query_embedding, 3)
    top_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(top_chunks)

    # 3. Prompt
    prompt = f"""You are a helpful chatbot assistant who replies to all the queries related to the context provided. Use the context provided to answer their queries.
Context:
{context}

Question:
{query}

Answer:"""

    # 4. Stream from Mistral
    stream = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for event in stream:
        if event.choices and event.choices[0].delta:
            yield event.choices[0].delta.content or ""


# ---------------------------
# Endpoints
# ---------------------------
# RAG Query Endpoint
@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query")
    if not query:
        return JSONResponse({"error": "Query is required"}, status_code=400)

    return StreamingResponse(generate_response(query), media_type="text/plain", headers={"X-Accel-Buffering": "no"})


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
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )