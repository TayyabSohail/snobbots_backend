import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


def process_and_index_pdf(file_bytes: bytes, filename: str, user_id: str):
    """Process a PDF (from memory), split text, embed, and upsert into Pinecone."""

    # Load API keys once
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Make index name unique per user (sanitize: lowercase, no spaces)
    INDEX_NAME = f"snobbots-{user_id.lower().replace(' ', '_')}"
    
    # Ensure index exists
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)

    # Extract text from PDF (in memory)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join(page.get_text() for page in doc)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Create embeddings & prepare vectors
    vectors = []
    for i, chunk in enumerate(chunks):
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk
        )
        embedding = resp.data[0].embedding
        vectors.append({
            "id": f"{user_id}_{filename}_{i}",
            "values": embedding,
            "metadata": {
                "chunk_text": chunk,
                "source": filename,
                "user_id": user_id
            }
        })

    # Upsert into Pinecone
    index.upsert(vectors=vectors)

    return {
        "filename": filename,
        "chunks_indexed": len(chunks),
        "index_name": INDEX_NAME,
        "user_id": user_id
    }