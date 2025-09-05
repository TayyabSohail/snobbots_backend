import fitz  # PyMuPDF
import docx  # python-docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import json

# Load keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


def process_and_index_data(
    user_id: str,
    filename: str = None,
    file_bytes: bytes = None,
    raw_text: str = None,
    qa_json: str | list = None
):
    """
    Process data (PDF, DOCX, TXT, raw text, or QA JSON), chunk, embed, and upsert into Pinecone.

    Args:
        user_id: ID of the user
        filename: Optional filename for file (PDF, DOCX, TXT)
        file_bytes: File bytes
        raw_text: Raw text input
        qa_json: JSON string OR list with [{"question": "...", "answer": "..."}]
    """

    # Unique index name per user
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

    # Collect text chunks
    chunks = []

    # Case 1: File upload
    if file_bytes and filename:
        ext = filename.lower().split(".")[-1]

        if ext == "pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "".join(page.get_text() for page in doc)

        elif ext == "docx":
            doc = docx.Document(file_bytes)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        elif ext == "txt":
            text = file_bytes.decode("utf-8")

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks.extend(text_splitter.split_text(text))

    # Case 2: Raw text
    if raw_text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks.extend(text_splitter.split_text(raw_text))

    # Case 3: QA JSON
    if qa_json:
        # Normalize qa_json -> list
        if isinstance(qa_json, str):
            try:
                qa_pairs = json.loads(qa_json)
            except Exception as e:
                raise ValueError(f"Invalid QA JSON string format: {e}")
        elif isinstance(qa_json, list):
            qa_pairs = qa_json
        else:
            raise ValueError("qa_json must be a JSON string or a list")

        # Validate and build chunks
        if not all(isinstance(item, dict) and "question" in item and "answer" in item for item in qa_pairs):
            raise ValueError("qa_json must be a list of {question, answer} objects")

        for i, qa in enumerate(qa_pairs):
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if q and a:
                chunks.append(f"Q: {q}\nA: {a}")

    # Safety check
    if not chunks:
        raise ValueError("No valid input provided (PDF/DOCX/TXT, raw_text, or qa_json required).")

    # Embed + upsert
    vectors = []
    for i, chunk in enumerate(chunks):
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk
        )
        embedding = resp.data[0].embedding
        vectors.append({
            "id": f"{user_id}_{filename or 'custom'}_{i}",
            "values": embedding,
            "metadata": {
                "chunk_text": chunk,
                "source": filename or "custom",
                "user_id": user_id
            }
        })

    index.upsert(vectors=vectors)

    return {
        "chunks_indexed": len(chunks),
        "index_name": INDEX_NAME,
        "user_id": user_id
    }