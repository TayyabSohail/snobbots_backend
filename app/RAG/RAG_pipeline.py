import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec

# Config
INDEX_NAME = "snobbots-index"  # Pinecone index name
PDF_FILE = "input_data/Technevity Inc. NDA 1.0.pdf"

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index (only if it doesn’t exist)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,           # OpenAI text-embedding-3-large = 3072 dims
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",          # or "gcp"
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text = extract_text_from_pdf(PDF_FILE)
chunks = text_splitter.split_text(text)

# Create embeddings + upsert into Pinecone
vectors = []
for i, chunk in enumerate(chunks):
    resp = client.embeddings.create(
        model="text-embedding-3-large", 
        input=chunk
    )
    embedding = resp.data[0].embedding
    vectors.append({
        "id": str(i),
        "values": embedding,
        "metadata": {"chunk_text": chunk}
    })

index.upsert(vectors=vectors)

print(f"✅ Indexed {len(chunks)} chunks into Pinecone with OpenAI embeddings")