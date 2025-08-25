import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from dotenv import load_dotenv
import os
from mistralai import Mistral
from pinecone import Pinecone
from pinecone import ServerlessSpec


INDEX_NAME = "snobbots-index"  # name of your Pinecone index
CHUNKS_FILE = "output_data/chunks.pkl"
PDF_FILE = "input_data/Technevity Inc. NDA 1.0.pdf"

# Load API keys
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


# Create index (only if it doesn’t exist)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,           # Mistral embeddings = 1024 dims
        metric="cosine",          # similarity metric
        spec=ServerlessSpec(      # ✅ required in new SDK
            cloud="aws",          # or "gcp"
            region="us-east-1"    # choose region where your project is hosted
        )
    )

index = pc.Index(INDEX_NAME)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split PDF text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text = extract_text_from_pdf(PDF_FILE)
chunks = text_splitter.split_text(text)

# Create embedding + upsert into Pinecone
vectors = []
for i, chunk in enumerate(chunks):
    resp = client.embeddings.create(model="mistral-embed", inputs=[chunk])
    embedding = resp.data[0].embedding
    vectors.append({
        "id": str(i),
        "values": embedding,
        "metadata": {"chunk_text": chunk}
    })

index.upsert(vectors=vectors)

# Save chunks locally too (optional)
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Indexed {len(chunks)} chunks into Pinecone with Mistral embeddings")