import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from dotenv import load_dotenv
import os
from mistralai import Mistral
from pinecone import Pinecone

INDEX_NAME = "developer-quickstart-py"  # name of your Pinecone index
CHUNKS_FILE = "output_data/chunks.pkl"
PDF_FILE = "input_data/Technevity Inc. NDA 1.0.pdf"

# Load API keys
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index (only if it doesn’t exist)
if not pc.has_index(INDEX_NAME):
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "mistral-embed",     # use your embedding model here
            "field_map": {"text": "chunk_text"}
        }
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
    resp = client.embeddings.create(model="llama-text-embed-v2", inputs=[chunk])
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