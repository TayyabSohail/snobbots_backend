import fitz  # New name for PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pickle
import os
from mistralai.client import MistralClient

INDEX_FILE = "output_data/faiss.index"
CHUNKS_FILE = "output_data/chunks.pkl"
PDF_FILE = "input_data/Faculty Retention Policy 2025 V1.0.docx"

# Set up Mistral API client
API_KEY = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=API_KEY)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
text = extract_text_from_pdf(PDF_FILE)
chunks = text_splitter.split_text(text)

# Get embeddings from Mistral API
embeddings = []
for chunk in chunks:
    resp = client.embeddings(
        model="mistral-embed",  # Replace with the actual Mistral embedding model name
        input=chunk
    )
    embeddings.append(resp.data[0].embedding)

embedding_matrix = np.array(embeddings, dtype="float32")

# Create FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Save FAISS index and chunks
faiss.write_index(index, INDEX_FILE)
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Indexed {len(chunks)} chunks using Mistral API embeddings")