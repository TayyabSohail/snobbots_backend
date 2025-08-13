import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
import os
from mistralai import Mistral

INDEX_FILE = "output_data/faiss.index"
CHUNKS_FILE = "output_data/chunks.pkl"
PDF_FILE = "input_data/Technevity Inc. NDA 1.0.pdf"

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text = extract_text_from_pdf(PDF_FILE)
chunks = text_splitter.split_text(text)

# Embed chunks using Mistral API
embeddings = []
for chunk in chunks:
    resp = client.embeddings.create(model="mistral-embed", inputs=[chunk])
    embeddings.append(resp.data[0].embedding)

embedding_matrix = np.array(embeddings, dtype="float32")

dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

faiss.write_index(index, INDEX_FILE)
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Indexed {len(chunks)} chunks using Mistral API embeddings")