import fitz  # New name for PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

INDEX_FILE = "output_data/faiss.index"
CHUNKS_FILE = "output_data/chunks.pkl"
PDF_FILE = "input_data/Faculty Retention Policy 2025 V1.0.docx"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
text = extract_text_from_pdf(PDF_FILE) # Store the extracted text in a variable
chunks = text_splitter.split_text(text)

# Uncomment the following two lines and comment third one when running for first time, this saves the model locally
# model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and popular
# model.save("./all-MiniLM-L6-v2")
model = SentenceTransformer("./all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

embedding_matrix = np.array(embeddings).astype("float32")

dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embedding_matrix)

faiss.write_index(index, INDEX_FILE)
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)
