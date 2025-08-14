from flask import Flask, request, Response
from dotenv import load_dotenv
import os
import faiss
import pickle
import numpy as np
import json
from mistralai import Mistral
from flask_cors import CORS

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialize Mistral V1 SDK client
client = Mistral(api_key=API_KEY)

# Load FAISS index and chunks
index = faiss.read_index("output_data/faiss.index")
with open("output_data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Streaming response generator using Mistral API
def generate_response(query):
    # 1. Get embeddings from Mistral
    embed_resp = client.embeddings.create(model="mistral-embed", inputs=[query])
    query_embedding = np.array(embed_resp.data[0].embedding, dtype="float32").reshape(1, -1)

    # 2. Retrieve relevant chunks using FAISS
    _, indices = index.search(query_embedding, 3)
    top_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(top_chunks)

    # 3. Build prompt
    prompt = f"""You are a helpful university assistant...
Context:
{context}

Question:
{query}

Answer:"""

    # 4. Stream completion from Mistral Chat
    stream = client.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": prompt}], stream=True)
    for event in stream:
        if event.choices and event.choices[0].delta:
            yield event.choices[0].delta.content or ""

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return {"error": "Query is required"}, 400

    return Response(generate_response(query), mimetype="text/plain", headers={"X-Accel-Buffering": "no"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)