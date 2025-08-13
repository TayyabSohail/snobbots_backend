from flask import Flask, request, Response,render_template
import faiss
import pickle
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load everything once at startup
index = faiss.read_index("output_data/faiss.index")
with open("output_data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer("./all-MiniLM-L6-v2")


# Generator to stream Mistral response
def generate_response(query):
    query_embedding = model.encode([query]).astype("float32")
    _, indices = index.search(query_embedding, 3)
    top_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(top_chunks)

    prompt = f"""You are a helpful university assistant. If the user greets you or makes small talk, respond conversationally. Otherwise, use the university prospectus context to answer their question.

Context:
{context}

Question:
{query}

Answer:"""

    # Stream response from Mistral
    #Ollama's default port is 11434
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral:7b-instruct-q4_0",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
            except Exception:
                continue

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return {"error": "Query is required"}, 400

    return Response(generate_response(query), mimetype="text/plain", headers={"X-Accel-Buffering": "no"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)