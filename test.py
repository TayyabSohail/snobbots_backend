import json
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from mistralai import Mistral

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
INDEX_NAME = "snobbots-index"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

index = pc.Index(INDEX_NAME)


def generate_response(query: str):
    # 1. Get embeddings
    embed_resp = client.embeddings.create(model="mistral-embed", inputs=[query])
    query_embedding = embed_resp.data[0].embedding

    # 2. Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    top_chunks = [match["metadata"]["chunk_text"] for match in results["matches"]]
    context = "\n\n".join(top_chunks)

    # 3. Prompt
    prompt = f"""You are a helpful chatbot assistant who replies to all the queries related to the context provided. 
Use the context provided to answer their queries.

Context:
{context}

Question:
{query}

Answer:"""

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }

    full_response = ""

    with requests.post(url, headers=headers, json=data, stream=True) as r:
        for line in r.iter_lines():
            if line:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue

                payload = decoded[len("data: "):]
                if payload == "[DONE]":
                    break

                obj = json.loads(payload)
                delta = obj["choices"][0]["delta"].get("content")
                if delta:
                    print(delta, end="", flush=True)  # live stream
                    full_response += delta

    print("\n\nFinal Response:", full_response)
    return full_response


generate_response("Information")