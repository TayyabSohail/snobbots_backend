from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "snobbots-index"
index = pc.Index(INDEX_NAME)

def generate_response(query: str):
    """Search Pinecone and stream AI response with context."""
    # Embedding
    embed_resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    query_embedding = embed_resp.data[0].embedding

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    top_chunks = [match["metadata"]["chunk_text"] for match in results["matches"]]
    context = "\n\n".join(top_chunks)

    # Prompt
    prompt = f"""You are a helpful chatbot assistant. Use context to answer.

Context:
{context}

Question:
{query}

Answer:"""

    # Stream
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content