from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def generate_response(query: str, user_id: str,chatbot_title:str):
    """Search Pinecone (user-specific index) and stream AI response with context."""

    # Build index name per user
    index_name = f"snobbots-{user_id.lower().replace(' ', '_')}"

    # ✅ Check if index exists
    if index_name not in pc.list_indexes().names():
        yield f"⚠️ No knowledge base found for user `{user_id}`. Please upload documents first."
        return

    index = pc.Index(index_name)
    
    if not chatbot_title:
        raise ValueError("chatbot_title is required to create a namespace")
    namespace = chatbot_title.strip().lower().replace(" ", "_")

    # Embedding for query
    embed_resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    query_embedding = embed_resp.data[0].embedding

    # Query Pinecone for most relevant chunks
    results = index.query(
        namespace=namespace,
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    top_chunks = [match["metadata"]["chunk_text"] for match in results["matches"]]
    context = "\n\n".join(top_chunks) if top_chunks else "No context found."

    # Prompt for LLM
    prompt = f"""You are a helpful chatbot assistant. Use the context to answer.

Context:
{context}

Question:
{query}

Answer:"""

    # Stream LLM response
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content