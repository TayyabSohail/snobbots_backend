from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel, model_validator
import json
from typing import List, Dict, Any, Optional, Union

from app.s3.s3_helper import (
    upload_file_to_s3,
    list_files_in_s3,
    get_file_from_s3,
    generate_presigned_url,
    delete_file_from_s3,
)
from app.RAG.auth_utils import get_current_user, get_api_key
from pinecone import Pinecone, ServerlessSpec
import os
# Internal RAG imports (internal calls, Option 1)
from app.RAG.pdf_processor import process_and_index_data,sanitize_id
from app.RAG.token_tracker import update_tokens
from app.RAG import routes as rag_routes  # to call fetch_and_index internally (async)
# NOTE: rag_routes.fetch_and_index is async; we'll await it in crawl flow below

s3_router = APIRouter(prefix="/s3", tags=["S3"])

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# ------------------ MODELS ------------------ #
class RawTextRequest(BaseModel):
    chatbot_title: str
    raw_text: str

class QARequest(BaseModel):
    chatbot_title: str
    qa_pairs: List[Dict[str, str]]  # [{"question": "...", "answer": "..."}]

class CrawlRequest(BaseModel):
    chatbot_title: str
    url: Optional[str] = None  # single URL
    urls: Optional[List[str]] = None  # multiple URLs
    
    @model_validator(mode='after')
    def validate_urls(self):
        """Validate that exactly one of url or urls is provided."""
        if self.url is None and self.urls is None:
            raise ValueError("Either 'url' or 'urls' must be provided")
        if self.url is not None and self.urls is not None:
            raise ValueError("Cannot provide both 'url' and 'urls'. Use one or the other.")
        return self
    
    @property
    def url_list(self) -> List[str]:
        """Returns a list of URLs, converting single URL to list if needed."""
        if self.url:
            return [self.url]
        return self.urls or []

class FetchRequest(BaseModel):
    chatbot_title: str

class RemoveRequest(BaseModel):
    chatbot_title: str
    filename: str

class RemoveCrawlRequest(BaseModel):
    chatbot_title: str
    url: str
    
# ------------------------------------------------------------------------------------------- #
# =========================================================================================== #
# -------------------------------------- UPLOAD APIs ---------------------------------------- #
# =========================================================================================== #
# ------------------------------------------------------------------------------------------- #

# ------------------ FILE UPLOAD ------------------ #
@s3_router.post("/upload/file")
async def upload_file_to_s3_api(
    file: UploadFile = File(...),
    chatbot_title: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload file to S3 and then index it via internal RAG call (process_and_index_data).
    Returns upload metadata + indexing result (if indexing succeeded/failed).
    """
    try:
        user_id = current_user["id"]
        chatbot_title = chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        file_bytes = await file.read()
        s3_key = f"{user_id}/{chatbot_title}/files/{file.filename}"

        result = upload_file_to_s3(file_bytes, s3_key, file.content_type)
        if result["status"] == "error":
            raise HTTPException(500, result["message"])

        # --- Call internal indexing (single call for this file) ---
        indexing_result = None
        indexing_errors = None
        try:
            proc_result = process_and_index_data(
                user_id=user_id,
                filename=file.filename,
                file_bytes=file_bytes,
                chatbot_title=chatbot_title,
            )

            # update token tracker
            try:
                update_tokens(
                    user_id=user_id,
                    chatbot_title=chatbot_title,
                    operation_type="file_upload",
                    tokens_used=proc_result.get("tokens_used", 0),
                )
            except Exception:
                # non-fatal if token update fails
                pass

            indexing_result = {
                "chunks_indexed": proc_result.get("chunks_indexed"),
                "tokens_used": proc_result.get("tokens_used"),
            }

        except Exception as e:
            indexing_errors = str(e)

        return {
            "url": result["url"],
            "filename": file.filename,
            "uploaded_by": user_id,
            "chatbot_title": chatbot_title,
            "source": "file",
            "indexing": {
                "result": indexing_result,
                "error": indexing_errors,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ RAW TEXT UPLOAD ------------------ #
@s3_router.post("/upload/raw")
async def upload_raw_to_s3_api(
    request: RawTextRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Upload raw text to S3 and index it (process_and_index_data).
    Returns upload metadata + indexing result.
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        s3_key = f"{user_id}/{chatbot_title}/raw/{chatbot_title}.txt"
        file_bytes = request.raw_text.encode("utf-8")

        result = upload_file_to_s3(file_bytes, s3_key, "text/plain")
        if result["status"] == "error":
            raise HTTPException(500, result["message"])

        # Index the raw text
        indexing_result = None
        indexing_errors = None
        try:
            proc_result = process_and_index_data(
                user_id=user_id,
                raw_text=request.raw_text,
                chatbot_title=chatbot_title,
            )

            try:
                update_tokens(
                    user_id=user_id,
                    chatbot_title=chatbot_title,
                    operation_type="raw_text",
                    tokens_used=proc_result.get("tokens_used", 0),
                )
            except Exception:
                pass

            indexing_result = {
                "chunks_indexed": proc_result.get("chunks_indexed"),
                "tokens_used": proc_result.get("tokens_used"),
            }
        except Exception as e:
            indexing_errors = str(e)

        return {
            "url": result["url"],
            "chatbot_title": chatbot_title,
            "uploaded_by": user_id,
            "source": "raw_text",
            "indexing": {
                "result": indexing_result,
                "error": indexing_errors,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ QA PAIRS UPLOAD ------------------ #
@s3_router.post("/upload/qa")
async def upload_qa_to_s3_api(
    request: QARequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Upload QA pairs JSON to S3 and index them (process_and_index_data with qa_json).
    Returns upload metadata + indexing result.
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        s3_key = f"{user_id}/{chatbot_title}/qa/{chatbot_title}.json"
        file_bytes = json.dumps(request.qa_pairs, indent=2).encode("utf-8")

        result = upload_file_to_s3(file_bytes, s3_key, "application/json")
        if result["status"] == "error":
            raise HTTPException(500, result["message"])

        # Index QA pairs
        indexing_result = None
        indexing_errors = None
        try:
            proc_result = process_and_index_data(
                user_id=user_id,
                qa_json=request.qa_pairs,
                chatbot_title=chatbot_title,
            )

            try:
                update_tokens(
                    user_id=user_id,
                    chatbot_title=chatbot_title,
                    operation_type="qa_pairs",
                    tokens_used=proc_result.get("tokens_used", 0),
                )
            except Exception:
                pass

            indexing_result = {
                "chunks_indexed": proc_result.get("chunks_indexed"),
                "tokens_used": proc_result.get("tokens_used"),
            }
        except Exception as e:
            indexing_errors = str(e)

        return {
            "url": result["url"],
            "chatbot_title": chatbot_title,
            "uploaded_by": user_id,
            "source": "qa_pairs",
            "indexing": {
                "result": indexing_result,
                "error": indexing_errors,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ WEB CRAWLING UPLOAD ------------------ #
@s3_router.post("/upload/crawl")
async def upload_crawl_to_s3_api(
    request: CrawlRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Save crawl URLs to S3 and then trigger internal RAG fetch_and_index for each URL.
    This uses your existing Playwright-based fetch_and_index in app.RAG.routes (internal async call).
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        # Get URLs list (handles both single and multiple)
        urls_list = request.url_list
        if not urls_list:
            raise HTTPException(400, "At least one URL must be provided")

        # Save the list of URLs as a newline-separated file
        content = "\n".join(urls_list)
        file_bytes = content.encode("utf-8")

        s3_key = f"{user_id}/{chatbot_title}/crawls/{chatbot_title}.txt"

        result = upload_file_to_s3(file_bytes, s3_key, "text/plain")
        if result["status"] == "error":
            raise HTTPException(500, result["message"])

        # Now call rag_routes.fetch_and_index internally for each URL.
        indexing_summary = {
            "success_count": 0,
            "failed": [],
            "details": []
        }

        for url in urls_list:
            try:
                # Build a small FetchRequest-like object expected by rag_routes.fetch_and_index
                # rag_routes.fetch_and_index signature: async def fetch_and_index(request: FetchRequest, current_user: dict)
                fetch_req = rag_routes.FetchRequest(
                    base_url=url,
                    endpoint="",  # treat whole URL as base_url; your fetch implementation should handle it
                    chatbot_title=chatbot_title,
                )

                # Call the internal async endpoint directly
                resp = await rag_routes.fetch_and_index(fetch_req, current_user)

                # Expect the response to be a dict matching your /crawl/fetch returns
                indexing_summary["success_count"] += 1
                indexing_summary["details"].append({
                    "url": url,
                    "result": resp
                })

                # Sum tokens if present (update token tracker already done by fetch_and_index internally,
                # but if you want to be explicit you could update tokens here too. We assume fetch_and_index updates tokens).
            except Exception as e:
                indexing_summary["failed"].append({
                    "url": url,
                    "error": str(e)
                })

        # Build response based on single vs multiple URLs
        response = {
            "url": result["url"],
            "chatbot_title": chatbot_title,
            "uploaded_by": user_id,
            "source": "web_crawling",
            "indexing_summary": indexing_summary
        }
        
        # Use saved_url for single, saved_urls for multiple (matching docs)
        if request.url:
            response["saved_url"] = request.url
        else:
            response["saved_urls"] = request.urls
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------------------------------------------------------------------------------- #
# =========================================================================================== #
# -------------------------------------- FETCH APIs ----------------------------------------- #
# =========================================================================================== #
# ------------------------------------------------------------------------------------------- #
    
# ------------------ FETCH FILES ------------------ #
@s3_router.post("/fetch/files")
async def fetch_files_api(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # 🔐 Check API key before fetching
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        prefix = f"{user_id}/{chatbot_title}/files/"
        objects = list_files_in_s3(prefix)
        files = []

        for obj in objects:
            key = obj["key"]
            presigned_url = generate_presigned_url(key, expires_in=3600)
            files.append({
                "filename": key.split("/")[-1],
                "url": presigned_url
            })

        return {"chatbot_title": chatbot_title, "files": files}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ FETCH RAW TEXTS ------------------ #
@s3_router.post("/fetch/raw")
async def fetch_raw_api(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # 🔐 Check API key before fetching
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        prefix = f"{user_id}/{chatbot_title}/raw/"
        objects = list_files_in_s3(prefix)

        raws = []
        for obj in objects:
            key = obj["key"]
            content = get_file_from_s3(key).decode("utf-8")
            raws.append({"filename": key.split("/")[-1], "content": content})

        return {"chatbot_title": chatbot_title, "raw_texts": raws}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ FETCH QA PAIRS ------------------ #
@s3_router.post("/fetch/qa")
async def fetch_qa_api(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # 🔐 Check API key before fetching
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        prefix = f"{user_id}/{chatbot_title}/qa/"
        objects = list_files_in_s3(prefix)

        qa_files = []
        for obj in objects:
            key = obj["key"]
            content = get_file_from_s3(key).decode("utf-8")
            try:
                qa_pairs = json.loads(content)
            except Exception:
                qa_pairs = []
            qa_files.append({"filename": key.split("/")[-1], "qa_pairs": qa_pairs})

        return {"chatbot_title": chatbot_title, "qa_data": qa_files}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------ FETCH CRAWLED URLS ------------------ #
@s3_router.post("/fetch/crawl")
async def fetch_crawl_api(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # 🔐 Check API key before fetching
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        s3_key = f"{user_id}/{chatbot_title}/crawls/{chatbot_title}.txt"
        content = get_file_from_s3(s3_key).decode("utf-8")
        
        # Split by newlines to get individual URLs
        urls = [url.strip() for url in content.split('\n') if url.strip()]

        # Format URLs to match the expected API documentation
        crawls = []
        for url in urls:
            crawls.append({
                "filename": f"{chatbot_title}.txt",
                "url": url
            })

        return {
            "chatbot_title": chatbot_title,
            "crawls": crawls
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    
# ------------------------------------------------------------------------------------------- #
# =========================================================================================== #
# -------------------------------------- REMOVE APIs ---------------------------------------- #
# =========================================================================================== #
# ------------------------------------------------------------------------------------------- #

# # ------------------ REMOVE FILE ------------------ #
# @s3_router.post("/remove/file")
# async def remove_file_api(
#     request: RemoveRequest,
#     current_user: dict = Depends(get_current_user),
# ):
#     try:
#         user_id = current_user["id"]
#         chatbot_title = request.chatbot_title.lower()

#         # 🔐 Check API key before removing
#         api_key = get_api_key(user_id, chatbot_title)
#         if not api_key:
#             raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

#         key = f"{user_id}/{chatbot_title}/files/{request.filename}"
#         result = delete_file_from_s3(key)

#         if result["status"] == "error":
#             raise HTTPException(404, result["message"])

#         return {"success": True, "removed_file": request.filename}
#     except Exception as e:
#         raise HTTPException(500, str(e))


# # ------------------ REMOVE RAW ------------------ #
# @s3_router.post("/remove/raw")
# async def remove_raw_api(
#     request: RemoveRequest,
#     current_user: dict = Depends(get_current_user),
# ):
#     try:
#         user_id = current_user["id"]
#         chatbot_title = request.chatbot_title.lower()

#         api_key = get_api_key(user_id, chatbot_title)
#         if not api_key:
#             raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

#         key = f"{user_id}/{chatbot_title}/raw/{request.filename}"
#         result = delete_file_from_s3(key)

#         if result["status"] == "error":
#             raise HTTPException(404, result["message"])

#         return {"success": True, "removed_file": request.filename}
#     except Exception as e:
#         raise HTTPException(500, str(e))


# # ------------------ REMOVE QA ------------------ #
# @s3_router.post("/remove/qa")
# async def remove_qa_api(
#     request: RemoveRequest,
#     current_user: dict = Depends(get_current_user),
# ):
#     try:
#         user_id = current_user["id"]
#         chatbot_title = request.chatbot_title.lower()

#         api_key = get_api_key(user_id, chatbot_title)
#         if not api_key:
#             raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

#         key = f"{user_id}/{chatbot_title}/qa/{request.filename}"
#         result = delete_file_from_s3(key)

#         if result["status"] == "error":
#             raise HTTPException(404, result["message"])

#         return {"success": True, "removed_file": request.filename}
#     except Exception as e:
#         raise HTTPException(500, str(e))


# ------------------ REMOVE CRAWL ------------------ #
@s3_router.post("/remove/crawl")
async def remove_crawl_api(
    request: RemoveRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        key = f"{user_id}/{chatbot_title}/crawls/{request.filename}"
        result = delete_file_from_s3(key)

        if result["status"] == "error":
            raise HTTPException(404, result["message"])

        return {"success": True, "removed_file": request.filename}
    except Exception as e:
        raise HTTPException(500, str(e))
    
    # ------------------ REMOVE RAW TEXT VECTORS ------------------ #
@s3_router.post("/remove/raw_vectors")
async def remove_raw_vectors_api(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        INDEX_NAME = f"snobbots-{sanitize_id(user_id.lower().replace(' ', '_'))}"
        namespace = sanitize_id(chatbot_title.strip().lower().replace(" ", "_"))

        index = pc.Index(INDEX_NAME)
        dimension = 3072  # your index dimension

        ids_to_delete = []
        top_k = 1000
        offset = 0

        while True:
            # Use a dummy vector for query; Pinecone ignores it if filter is present
            response = index.query(
                vector=[0.0] * dimension,
                top_k=top_k,
                include_metadata=True,
                filter={"source": "raw_text", "user_id": user_id},
                namespace=namespace
            )

            matches = response.get("matches", [])
            if not matches:
                break

            ids_to_delete.extend([m["id"] for m in matches])

            # Stop if fewer than top_k returned
            if len(matches) < top_k:
                break

            offset += top_k

        if not ids_to_delete:
            return {"message": "No raw_text vectors found to delete."}

        # Delete in batches
        batch_size = 500
        for i in range(0, len(ids_to_delete), batch_size):
            index.delete(ids=ids_to_delete[i:i + batch_size], namespace=namespace)

        return {"message": f"Deleted {len(ids_to_delete)} raw_text vectors from Pinecone."}

    except Exception as e:
        raise HTTPException(500, str(e))
    
    # ------------------ REMOVE QA VECTORS ------------------ #
@s3_router.post("/remove/qa_vectors")
async def remove_qa_vectors_api(
    request: FetchRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete all Pinecone vectors with metadata.source == 'qa_json' for the given chatbot.
    Uses dummy vector + metadata filter + batching to handle large numbers of vectors.
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        INDEX_NAME = f"snobbots-{sanitize_id(user_id.lower().replace(' ', '_'))}"
        namespace = sanitize_id(chatbot_title.strip().lower().replace(" ", "_"))

        index = pc.Index(INDEX_NAME)
        dimension = 3072  # match your index dimension

        ids_to_delete = []
        top_k = 1000

        while True:
            # Dummy vector to enable metadata filtering
            response = index.query(
                vector=[0.0] * dimension,
                top_k=top_k,
                include_metadata=True,
                filter={"source": "qa_json", "user_id": user_id},
                namespace=namespace
            )

            matches = response.get("matches", [])
            if not matches:
                break

            ids_to_delete.extend([m["id"] for m in matches])

            if len(matches) < top_k:
                break

        if not ids_to_delete:
            return {"message": "No QA vectors found to delete."}

        # Delete in batches
        batch_size = 500
        for i in range(0, len(ids_to_delete), batch_size):
            index.delete(ids=ids_to_delete[i:i + batch_size], namespace=namespace)

        return {"message": f"Deleted {len(ids_to_delete)} QA vectors from Pinecone."}

    except Exception as e:
        raise HTTPException(500, str(e))
    
    
    # ------------------ REMOVE VECTORS FOR SPECIFIC FILE ------------------ #
@s3_router.post("/remove/file_vectors")
async def remove_file_vectors_api(
    request: RemoveRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete all Pinecone vectors corresponding to a specific file or raw upload.
    Uses metadata.source = request.filename to filter vectors.
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()
        filename = request.filename

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        INDEX_NAME = f"snobbots-{sanitize_id(user_id.lower().replace(' ', '_'))}"
        namespace = sanitize_id(chatbot_title.strip().lower().replace(" ", "_"))

        index = pc.Index(INDEX_NAME)
        dimension = 3072  # match your index

        ids_to_delete = []
        top_k = 1000

        while True:
            # Dummy vector for metadata-only query
            response = index.query(
                vector=[0.0] * dimension,
                top_k=top_k,
                include_metadata=True,
                filter={"source": filename, "user_id": user_id},
                namespace=namespace
            )

            matches = response.get("matches", [])
            if not matches:
                break

            ids_to_delete.extend([m["id"] for m in matches])

            if len(matches) < top_k:
                break

        if not ids_to_delete:
            return {"message": f"No vectors found for file '{filename}'."}

        # Delete in batches
        batch_size = 500
        for i in range(0, len(ids_to_delete), batch_size):
            index.delete(ids=ids_to_delete[i:i + batch_size], namespace=namespace)

        return {"message": f"Deleted {len(ids_to_delete)} vectors for file '{filename}'."}

    except Exception as e:
        raise HTTPException(500, str(e))
    
    
    

@s3_router.post("/remove/crawl_vectors")
async def remove_crawl_vectors(
    request: RemoveCrawlRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Remove all Pinecone vectors for a specific crawled page.
    Filters by user_id and source (url).
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(403, f"No active API key found for chatbot '{chatbot_title}'")

        # Compute index and namespace
        INDEX_NAME = f"snobbots-{sanitize_id(user_id.lower().replace(' ', '_'))}"
        namespace = sanitize_id(chatbot_title.strip().lower().replace(" ", "_"))

        if INDEX_NAME not in pc.list_indexes().names():
            raise HTTPException(404, f"Pinecone index '{INDEX_NAME}' not found")

        index = pc.Index(INDEX_NAME)

        # Construct full source string
        source_str = f"{request.url.rstrip('/')}"

        # Delete vectors by filter
        resp = index.delete(
            namespace=namespace,
            filter={
                "source": source_str,
                "user_id": user_id
            }
        )

        return {
            "success": True,
            "removed_count": resp.get("deletedCount", "unknown"),
            "source": source_str,
            "namespace": namespace,
        }

    except Exception as e:
        raise HTTPException(500, str(e))
    
    
    
    
    
# ------------------------------------------------------------------------------------------- #
# =========================================================================================== #
# -------------------------------------- COMBINED REMOVE APIs ------------------------------- #
# =========================================================================================== #
# ------------------------------------------------------------------------------------------- #
# ------------------ REMOVE FILE + VECTORS ------------------ #
@s3_router.post("/remove/file")
async def remove_file_and_vectors_api(
    request: RemoveRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Remove file from S3 AND delete all Pinecone vectors whose metadata.source == filename.
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()
        filename = request.filename

        # 🔐 Validate API key
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(
                403,
                f"No active API key found for chatbot '{chatbot_title}'"
            )

        # ---------------------------------------------------------
        # 1️⃣ DELETE FILE FROM S3
        # ---------------------------------------------------------
        key = f"{user_id}/{chatbot_title}/files/{filename}"
        s3_result = delete_file_from_s3(key)

        if s3_result["status"] == "error":
            raise HTTPException(404, s3_result["message"])

        # ---------------------------------------------------------
        # 2️⃣ DELETE CORRESPONDING PINECONE VECTORS
        # ---------------------------------------------------------
        INDEX_NAME = f"snobbots-{sanitize_id(user_id.lower().replace(' ', '_'))}"
        namespace = sanitize_id(chatbot_title.strip().lower().replace(" ", "_"))

        index = pc.Index(INDEX_NAME)
        dimension = 3072  # match embedding dimension

        ids_to_delete = []
        top_k = 1000

        while True:
            response = index.query(
                vector=[0.0] * dimension,           # dummy vector
                top_k=top_k,
                include_metadata=True,
                filter={"source": filename, "user_id": user_id},
                namespace=namespace
            )

            matches = response.get("matches", [])
            if not matches:
                break

            ids_to_delete.extend([m["id"] for m in matches])

            if len(matches) < top_k:
                break

        # Delete in Pinecone
        deleted_count = 0
        if ids_to_delete:
            batch_size = 500
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i + batch_size]
                index.delete(ids=batch, namespace=namespace)
                deleted_count += len(batch)

        return {
            "success": True,
            "removed_file": filename,
            "vectors_deleted": deleted_count,
            "message": f"Removed file '{filename}' and deleted {deleted_count} vectors."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    
    
@s3_router.post("/remove/qa")
async def remove_qa_and_vectors_api(
    request: RemoveRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    1) Deletes the QA file from S3.
    2) Deletes all Pinecone vectors where:
        metadata.source == "qa_json"
        metadata.filename == request.filename
        metadata.user_id == current_user.id
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # Validate API Key
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(
                403,
                f"No active API key found for chatbot '{chatbot_title}'"
            )

        # ----------- STEP 1: DELETE FILE FROM S3 -----------
        key = f"{user_id}/{chatbot_title}/qa/{request.filename}"
        result = delete_file_from_s3(key)

        if result["status"] == "error":
            raise HTTPException(404, result["message"])

        # ----------- STEP 2: DELETE MATCHING PINECONE VECTORS -----------

        INDEX_NAME = f"snobbots-{sanitize_id(str(user_id).lower())}"
        namespace = sanitize_id(chatbot_title.strip().lower())

        index = pc.Index(INDEX_NAME)
        dimension = 3072
        top_k = 1000
        ids_to_delete = []

        # Query in loops until no more matches
        while True:
            response = index.query(
                vector=[0.0] * dimension,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
                filter={
                    "source": "qa_json",
                    "user_id": user_id
                }
            )

            matches = response.get("matches", [])
            if not matches:
                break

            ids_to_delete.extend([m["id"] for m in matches])

            if len(matches) < top_k:
                break

        # Perform deletion
        if ids_to_delete:
            batch_size = 500
            for i in range(0, len(ids_to_delete), batch_size):
                index.delete(
                    ids=ids_to_delete[i:i + batch_size],
                    namespace=namespace
                )

        return {
            "success": True,
            "removed_file": request.filename,
            "deleted_vectors": len(ids_to_delete)
        }

    except Exception as e:
        raise HTTPException(500, str(e))
    
@s3_router.post("/remove/raw")
async def remove_raw_and_vectors_api(
    request: RemoveRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    1) Delete RAW file from S3.
    2) Delete all Pinecone vectors that match:
        metadata.source == "raw_text"
        metadata.filename == request.filename
        metadata.user_id == current_user.id
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # Validate chatbot API key
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(
                403, f"No active API key found for chatbot '{chatbot_title}'"
            )

        # ---------------------- STEP 1: DELETE FROM S3 ----------------------
        key = f"{user_id}/{chatbot_title}/raw/{request.filename}"
        result = delete_file_from_s3(key)

        if result["status"] == "error":
            raise HTTPException(404, result["message"])

        # ---------------------- STEP 2: DELETE FROM PINECONE ----------------------
        INDEX_NAME = f"snobbots-{sanitize_id(str(user_id).lower())}"
        namespace = sanitize_id(chatbot_title.strip().lower())

        index = pc.Index(INDEX_NAME)
        dimension = 3072  # Your index dimension
        top_k = 1000

        ids_to_delete = []

        # Loop to collect ALL matching vectors
        while True:
            response = index.query(
                vector=[0.0] * dimension,      # Dummy vector
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
                filter={
                    "source": "raw_text",
                    "user_id": user_id
                }
            )

            matches = response.get("matches", [])
            if not matches:
                break

            ids_to_delete.extend([m["id"] for m in matches])

            if len(matches) < top_k:
                break

        # No matching vectors?
        if not ids_to_delete:
            return {
                "success": True,
                "removed_file": request.filename,
                "deleted_vectors": 0
            }

        # Batch delete
        batch_size = 500
        for i in range(0, len(ids_to_delete), batch_size):
            index.delete(
                ids=ids_to_delete[i:i + batch_size],
                namespace=namespace
            )

        return {
            "success": True,
            "removed_file": request.filename,
            "deleted_vectors": len(ids_to_delete)
        }

    except Exception as e:
        raise HTTPException(500, str(e))
    
    
@s3_router.post("/remove/crawl")
async def remove_crawl_and_vectors_api(
    request: RemoveCrawlRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    1) Delete crawl file from S3.
    2) Remove all Pinecone vectors that belong to this crawled page:
       metadata.source == base_url + endpoint
       metadata.user_id == current_user.id
    """
    try:
        user_id = current_user["id"]
        chatbot_title = request.chatbot_title.lower()

        # Validate chatbot API key
        api_key = get_api_key(user_id, chatbot_title)
        if not api_key:
            raise HTTPException(
                403, f"No active API key found for chatbot '{chatbot_title}'"
            )

        # ---------------------- STEP 1: DELETE FROM S3 ----------------------
        key = f"{user_id}/{chatbot_title}/crawls/{request.filename}"
        result = delete_file_from_s3(key)

        if result["status"] == "error":
            raise HTTPException(404, result["message"])

        # ---------------------- STEP 2: DELETE FROM PINECONE ----------------------
        INDEX_NAME = f"snobbots-{sanitize_id(str(user_id).lower())}"
        namespace = sanitize_id(chatbot_title.strip().lower())

        if INDEX_NAME not in pc.list_indexes().names():
            raise HTTPException(404, f"Pinecone index '{INDEX_NAME}' not found")

        index = pc.Index(INDEX_NAME)

        # Full source string used during crawling
        source_str = f"{request.base_url.rstrip('/')}{request.endpoint}"

        # Direct delete with metadata filter
        delete_response = index.delete(
            namespace=namespace,
            filter={
                "source": source_str,
                "user_id": user_id
            }
        )

        return {
            "success": True,
            "removed_file": request.filename,
            "deleted_vectors": delete_response.get("deletedCount", "unknown"),
            "source": source_str,
        }

    except Exception as e:
        raise HTTPException(500, str(e))