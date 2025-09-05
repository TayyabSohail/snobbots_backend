from fastapi import APIRouter
from pydantic import BaseModel
from app.RAG.rag_helper import generate_response
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, Form, HTTPException
from app.RAG.pdf_processor import process_and_index_data
from app.RAG.auth_utils import get_current_user
from fastapi import Depends
from typing import Optional



rag_router = APIRouter(prefix="/rag", tags=["RAG"])

class QueryRequest(BaseModel):
    query: str


@rag_router.post("/ask")
async def ask(
    request: QueryRequest,
    current_user:dict = Depends(get_current_user)
    ):
    
    user_id = current_user["id"]
    full_text = "".join([chunk for chunk in generate_response(request.query,user_id)])
    return JSONResponse({"answer": full_text})



@rag_router.post("/docs")
def docs(
    file: Optional[UploadFile] = File(None),              # optional file upload
    raw_text: Optional[str] = Form(None),                 # optional plain text
    qa_json: Optional[str] = Form(None),                  # optional QA JSON
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["id"]

    # Case 1: File upload (.pdf, .docx, .txt)
    if file:
        if not (file.filename.lower().endswith(".pdf") or
                file.filename.lower().endswith(".docx") or
                file.filename.lower().endswith(".txt")):
            raise HTTPException(status_code=400, detail="Only .pdf, .docx, or .txt files are supported")

        file_bytes = file.file.read()
        result = process_and_index_data(
            user_id=user_id,
            filename=file.filename,
            file_bytes=file_bytes
        )
        return result

    # Case 2: Raw text
    if raw_text:
        result = process_and_index_data(
            user_id=user_id,
            raw_text=raw_text
        )
        return result

    # Case 3: QA JSON
    if qa_json:
        result = process_and_index_data(
            user_id=user_id,
            qa_json=qa_json
        )
        return result

    # If none provided
    raise HTTPException(status_code=400, detail="You must provide either a file, raw_text, or qa_json")