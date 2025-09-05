from fastapi import APIRouter
from pydantic import BaseModel
from app.RAG.rag_helper import generate_response
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, HTTPException
from app.RAG.pdf_processor import process_and_index_pdf
from app.RAG.auth_utils import get_current_user
from fastapi import Depends


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
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")
    
    user_id = current_user["id"]

    file_bytes = file.file.read()
    result = process_and_index_pdf(file_bytes, file.filename, user_id)

    return result