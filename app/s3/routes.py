from fastapi import APIRouter
from fastapi import APIRouter, UploadFile, File, HTTPException
import boto3
from app.s3.s3_helper import upload_file_to_s3


s3_router = APIRouter(prefix="/s3", tags=["RAG"])

@s3_router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to S3 and return the file URL.
    """
    try:
        file_bytes = await file.read()
        result = upload_file_to_s3(file_bytes, file.filename, file.content_type)

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return {"url": result["url"], "filename": file.filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))