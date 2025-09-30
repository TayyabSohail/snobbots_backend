import os
from app.s3.s3_client import s3_client

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

def upload_file_to_s3(file_bytes, filename, content_type="application/octet-stream"):
    try:
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=file_bytes,
            ContentType=content_type
        )
        file_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{filename}"
        return {"status": "success", "url": file_url}
    except Exception as e:
        return {"status": "error", "message": str(e)}