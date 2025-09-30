import os
from botocore.exceptions import ClientError
from app.s3.s3_client import s3_client, BUCKET_NAME, AWS_REGION


def upload_file_to_s3(file_bytes: bytes, s3_key: str, content_type: str = "application/octet-stream"):
    """
    Upload bytes to S3 at the given key. Returns dict with status and url/message.
    s3_key should include any desired folders, e.g. "user-id/files/my.pdf"
    """
    try:
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=file_bytes,
            ContentType=content_type,
        )

        #us-east-1 is a historical case, that's why making an if for that
        if AWS_REGION and AWS_REGION != "us-east-1":
            url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        else:
            url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_key}"

        return {"status": "success", "url": url}

    except ClientError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}