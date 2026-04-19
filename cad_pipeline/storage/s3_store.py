"""s3_store.py — S3 upload helpers for the CAD pipeline.

S3 key structure (theo thiết kế trong logic_project):
  folders/{folder_id}/files/{file_id}/original/{filename}
  folders/{folder_id}/files/{file_id}/pages/page_{n}.png
  folders/{folder_id}/files/{file_id}/blocks/page_{n}/{type}_{i}.png

Public URL pattern:
  https://{endpoint}/{bucket}/folders/...
  hoặc CloudFront: https://{cdn_domain}/folders/...
"""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from cad_pipeline.config import (
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_ACCESS_KEY,
    S3_SECRET_KEY,
    S3_REGION,
    S3_PUBLIC_BASE_URL,
)


_s3_client = None


def get_client():
    global _s3_client
    if _s3_client is None:
        kwargs: dict = dict(
            region_name=S3_REGION or None,
            aws_access_key_id=S3_ACCESS_KEY or None,
            aws_secret_access_key=S3_SECRET_KEY or None,
        )
        if S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = S3_ENDPOINT_URL
        _s3_client = boto3.client("s3", **kwargs)
    return _s3_client


def original_key(folder_id: str, file_id: str, filename: str) -> str:
    return f"folders/{folder_id}/files/{file_id}/original/{filename}"


def page_key(folder_id: str, file_id: str, page_number: int) -> str:
    return f"folders/{folder_id}/files/{file_id}/pages/page_{page_number}.png"


def block_key(
    folder_id: str,
    file_id: str,
    page_number: int,
    block_type: str,
    block_index: int,
) -> str:
    return (
        f"folders/{folder_id}/files/{file_id}/blocks/"
        f"page_{page_number}/{block_type}_{block_index}.png"
    )


def public_url(key: str) -> str:
    base = S3_PUBLIC_BASE_URL.rstrip("/")
    if base:
        return f"{base}/{key}"
    return f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"


def upload_file(
    local_path: Path | str,
    key: str,
    content_type: str = "application/octet-stream",
) -> str:
    local_path = Path(local_path)
    get_client().upload_file(
        Filename=str(local_path),
        Bucket=S3_BUCKET,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )
    return public_url(key)


def upload_bytes(
    data: bytes,
    key: str,
    content_type: str = "image/png",
) -> str:
    get_client().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    return public_url(key)


def upload_original_file(
    local_path: Path | str,
    folder_id: str,
    file_id: str,
    filename: str,
) -> str:
    key = original_key(folder_id, file_id, filename)
    ct = "application/pdf" if str(local_path).endswith(".pdf") else "application/octet-stream"
    return upload_file(local_path, key, content_type=ct)


def upload_page_image(
    local_path: Path | str,
    folder_id: str,
    file_id: str,
    page_number: int,
) -> str:
    key = page_key(folder_id, file_id, page_number)
    return upload_file(local_path, key, content_type="image/png")


def upload_block_crop(
    image_bytes: bytes,
    folder_id: str,
    file_id: str,
    page_number: int,
    block_type: str,
    block_index: int,
) -> str:
    key = block_key(folder_id, file_id, page_number, block_type, block_index)
    return upload_bytes(image_bytes, key, content_type="image/png")


def key_exists(key: str) -> bool:
    try:
        get_client().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise
