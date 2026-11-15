"""
Supabase S3-Compatible Cloud Storage Client for Mother's Hut Smart Traffic & Gate Console.
Directly stores and serves all media captures, annotated images, videos, and plate crops:
- Endpoint: https://fksxkdabjoihtoebpuxq.storage.supabase.co/storage/v1/s3
- Bucket: mothers-hut-storage
- Zero local disk accumulation, zero base64 memory overhead.
"""

import os
import uuid
import logging
from datetime import datetime, timezone
import zoneinfo
import cv2
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("storage_s3")

# Indian Standard Time (IST) for S3 storage prefixes
IST = zoneinfo.ZoneInfo("Asia/Kolkata")

def get_ist_date_str():
    return datetime.now(timezone.utc).astimezone(IST).strftime("%Y-%m-%d")

# Supabase S3 Configuration
S3_ENDPOINT = os.getenv("SUPABASE_S3_ENDPOINT", "https://fksxkdabjoihtoebpuxq.storage.supabase.co/storage/v1/s3").strip()
S3_ACCESS_KEY = os.getenv("SUPABASE_S3_ACCESS_KEY_ID", "f20805943a7e1528a7c582fb9e3bcfd6").strip()
S3_SECRET_KEY = os.getenv("SUPABASE_S3_SECRET_ACCESS_KEY", "0b7207704039c29e9f6fa03676fa203c5de87423ac752881b467b2251d5ebbb8").strip()
S3_BUCKET = os.getenv("SUPABASE_S3_BUCKET", "mothers-hut-storage").strip()
S3_REGION = os.getenv("SUPABASE_S3_REGION", "ap-south-1").strip()

_s3_client = None
_bucket_verified = False


def get_s3_client():
    """Initializes and returns cached Boto3 S3 client for Supabase Storage."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    if not S3_ACCESS_KEY or not S3_SECRET_KEY:
        logger.warning("[Storage S3] Credentials not configured.")
        return None

    try:
        import boto3
        from botocore.config import Config

        _s3_client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION,
            config=Config(
                s3={"addressing_style": "path"},
                signature_version="s3v4",
                connect_timeout=6,
                read_timeout=15,
                retries={"max_attempts": 2}
            )
        )
        return _s3_client
    except Exception as e:
        logger.error(f"[Storage S3] Initialization error: {e}")
        return None


def ensure_bucket_exists():
    """Ensures target Supabase storage bucket exists."""
    global _bucket_verified
    if _bucket_verified:
        return True

    client = get_s3_client()
    if not client:
        return False

    try:
        resp = client.list_buckets()
        existing = [b["Name"] for b in resp.get("Buckets", [])]
        if S3_BUCKET not in existing:
            client.create_bucket(Bucket=S3_BUCKET)
            logger.info(f"[Storage S3] Created bucket: {S3_BUCKET}")
        _bucket_verified = True
        return True
    except Exception as e:
        logger.warning(f"[Storage S3] Bucket verification note: {e}")
        _bucket_verified = True
        return True


def generate_presigned_url(key: str, expires_in: int = 604800) -> str:
    """Generates an authenticated presigned URL for an object (valid up to 7 days)."""
    client = get_s3_client()
    if not client:
        return ""
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires_in
        )
        return url
    except Exception as e:
        logger.warning(f"[Storage S3] URL generation note for {key}: {e}")
        return ""


def upload_bytes(data: bytes, key: str, content_type: str = "image/jpeg", expires_in: int = 604800) -> str:
    """
    Uploads raw bytes directly to Supabase S3 storage.
    Returns presigned streaming/viewing URL.
    """
    if not data:
        return ""

    client = get_s3_client()
    if not client:
        return ""

    try:
        ensure_bucket_exists()
        client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=data,
            ContentType=content_type
        )
        return generate_presigned_url(key, expires_in=expires_in)
    except Exception as e:
        logger.error(f"[Storage S3] Upload failed for {key}: {e}")
        return ""


def upload_file(file_path: str, key: str = None, content_type: str = "video/mp4", expires_in: int = 604800) -> str:
    """
    Uploads a local file to Supabase S3 storage.
    Returns presigned streaming/viewing URL.
    """
    if not file_path or not os.path.exists(file_path):
        return ""

    if not key:
        today_str = get_ist_date_str()
        ext = os.path.splitext(file_path)[1] or ".mp4"
        key = f"videos/{today_str}/{uuid.uuid4().hex[:12]}{ext}"

    client = get_s3_client()
    if not client:
        return ""

    try:
        ensure_bucket_exists()
        with open(file_path, "rb") as f:
            client.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=f,
                ContentType=content_type
            )
        return generate_presigned_url(key, expires_in=expires_in)
    except Exception as e:
        logger.error(f"[Storage S3] File upload failed for {key}: {e}")
        return ""


def upload_plate_crop(crop_bgr, plate_text: str = "plate") -> str:
    """
    Encodes an authentic license plate crop in-memory and uploads to Supabase S3.
    Returns presigned cloud image URL.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    success, buffer = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        return ""

    today_str = get_ist_date_str()
    clean_txt = "".join(c for c in plate_text if c.isalnum()).upper() or "PLATE"
    key = f"plates/{today_str}/{clean_txt}_{uuid.uuid4().hex[:8]}.jpg"
    return upload_bytes(buffer.tobytes(), key, content_type="image/jpeg")


def upload_annotated_image(img_bgr) -> str:
    """
    Encodes an annotated surveillance capture in-memory and uploads to Supabase S3.
    Returns presigned cloud image URL.
    """
    if img_bgr is None or img_bgr.size == 0:
        return ""

    success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return ""

    today_str = get_ist_date_str()
    key = f"annotated/{today_str}/capture_{uuid.uuid4().hex[:10]}.jpg"
    return upload_bytes(buffer.tobytes(), key, content_type="image/jpeg")


def upload_annotated_video(video_path: str) -> str:
    """
    Uploads an annotated surveillance video to Supabase S3.
    Returns presigned cloud streaming URL.
    """
    today_str = get_ist_date_str()
    key = f"annotated/{today_str}/video_{uuid.uuid4().hex[:10]}.mp4"
    return upload_file(video_path, key=key, content_type="video/mp4")


def is_configured() -> bool:
    """Checks if Supabase S3 credentials are configured."""
    return bool(S3_ACCESS_KEY and S3_SECRET_KEY)
