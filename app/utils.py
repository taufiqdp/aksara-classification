import logging
import os
import uuid
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv(override=True)
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


s3_client = boto3.client(
    "s3",
    aws_access_key_id=S3_ACCESS_KEY_ID,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    endpoint_url=S3_ENDPOINT_URL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_to_s3(
    image_data: bytes, filename: str | None, prediction: str, probability: float
) -> None:
    try:
        if not filename:
            filename = "image.jpg"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]

        s3_key = f"{prediction}/{timestamp}_{unique_id}_{filename}"

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=image_data,
            Metadata={
                "prediction": prediction,
                "probability": str(probability),
                "timestamp": timestamp,
                "original_filename": filename,
                "prediction_category": prediction,
            },
        )

        logger.info(f"Successfully uploaded {filename} to S3 at {s3_key}")

    except ClientError as e:
        logger.error(f"Failed to upload {filename} to S3: {e}")
    except Exception as e:
        logger.error(f"Unexpected error uploading {filename} to S3: {e}")
