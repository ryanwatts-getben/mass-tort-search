import pandas as pd
import logging
import boto3
from botocore.exceptions import ClientError
from utils import get_s3_txt_files

logger = logging.getLogger(__name__)

def ingest_data_from_s3(s3_client, bucket, prefix):
    try:
        bucket = str(bucket)
        logger.info(f"Attempting to list .txt files in bucket: {bucket} with prefix: {prefix}")
        
        txt_files = get_s3_txt_files(s3_client, bucket, prefix)

        if not txt_files:
            logger.warning(f"No .txt files found in bucket: {bucket} with prefix: {prefix}")
            return []

        file_contents = []
        for file_key in txt_files:
            logger.info(f"Retrieving file: {file_key}")
            try:
                file_obj = s3_client.get_object(Bucket=bucket, Key=file_key)
                file_content = file_obj['Body'].read().decode('utf-8')
                file_contents.append({'Key': file_key, 'Content': file_content})
            except Exception as e:
                logger.error(f"Error retrieving or decoding file {file_key}: {str(e)}")

        logger.info(f"Successfully retrieved {len(file_contents)} .txt files from S3")
        return file_contents

    except Exception as e:
        logger.error(f"Error ingesting data from S3: {str(e)}")
        raise
