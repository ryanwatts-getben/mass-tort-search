import pandas as pd
import logging
import boto3
from botocore.exceptions import ClientError
from utils import get_s3_txt_files_in_folders, load_config
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

def ingest_data_from_s3(s3_client, bucket, prefix=None) -> Dataset:
    """Ingest data from S3 bucket and return as Dataset."""
    try:
        config = load_config('config/config.yaml')
        use_prefixes = config['aws']['s3'].get('use_prefixes', True)
        prefixes = [prefix] if prefix else config['aws']['s3'].get('prefixes', [])
        max_workers = config['aws']['s3'].get('max_workers', 10)

        if not use_prefixes:
            prefixes = ['']  # Set to empty string to list all files

        logger.info(f"Attempting to list .txt files in bucket: {bucket} for prefixes: {prefixes}")

        txt_files = get_s3_txt_files_in_folders(s3_client, bucket, prefixes)

        # Filter out files with '_full.txt' in their filename
        txt_files = [file for file in txt_files if not file.endswith('_full.txt')]

        if not txt_files:
            logger.warning(f"No .txt files found inside specified folders in bucket: {bucket}")
            return Dataset.from_dict({})  # Return an empty Dataset if no files found

        file_contents = []

        # Use ThreadPoolExecutor for concurrent file retrieval
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {executor.submit(retrieve_file_content, s3_client, bucket, file_key): file_key for file_key in txt_files}
            for future in as_completed(future_to_key):
                file_key = future_to_key[future]
                try:
                    content = future.result()
                    if content is not None:
                        file_contents.append({'Key': file_key, 'Content': content})
                except Exception as e:
                    logger.error(f"Error retrieving file {file_key}: {str(e)}")

        logger.info(f"Successfully retrieved {len(file_contents)} .txt files from S3")

        # Convert directly to Dataset instead of DataFrame
        dataset = Dataset.from_dict({
            'Key': [item['Key'] for item in file_contents],
            'Content': [item['Content'] for item in file_contents]
        })

        logger.info(f"Created dataset with {len(dataset)} examples")
        return dataset

    except Exception as e:
        logger.error(f"Error ingesting data from S3: {str(e)}")
        raise

def retrieve_file_content(s3_client, bucket, file_key):
    """Retrieve the content of a file from S3."""
    try:
        logger.info(f"Retrieving file: {file_key}")
        file_obj = s3_client.get_object(Bucket=bucket, Key=file_key)
        return file_obj['Body'].read().decode('utf-8')
    except Exception as e:
        logger.error(f"Error retrieving or decoding file {file_key}: {str(e)}")
        return None
