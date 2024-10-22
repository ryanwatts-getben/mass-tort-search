import pandas as pd
import logging
import boto3
from botocore.exceptions import ClientError
from utils import get_s3_txt_files_in_folders, load_config

logger = logging.getLogger(__name__)

def ingest_data_from_s3(s3_client, bucket):
    try:
        config = load_config('config/config.yaml')
        prefixes = config['aws']['s3'].get('prefixes', [])
        
        logger.info(f"Attempting to list .txt files in bucket: {bucket} for prefixes: {prefixes}")
        
        txt_files = get_s3_txt_files_in_folders(s3_client, bucket, prefixes)

        if not txt_files:
            logger.warning(f"No .txt files found inside specified folders in bucket: {bucket}")
            return pd.DataFrame()  # Return an empty DataFrame if no files found

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
        
        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(file_contents)
        return df

    except Exception as e:
        logger.error(f"Error ingesting data from S3: {str(e)}")
        raise
