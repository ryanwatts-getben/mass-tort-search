import os
import boto3
from dotenv import load_dotenv
from pinecone import Pinecone
from data_ingestion import ingest_data_from_s3
from preprocessing import preprocess_data
from entity_extraction import extract_entities
from relationship_extraction import extract_relationships
from vectorization import vectorize_documents
from pinecone_operations import upload_to_pinecone, search_pinecone, index_name
from scoring import score_and_rank_results
from utils import load_search_criteria, generate_report, load_config, setup_logging
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd

# Enable Copy-on-Write
pd.options.mode.copy_on_write = True

# Set up logging
logger = setup_logging()

# Load configuration
config = load_config('config/config.yaml')
AWS_REGION = config['aws']['region']
AWS_BUCKET_NAME = config['aws']['s3']['bucket']
AWS_PREFIXES = config['aws']['s3'].get('prefixes', [])
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
print(AWS_BUCKET_NAME)

# Initialize S3 client
s3 = boto3.client('s3', 
                  region_name=AWS_REGION,
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = config['pinecone']['index_name']

def process_batch(batch):
    # Preprocessing
    batch = preprocess_data(batch)
    
    # Entity Extraction
    batch = extract_entities(batch)
    
    # Relationship Extraction
    batch = extract_relationships(batch)
    
    # Vectorization
    batch = vectorize_documents(batch)
    
    return batch

def main():
    try:
        # 1. Data Ingestion
        logger.info("Starting data ingestion from S3")
        raw_data = ingest_data_from_s3(s3, AWS_BUCKET_NAME)
        
        if raw_data.empty:
            logger.error("No data retrieved from S3. Exiting pipeline.")
            return
        
        # Process data in batches
        batch_size = 32  # Adjust this based on your memory constraints
        for i in range(0, len(raw_data), batch_size):
            batch = raw_data[i:i+batch_size]
            processed_batch = process_batch(batch)
            
            # Upload to Pinecone
            upload_to_pinecone(processed_batch, index_name)
        
        # 6. Search and Score
        search_criteria = load_search_criteria(config)
        search_results = search_pinecone(search_criteria, index_name)
        
        # 7. Rank and Report
        final_results = score_and_rank_results(search_results)
        generate_report(final_results)

    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
