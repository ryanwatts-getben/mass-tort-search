import os
import boto3
from dotenv import load_dotenv
from pinecone import Pinecone
from data_ingestion import ingest_data_from_s3
from preprocessing import preprocess_data
from entity_extraction import extract_entities
from relationship_extraction import extract_relationships
from vectorization import vectorize_documents
from pinecone_operations import upload_to_pinecone, search_pinecone
from scoring import score_and_rank_results
from utils import load_search_criteria, generate_report
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and initialize clients
load_dotenv()
AWS_REGION = os.getenv('AWS_REGION')
AWS_BUCKET_NAME = os.getenv('AWS_LOVELY_BUCKET')
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

def main():
    try:
        # 1. Data Ingestion
        logger.info("Starting data ingestion from S3")
        raw_data = ingest_data_from_s3(s3, AWS_BUCKET_NAME, '1225')
        
        if not raw_data:
            logger.warning("No data was ingested from S3. Exiting process.")
            return
        
        # 2. Preprocessing
        preprocessed_data = preprocess_data(raw_data)
        
        if preprocessed_data.empty:
            logger.warning("No data remained after preprocessing. Exiting process.")
            return
        
        # 3. Entity Extraction
        preprocessed_data = extract_entities(preprocessed_data)
        
        # 4. Relationship Extraction
        preprocessed_data = extract_relationships(preprocessed_data)
        
        # 5. Vectorization
        preprocessed_data = vectorize_documents(preprocessed_data)
        
        # 6. Upload to Pinecone
        upload_to_pinecone(preprocessed_data)
        
        # 7. Search and Score
        search_criteria = load_search_criteria()
        search_results = search_pinecone(search_criteria)
        
        # 8. Rank and Report
        final_results = score_and_rank_results(search_results)
        generate_report(final_results)

    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
