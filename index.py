import boto3
import json
from tqdm import tqdm
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for verbose output, change to INFO or WARNING to reduce verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='indexing.log',  # Log output will be saved to indexing.log
    filemode='w'  # Overwrite the log file each time the script runs
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
AWS_REGION = os.getenv('AWS_REGION')

# Initialize clients
s3 = boto3.client('s3', region_name=AWS_REGION)
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
openai_client = OpenAI()

# Initialize the Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Constants
BUCKET_NAME = os.getenv('AWS_LOVELY_BUCKET')
INDEX_NAME = 'lovely'
BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072  # Dimension for text-embedding-3-large

def get_embedding(text):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.debug(f"Generating embedding for text of length {len(text)}")
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Embedding generated successfully")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to get embedding after {max_retries} attempts")
                return None
            sleep_time = 2 ** attempt
            logger.debug(f"Retrying after sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)  # Exponential backoff

def process_and_embed_document(document):
    key = document['Key']
    logger.debug(f"Processing document: {key}")
    try:
        content = document['Body'].read()
        text = content.decode('utf-8')
        logger.debug(f"Decoded document {key} using utf-8 encoding")
    except UnicodeDecodeError as e:
        logger.warning(f"utf-8 decoding failed for document {key}: {e}")
        try:
            text = content.decode('latin-1')
            logger.debug(f"Decoded document {key} using latin-1 encoding")
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode document {key}: {e}")
            return None
    except Exception as e:
        logger.error(f"Error reading document {key}: {e}")
        return None

    if not text.strip():
        logger.warning(f"Document {key} is empty after decoding")
        return None

    embedding = get_embedding(text)
    if embedding is None:
        logger.error(f"Failed to generate embedding for document {key}")
        return None

    return {
        'id': key,
        'values': embedding,
        'metadata': {'text': text[:1000]}
    }

def index_s3_bucket():
    logger.info("Starting indexing process")
    try:
        existing_indexes = pc.list_indexes().names()
        logger.debug(f"Existing Pinecone indexes: {existing_indexes}")
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region=PINECONE_ENVIRONMENT
                )
            )
            logger.info(f"Index '{INDEX_NAME}' created")
        else:
            logger.info(f"Index '{INDEX_NAME}' already exists")
    except Exception as e:
        logger.error(f"Error creating/checking index '{INDEX_NAME}': {e}")
        return

    index = pc.Index(INDEX_NAME)
    
    paginator = s3.get_paginator('list_objects_v2')
    logger.debug(f"Listing objects in S3 bucket '{BUCKET_NAME}'")
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME)

    total_files = 0
    batch = []
    for page in page_iterator:
        contents = page.get('Contents', [])
        logger.debug(f"Retrieved {len(contents)} objects from S3")
        for obj in tqdm(contents, desc="Processing documents"):
            key = obj['Key']
            total_files += 1
            if not key.lower().endswith('.txt'):
                logger.info(f"Skipping non-text file: {key}")
                continue

            try:
                document = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            except Exception as e:
                logger.error(f"Error retrieving document {key} from S3: {e}")
                continue

            document['Key'] = key  # Add the key to the document for logging
            processed_doc = process_and_embed_document(document)
            if processed_doc is None:
                logger.warning(f"Skipping document {key} due to processing failure")
                continue

            batch.append(processed_doc)
            
            if len(batch) >= BATCH_SIZE:
                try:
                    logger.debug(f"Upserting batch of {len(batch)} vectors to Pinecone")
                    index.upsert(vectors=batch)
                    logger.info(f"Successfully upserted batch of {len(batch)} vectors")
                except Exception as e:
                    logger.error(f"Error upserting batch: {e}")
                batch = []
    
    if batch:
        try:
            logger.debug(f"Upserting final batch of {len(batch)} vectors to Pinecone")
            index.upsert(vectors=batch)
            logger.info(f"Successfully upserted final batch of {len(batch)} vectors")
        except Exception as e:
            logger.error(f"Error upserting final batch: {e}")

    logger.info(f"Indexing process completed. Total files processed: {total_files}")

if __name__ == "__main__":
    index_s3_bucket()
    logger.info("Indexing complete!")
