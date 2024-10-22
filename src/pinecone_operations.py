from pinecone import Pinecone, ServerlessSpec, PineconeException
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import logging
from utils import load_config, setup_logging

# Set up logging
logger = setup_logging()

# Load configuration
config = load_config('config/config.yaml')
metadata_config = config.get('metadata', {})
MAX_METADATA_SIZE = metadata_config.get('max_size', 40000)  # Default to 40KB if not specified

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')  # Default to 'aws' if not specified
PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')  # Default to 'us-east-1' if not specified

# Initialize the Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load pre-trained ClinicalBERT model
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
index_name = config.get('pinecone', {}).get('index_name', 'medical-records4')

def ensure_index_exists(index_name):
    """
    Ensure that the Pinecone index exists, creating it if necessary.
    """
    try:
        existing_indexes = pc.list_indexes()
        if index_name not in existing_indexes:
            logger.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=768,  # ClinicalBERT dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            logger.info(f"Index {index_name} created successfully.")
        else:
            logger.info(f"Index {index_name} already exists.")
    except PineconeException as e:
        if "already exists" in str(e):
            logger.info(f"Index {index_name} already exists. Proceeding with existing index.")
        else:
            logger.error(f"Error ensuring index exists: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error ensuring index exists: {str(e)}")
        raise

def upload_to_pinecone(processed_data, index_name):
    """
    Upload vectors and metadata to Pinecone index.
    """
    try:
        # Ensure the index exists before processing batches
        ensure_index_exists(index_name)
        
        index = pc.Index(index_name)
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(processed_data), batch_size):
            batch = processed_data[i:i+batch_size]
            vectors = []
            
            for _, row in batch.iterrows():
                # Ensure metadata size is within limits
                metadata = row['metadata']
                metadata_size = sum(len(str(v)) for v in metadata.values())
                if metadata_size > MAX_METADATA_SIZE:
                    logger.warning(f"Metadata size exceeds {MAX_METADATA_SIZE} bytes for vector {row.name}. Truncating metadata.")
                    # Truncate metadata
                    while metadata_size > MAX_METADATA_SIZE:
                        for key in list(metadata.keys()):
                            if isinstance(metadata[key], list):
                                if len(metadata[key]) > 1:
                                    metadata[key] = metadata[key][:-1]
                                else:
                                    del metadata[key]
                            elif isinstance(metadata[key], str):
                                metadata[key] = metadata[key][:len(metadata[key])//2]
                        metadata_size = sum(len(str(v)) for v in metadata.values())
                
                vector_data = {
                    'id': str(row.name),  # Use the index as the ID
                    'values': row['vector'].tolist(),
                    'metadata': metadata
                }
                vectors.append(vector_data)
            
            logger.info(f"Upserting batch of {len(vectors)} vectors")
            try:
                index.upsert(vectors=vectors)
            except Exception as e:
                logger.error(f"Error upserting batch: {str(e)}")
                # Log the problematic vectors
                for vector in vectors:
                    logger.error(f"Problematic vector: {vector}")
                raise
            
    except Exception as e:
        logger.error(f"Error in upload_to_pinecone: {str(e)}")
        raise

def vectorize_search_criteria(criteria):
    """
    Vectorize the search criteria using ClinicalBERT.
    """
    try:
        # Combine criteria into a single text
        combined_text = ' '.join([f"{key}: {value}" for key, value in criteria.items()])
        
        # Tokenize and get BERT embeddings
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the [CLS] token embedding as the vector
        vector = outputs.last_hidden_state[0][0].numpy()
        return vector.tolist()
        
    except Exception as e:
        logger.error(f"Error in vectorize_search_criteria: {str(e)}")
        raise

def search_pinecone(search_criteria, index_name):
    """
    Search Pinecone index with the given criteria.
    """
    try:
        index = pc.Index(index_name)
        
        # Convert search criteria to vector (use the same vectorization process as documents)
        search_vector = vectorize_search_criteria(search_criteria)
        
        results = index.query(vector=search_vector, top_k=100, include_metadata=True)
        return {'matches': results['matches']}
        
    except Exception as e:
        logger.error(f"Error in search_pinecone: {str(e)}")
        raise

def delete_from_pinecone(document_ids, index_name):
    """
    Delete specific documents from Pinecone index.
    """
    try:
        index = pc.Index(index_name)
        index.delete(ids=document_ids)
        logger.info(f"Successfully deleted {len(document_ids)} documents from Pinecone")
        
    except Exception as e:
        logger.error(f"Error in delete_from_pinecone: {str(e)}")
        raise

def get_document_by_id(document_id, index_name):
    """
    Retrieve a specific document from Pinecone by its ID.
    """
    try:
        index = pc.Index(index_name)
        result = index.fetch(ids=[document_id])
        return result.vectors.get(document_id)
        
    except Exception as e:
        logger.error(f"Error in get_document_by_id: {str(e)}")
        raise
