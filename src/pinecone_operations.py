from pinecone.grpc import PineconeGRPC as Pinecone
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import logging
from utils import load_config, setup_logging
from datasets import Dataset
from typing import Dict, List, Any
from tqdm import tqdm
import numpy as np
import sys
import json


# Set up logging
logger = setup_logging()

# Load configuration
config = load_config('config/config.yaml')
model_config = config.get('models', {})
bert_model_name = model_config.get('bert', "medicalai/ClinicalBERT")
metadata_config = config.get('metadata', {})
MAX_MESSAGE_SIZE = config.get('pinecone', {}).get('grpc', {}).get('max_message_length', 4194304)  # 4MB default

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
index_name = config.get('pinecone', {}).get('index_name', 'medical-records4')

# Initialize the Pinecone client with GRPC
pc = Pinecone(api_key=PINECONE_API_KEY)

def validate_dataset(dataset: Dataset) -> bool:
    """Validate that dataset contains all required fields."""
    try:
        required_fields = ['vector', 'cleaned_text', 'case_id', 'document_id', 'page_number']
        missing_fields = [field for field in required_fields if field not in dataset.features]
        
        if missing_fields:
            logger.error(f"Missing required fields in dataset: {missing_fields}")
            return False
            
        # Validate vector dimension
        if 'vector' in dataset.features:
            sample_vector = dataset[0]['vector']
            if not isinstance(sample_vector, list) or len(sample_vector) != 768:
                logger.error(f"Invalid vector dimension: {len(sample_vector) if isinstance(sample_vector, list) else type(sample_vector)}")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        return False

def prepare_vectors(examples: Dict[str, List[Any]], batch_start_idx: int = 0) -> List[Dict]:
    """Prepare vectors for Pinecone upload, using pre-prepared metadata."""
    try:
        vectors = []
        for i in range(len(examples['vector'])):
            try:
                # Validate vector
                vector = examples['vector'][i]
                if not isinstance(vector, list) or len(vector) != 768:
                    logger.warning(f"Invalid vector at index {i}: Expected list of length 768.")
                    continue

                # Check if vector is all zeros
                if all(v == 0.0 for v in vector):
                    logger.warning(f"Zero vector at index {i}, indicating potential issues in vectorization.")
                    continue  # Skip zero vectors

                # Use prepared metadata
                metadata = examples['metadata'][i]

                # Create vector data
                vector_data = {
                    'id': f"{metadata['case_id']}_{metadata['document_id']}_{metadata['page_number']}_{batch_start_idx + i}",
                    'values': vector,
                    'metadata': metadata
                }
                vectors.append(vector_data)

            except Exception as e:
                logger.error(f"Error processing vector {i}: {str(e)}")
                continue

        return vectors

    except Exception as e:
        logger.error(f"Error preparing vectors: {str(e)}")
        return []

def upload_to_pinecone(dataset: Dataset, index_name: str):
    """Upload vectors to Pinecone, adjusting batch size to prevent message size exceeding limit."""
    try:
        # Validate dataset
        if not validate_dataset(dataset):
            raise ValueError("Invalid dataset structure")

        # Get index
        index = pc.Index(index_name)

        # Start with initial batch size
        batch_size = 100  # Start with 100 and adjust if necessary
        total_uploaded = 0

        # Use tqdm for progress tracking
        with tqdm(total=len(dataset), desc="Uploading to Pinecone") as pbar:
            i = 0
            while i < len(dataset):
                try:
                    # Get batch
                    end_idx = min(i + batch_size, len(dataset))
                    batch = dataset.select(range(i, end_idx))

                    # Prepare vectors
                    vectors = prepare_vectors(batch, batch_start_idx=i)

                    if vectors:
                        # Check message size
                        message_size = sys.getsizeof(json.dumps([v['metadata'] for v in vectors]))
                        if message_size > MAX_MESSAGE_SIZE:
                            # Reduce batch size
                            logger.warning(f"Batch size {batch_size} too large with message size {message_size} bytes, reducing batch size")
                            batch_size = max(1, batch_size // 2)
                            continue  # Retry with smaller batch size

                        # Upsert following Pinecone format
                        response = index.upsert(
                            vectors=vectors,
                            namespace="default"
                        )

                        # Update progress
                        uploaded_count = len(vectors)  # We know how many we sent
                        total_uploaded += uploaded_count
                        pbar.update(uploaded_count)

                        logger.debug(f"Batch {i//batch_size + 1}: Uploaded {uploaded_count} vectors")

                        # Increase batch size slowly to improve performance
                        batch_size = min(batch_size + 10, 100)  # Don't exceed 100 as per recommendations

                    i += len(batch)  # Move to next batch

                except Exception as e:
                    logger.error(f"Error uploading batch starting at index {i}: {str(e)}")
                    if batch_size > 1:
                        # Reduce batch size and retry
                        batch_size = max(1, batch_size // 2)
                        logger.warning(f"Reducing batch size to {batch_size} and retrying")
                    else:
                        # Skip this vector if batch size is 1 and still failing
                        i += 1
                        pbar.update(1)
                        logger.error(f"Skipping vector at index {i} due to error")
                    continue

        logger.info(f"Successfully uploaded {total_uploaded} vectors to Pinecone")

    except Exception as e:
        logger.error(f"Error in upload_to_pinecone: {str(e)}")
        raise

def search_pinecone(search_criteria: Dict, index_name: str) -> Dict:
    """Search Pinecone index using GRPC client."""
    try:
        index = pc.Index(index_name)
        
        # Convert search criteria to vector
        search_vector = vectorize_search_criteria(search_criteria)
        
        # Query index
        results = index.query(
            vector=search_vector,
            top_k=100,
            include_metadata=True,
            namespace="default"
        )
        
        return {'matches': results.matches}
        
    except Exception as e:
        logger.error(f"Error in search_pinecone: {str(e)}")
        raise

def vectorize_search_criteria(criteria: Dict) -> List[float]:
    """Vectorize search criteria."""
    try:
        # Initialize tokenizer and model if not already done
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        model = AutoModel.from_pretrained(bert_model_name)
        
        # Combine criteria into text
        text = ' '.join(f"{k}: {v}" for k, v in criteria.items())
        
        # Tokenize and get embeddings
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            vector = outputs.last_hidden_state[0][0].numpy().tolist()
        
        return vector
        
    except Exception as e:
        logger.error(f"Error vectorizing search criteria: {str(e)}")
        raise

def delete_from_pinecone(document_ids: List[str], index_name: str):
    """Delete vectors from Pinecone using GRPC client."""
    try:
        index = pc.Index(index_name)
        index.delete(ids=document_ids)
        logger.info(f"Successfully deleted {len(document_ids)} vectors")
        
    except Exception as e:
        logger.error(f"Error in delete_from_pinecone: {str(e)}")
        raise

