from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import logging

# Set up logging
logger = logging.getLogger(__name__)

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

def upload_to_pinecone(df):
    """
    Upload vectors and metadata to Pinecone index.
    """
    try:
        index_name = "medical-records1"
        
        # Create index if it doesn't exist
        if index_name not in pc.list_indexes():
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
        
        index = pc.Index(index_name)
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            vectors = []
            
            for _, row in batch.iterrows():
                vector_data = {
                    'id': str(row.name),
                    'values': row['vector'].tolist(),
                    'metadata': {
                        'text': row['metadata']['text'],
                        'case_id': row['metadata']['case_id'],
                        'document_id': row['metadata']['document_id'],
                        'icd10_codes': row['metadata']['icd10_codes'],
                        'symptoms': row['metadata']['symptoms'],
                        'lab_results': row['metadata']['lab_results'],
                        'other_conditions': row['metadata']['other_conditions'],
                        'diagnostic_procedures': row['metadata']['diagnostic_procedures'],
                        'treatment_options': row['metadata']['treatment_options'],
                        'complications': row['metadata']['complications']
                    }
                }
                vectors.append(vector_data)
            
            logger.info(f"Upserting batch of {len(vectors)} vectors")
            index.upsert(vectors=vectors)
            
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

def search_pinecone(search_criteria):
    """
    Search Pinecone index with the given criteria.
    """
    try:
        index = pc.Index("medical-records1")
        
        # Convert search criteria to vector
        search_vector = vectorize_search_criteria(search_criteria)
        
        # Perform the search
        results = index.query(
            vector=search_vector,
            top_k=100,
            include_metadata=True
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in search_pinecone: {str(e)}")
        raise

def delete_from_pinecone(document_ids):
    """
    Delete specific documents from Pinecone index.
    """
    try:
        index = pc.Index("medical-records1")
        index.delete(ids=document_ids)
        logger.info(f"Successfully deleted {len(document_ids)} documents from Pinecone")
        
    except Exception as e:
        logger.error(f"Error in delete_from_pinecone: {str(e)}")
        raise

def get_document_by_id(document_id):
    """
    Retrieve a specific document from Pinecone by its ID.
    """
    try:
        index = pc.Index("medical-records1")
        result = index.fetch(ids=[document_id])
        return result.vectors.get(document_id)
        
    except Exception as e:
        logger.error(f"Error in get_document_by_id: {str(e)}")
        raise