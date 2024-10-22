import boto3
import json
from tqdm import tqdm
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
from pinecone import ServerlessSpec
import io
import aioboto3
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='indexing.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
AWS_REGION = os.getenv('AWS_REGION')

# Initialize clients
s3 = boto3.client('s3', region_name=AWS_REGION)
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

# Initialize the Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize ClinicalBERT model and tokenizer for embeddings
embedding_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
embedding_model = AutoModel.from_pretrained("medicalai/ClinicalBERT").to(device)

# Initialize ClinicalBERT model and tokenizer for NER
ner_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
ner_model = AutoModelForTokenClassification.from_pretrained("medicalai/ClinicalBERT").to(device)

# Initialize NER pipeline
ner_pipeline = pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

# Constants
BUCKET_NAME = os.getenv('AWS_LOVELY_BUCKET')
INDEX_NAME = 'lovely-medicalai-clinicalbert'
BATCH_SIZE = 32  # Adjust based on your GPU memory
EMBEDDING_DIMENSION = 768  # Dimension for Bio_ClinicalBERT

def get_embedding(text):
    inputs = embedding_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

def process_batch(texts):
    inputs = embedding_tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = embeddings.cpu().numpy()
    return embeddings

def extract_entities(text):
    max_length = 512  # Model's max length

    entities = {
        'icd10_codes': [],
        'symptoms': [],
        'lab_results': [],
        'other_conditions': [],
        'diagnostic_procedures': [],
        'treatment_options': [],
        'complications': []
    }

    # Split text into smaller chunks that won't exceed max_length after tokenization
    sentences = text.split('. ')
    current_chunk = ''
    for sentence in sentences:
        # Temporarily add the sentence to the current chunk
        temp_chunk = current_chunk + sentence + '. '
        # Tokenize the temp_chunk
        tokenized = ner_tokenizer(temp_chunk, truncation=False, return_tensors='pt')
        if tokenized.input_ids.shape[1] <= max_length:
            # If within limit, set current_chunk to temp_chunk
            current_chunk = temp_chunk
        else:
            # Process the current_chunk
            ner_results = ner_pipeline(current_chunk, truncation=True, max_length=max_length)
            # Aggregate entities
            for entity in ner_results:
                ent_text = entity['word']
                ent_label = entity['entity_group']
                # Map labels to your categories
                if ent_label == 'icd10_codes':
                    entities['icd10_codes'].append(ent_text)
                elif ent_label == 'symptoms':
                    entities['symptoms'].append(ent_text)
                elif ent_label == 'lab_results':
                    entities['lab_results'].append(ent_text)
                elif ent_label == 'other_conditions':
                    entities['other_conditions'].append(ent_text)
                elif ent_label == 'diagnostic_procedures':
                    entities['diagnostic_procedures'].append(ent_text)
                elif ent_label == 'treatment_options':
                    entities['treatment_options'].append(ent_text)
                elif ent_label == 'complications':
                    entities['complications'].append(ent_text)
            # Start a new chunk
            current_chunk = sentence + '. '
    # Process any remaining text
    if current_chunk:
        ner_results = ner_pipeline(current_chunk, truncation=True, max_length=max_length)
        for entity in ner_results:
            ent_text = entity['word']
            ent_label = entity['entity_group']
            # Map labels to your categories
            if ent_label == 'icd10_codes':
                entities['icd10_codes'].append(ent_text)
            elif ent_label == 'symptoms':
                entities['symptoms'].append(ent_text)
            elif ent_label == 'lab_results':
                entities['lab_results'].append(ent_text)
            elif ent_label == 'other_conditions':
                entities['other_conditions'].append(ent_text)
            elif ent_label == 'diagnostic_procedures':
                entities['diagnostic_procedures'].append(ent_text)
            elif ent_label == 'treatment_options':
                entities['treatment_options'].append(ent_text)
            elif ent_label == 'complications':
                entities['complications'].append(ent_text)
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities

def process_document_text(document):
    key = document['Key']
    try:
        content = document['Body'].read()
        text = content.decode('utf-8')
        
        # Initialize data variable
        data = None
        
        # Check if the content is JSON
        try:
            data = json.loads(text)
            # Return the JSON string as is for entity extraction
            return text
        except json.JSONDecodeError:
            # If not JSON, proceed with text processing
            pass

        processed_text = ""
        if isinstance(data, dict):
            for condition, details in data.items():
                processed_text += f"Condition: {condition}\n"
                if isinstance(details, dict):
                    for key, value in details.items():
                        processed_text += f"{key}: {value}\n"
                else:
                    processed_text += f"Details: {details}\n"
                processed_text += "\n"
        else:
            processed_text = text
        
        return processed_text
    except UnicodeDecodeError as e:
        logger.warning(f"utf-8 decoding failed for document {key}: {e}")
        try:
            text = content.decode('latin-1')
            return text
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode document {key}: {e}")
            return None
    except Exception as e:
        logger.error(f"Error processing document {key}: {e}")
        return None

def extract_case_id(key):
    key_without_suffix = key.replace('_full.txt', '')
    key_parts = key_without_suffix.split('/')
    return key_parts[-2] if len(key_parts) >= 2 else 'unknown_case_id'

def extract_document_id(key):
    key_without_suffix = key.replace('_full.txt', '')
    key_parts = key_without_suffix.split('/')
    return key_parts[-1]

def extract_case_and_document_id(key):
    try:
        # Remove '_full.txt' suffix
        key_without_suffix = key.replace('_full.txt', '')
        # Split the key by '/'
        key_parts = key_without_suffix.split('/')
        if len(key_parts) >= 2:
            case_id = key_parts[-2]  # <directory>
            document_id = key_parts[-1]  # <filename>
        else:
            case_id = 'unknown_case_id'
            document_id = key_parts[-1]
        logger.debug(f"Extracted case_id: {case_id}, document_id: {document_id}")
    except Exception as e:
        logger.error(f"Error extracting case_id and document_id from key {key}: {e}")
        case_id = 'unknown_case_id'
        document_id = 'unknown_document_id'
    return case_id, document_id

def extract_metadata(document):
    processed_text = process_document_text(document)
    if processed_text is None:
        return None

    # Extract entities from the processed text
    entities = extract_entities(processed_text)
    
    # Extract case_id and document_id from the key
    key = document['Key']
    case_id, document_id = extract_case_and_document_id(key)
    
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
    
    # Tokenize and ensure the sequence length is exactly 512
    inputs = tokenizer(processed_text, 
                       return_tensors="pt", 
                       max_length=512, 
                       truncation=True, 
                       padding='max_length')
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the last hidden state as the document embedding
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # Include extracted entities in metadata
    metadata = {
        'text': processed_text[:1000],
        'case_id': case_id,
        'document_id': document_id,
        'icd10_codes': entities['icd10_codes'],
        'symptoms': entities['symptoms'],
        'lab_results': entities['lab_results'],
        'other_conditions': entities['other_conditions'],
        'diagnostic_procedures': entities['diagnostic_procedures'],
        'treatment_options': entities['treatment_options'],
        'complications': entities['complications']
    }
    return {
        'processed_text': processed_text,
        'metadata': metadata,
        'Key': key,
        'embeddings': embeddings.tolist()
    }
def process_and_embed_documents(documents):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_metadata, documents))
    texts = []
    metadatas = []
    keys = []
    for result in results:
        if result is not None:
            texts.append(result['processed_text'])
            metadatas.append(result['metadata'])
            keys.append(result['Key'])
    embeddings = process_batch(texts)
    processed_docs = []
    for i, embedding in enumerate(embeddings):
        processed_docs.append({
            'id': keys[i],
            'values': embedding.tolist(),
            'metadata': metadatas[i]
        })
    return processed_docs

async def fetch_document(s3_client, bucket_name, key):
    # Asynchronously fetch the document from S3
    try:
        response = await s3_client.get_object(Bucket=bucket_name, Key=key)
        content = await response['Body'].read()
        return {'Key': key, 'Body': io.BytesIO(content)}
    except Exception as e:
        logger.error(f"Error fetching document {key} from S3: {e}")
        return None

async def fetch_documents(bucket_name, keys):
    # Create an aioboto3 Session
    session = aioboto3.Session()
    # Use the session to create a client
    async with session.client('s3', region_name=AWS_REGION) as s3_client:
        tasks = [fetch_document(s3_client, bucket_name, key) for key in keys]
        documents = await asyncio.gather(*tasks)
        # Filter out None results in case of errors
        documents = [doc for doc in documents if doc is not None]
    return documents

async def index_s3_bucket():
    logger.info("Starting indexing process")
    try:
        existing_indexes = pc.list_indexes()
        logger.debug(f"Existing Pinecone indexes: {existing_indexes}")
        if INDEX_NAME not in existing_indexes.names():
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
            logger.info(f"Index '{INDEX_NAME}' already exists, skipping creation")
    except Exception as e:
        logger.error(f"Error checking/creating index '{INDEX_NAME}': {e}")
        return

    try:
        index = pc.Index(INDEX_NAME)
        logger.info(f"Successfully connected to index '{INDEX_NAME}'")
    except Exception as e:
        logger.error(f"Error connecting to index '{INDEX_NAME}': {e}")
        return
    
    paginator = s3.get_paginator('list_objects_v2')
    logger.debug(f"Listing objects in S3 bucket '{BUCKET_NAME}'")
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME)

    total_files = 0
    keys_batch = []
    for page in page_iterator:
        contents = page.get('Contents', [])
        logger.debug(f"Retrieved {len(contents)} objects from S3")
        for obj in contents:
            key = obj['Key']
            total_files += 1
            if not key.lower().endswith('.txt'):
                logger.info(f"Skipping non-text file: {key}")
                continue
            keys_batch.append(key)
            if len(keys_batch) >= BATCH_SIZE:
                # Use await instead of asyncio.run
                documents = await fetch_documents(BUCKET_NAME, keys_batch)
                processed_docs = process_and_embed_documents(documents)
                try:
                    logger.debug(f"Upserting batch of {len(processed_docs)} vectors to Pinecone")
                    index.upsert(vectors=processed_docs)
                    logger.info(f"Successfully upserted batch of {len(processed_docs)} vectors")
                except Exception as e:
                    logger.error(f"Error upserting batch: {e}")
                keys_batch = []
    # Handle any remaining keys
    if keys_batch:
        # Use await instead of asyncio.run
        documents = await fetch_documents(BUCKET_NAME, keys_batch)
        processed_docs = process_and_embed_documents(documents)
        try:
            logger.debug(f"Upserting final batch of {len(processed_docs)} vectors to Pinecone")
            index.upsert(vectors=processed_docs)
            logger.info(f"Successfully upserted final batch of {len(processed_docs)} vectors")
        except Exception as e:
            logger.error(f"Error upserting final batch: {e}")

    logger.info(f"Indexing process completed. Total files processed: {total_files}")

if __name__ == "__main__":
    asyncio.run(index_s3_bucket())
    logger.info("Indexing complete!")
