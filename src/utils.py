import yaml
import logging
from functools import wraps
import time
import itertools
from datasets import Dataset
from typing import Dict, Any, List
import json
import sys

logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load configuration globally
config = load_config('config/config.yaml')
metadata_config = config.get('metadata', {})
MAX_METADATA_SIZE = metadata_config.get('MAX_METADATA_SIZE', 40000)
ENTITY_LIMIT = metadata_config.get('entity_limit', 5)
TEXT_LIMIT = metadata_config.get('text_limit', 100)

def setup_logging(log_level='INFO'):
    """Set up logging configuration."""
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
        # Remove any previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    return logging.getLogger(__name__)

def timer(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def load_search_criteria(config):
    """Load search criteria from config."""
    return config.get('search_criteria', {})

def get_s3_txt_files_in_folders(s3_client, bucket, prefixes):
    """Get list of .txt files from the S3 bucket that are inside specified folders."""
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        txt_files = []
        for prefix in prefixes:
            response_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
            for page in response_iterator:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.txt'):
                        txt_files.append(key)
        return txt_files
    except Exception as e:
        logger.error(f"Error in get_s3_txt_files_in_folders: {str(e)}")
        return []

def extract_case_and_document_id(key):
    """Extract case_id, document_id, and page_number from the S3 object key."""
    try:
        # Split the key by '/'
        key_parts = key.split('/')
        if len(key_parts) >= 2:
            case_id = key_parts[-2]  # <case_id>
            file_name = key_parts[-1]  # <document_id>_full.txt or <document_id>_page<n>.txt

            # Extract document_id and page_number
            if '_full.txt' in file_name:
                document_id = file_name.replace('_full.txt', '')
                page_number = 'FULL'
            elif '_page' in file_name:
                document_id, page_info = file_name.split('_page', 1)
                page_number = page_info.lstrip('_').replace('.txt', '').replace('.pdf', '')
            else:
                document_id = file_name.replace('.txt', '').replace('.pdf', '')
                page_number = 'UNKNOWN'

            return case_id, document_id, page_number
        else:
            logger.error(f"Unexpected key format: {key}")
            return 'unknown_case_id', 'unknown_document_id', 'UNKNOWN'
    except Exception as e:
        logger.error(f"Error extracting case_id, document_id, and page_number from key {key}: {e}")
        return 'unknown_case_id', 'unknown_document_id', 'UNKNOWN'

def prepare_metadata(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare metadata for Pinecone upsert, ensuring all fields are strings and size does not exceed MAX_METADATA_SIZE."""
    try:
        metadata_list = []
        num_examples = len(examples['cleaned_text'])
        entities_list = examples.get('entities', [])
        relationships_list = examples.get('relationships', [])

        for i in range(num_examples):
            # Create base metadata
            metadata = {
                'text': str(examples['cleaned_text'][i])[:TEXT_LIMIT],
                'case_id': str(examples['case_id'][i]),
                'document_id': str(examples['document_id'][i]),
                'page_number': str(examples['page_number'][i]),
            }

            # Add entities to metadata under respective keys if they exist
            if i < len(entities_list) and entities_list[i]:
                entities = entities_list[i]
                for entity_type, entity_list in entities.items():
                    if isinstance(entity_list, list) and entity_list:
                        # Ensure all entities are strings and filter out None values
                        cleaned_entities = [str(e) for e in entity_list if e is not None]
                        # Limit number of entities per type
                        limited_entities = cleaned_entities[:ENTITY_LIMIT]
                        # Add to metadata if list is not empty
                        if limited_entities:
                            metadata[entity_type] = limited_entities

            # Add relationships to metadata under separate keys if they exist
            if i < len(relationships_list) and relationships_list[i]:
                relationships = relationships_list[i]
                if isinstance(relationships, dict):
                    # Handle 'syntactic_relationships'
                    if 'syntactic' in relationships:
                        syntactic_rels = [
                            f"{str(t[0])}_{str(t[1])}_{str(t[2])}"
                            for t in relationships.get('syntactic', [])
                            if t is not None
                        ]
                        if syntactic_rels:
                            metadata['syntactic_relationships'] = syntactic_rels
                    # Handle 'custom_relationships'
                    if 'custom' in relationships:
                        custom_rels = [
                            str(t) for t in relationships.get('custom', []) if t is not None
                        ]
                        if custom_rels:
                            metadata['custom_relationships'] = custom_rels

            # Ensure all metadata values are acceptable types and remove any None values
            metadata = validate_metadata(metadata)

            # Ensure metadata size does not exceed MAX_METADATA_SIZE
            metadata_json = json.dumps(metadata)
            metadata_size = sys.getsizeof(metadata_json)
            if metadata_size > MAX_METADATA_SIZE:
                logger.warning(f"Metadata size before trimming: {metadata_size} bytes")

                # Trim entity lists for each entity type
                entity_keys = [key for key in metadata.keys() if key in entities_list[i]]
                for key in entity_keys:
                    if isinstance(metadata[key], list) and len(metadata[key]) > 1:
                        metadata[key] = metadata[key][:1]  # Keep only the first entity

                # Trim relationships
                if 'syntactic_relationships' in metadata:
                    metadata['syntactic_relationships'] = metadata['syntactic_relationships'][:1]
                if 'custom_relationships' in metadata:
                    metadata['custom_relationships'] = metadata['custom_relationships'][:1]

                # Recalculate metadata size
                metadata_json = json.dumps(metadata)
                metadata_size = sys.getsizeof(metadata_json)

                logger.warning(f"Metadata size after trimming entities and relationships: {metadata_size} bytes")

                if metadata_size > MAX_METADATA_SIZE:
                    # As a last resort, shorten the text
                    metadata['text'] = metadata['text'][:50]
                    metadata_json = json.dumps(metadata)
                    metadata_size = sys.getsizeof(metadata_json)
                    logger.warning(f"Metadata size after trimming text: {metadata_size} bytes")

                    if metadata_size > MAX_METADATA_SIZE:
                        logger.error(f"Metadata size still exceeds MAX_METADATA_SIZE after trimming: {metadata_size} bytes")
                        raise ValueError("Unable to reduce metadata size below MAX_METADATA_SIZE")

            metadata_list.append(metadata)

        # Return all original examples along with the new 'metadata' field
        return {**examples, 'metadata': metadata_list}
    except Exception as e:
        logger.error(f"Error preparing metadata: {str(e)}")
        # Handle error appropriately, e.g., return empty metadata
        metadata_list = [{
            'text': '',
            'case_id': 'unknown',
            'document_id': 'unknown',
            'page_number': 'unknown',
        } for _ in range(len(examples.get('cleaned_text', [])))]
        return {**examples, 'metadata': metadata_list}

def generate_report(results: List[Dict[str, Any]]):
    """
    Generate a report from the final ranked results.
    """
    if not results:
        print("No results to display.")
        return

    print("\nSearch Results Summary:")
    print("-" * 80)
    for result in results:
        print(f"Document ID: {result.get('document_id', 'N/A')}")
        print(f"Case ID: {result.get('case_id', 'N/A')}")
        print(f"Page Number: {result.get('page_number', 'N/A')}")
        print(f"Score: {result.get('score', 0.0):.4f}")
        print(f"Text Snippet: {result.get('text', '')[:200]}...")  # Display first 200 characters
        print("-" * 80)

def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean metadata to ensure it doesn't contain None values."""
    clean_metadata = {}
    for key, value in metadata.items():
        if value is None:
            # Remove keys with None values
            continue
        elif isinstance(value, list):
            # Ensure all elements are strings and remove None values
            cleaned_list = [str(v) for v in value if v is not None]
            if cleaned_list:
                clean_metadata[key] = cleaned_list
        else:
            # Convert other types to strings
            clean_metadata[key] = str(value)
    return clean_metadata
