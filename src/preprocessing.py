import re
import nltk
import ssl
from nltk.tokenize import sent_tokenize
import pandas as pd
import logging
from utils import extract_case_and_document_id
import os
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Any
from tqdm import tqdm
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Set NLTK data path and initialize
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'punkt'))

def initialize_nltk():
    """Initialize NLTK resources safely."""
    try:
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Check and download punkt if needed
        if not nltk.data.find('tokenizers/punkt'):
            logger.info("Downloading punkt tokenizer...")
            nltk.download('punkt', download_dir=os.path.join(os.path.dirname(__file__), 'punkt'), quiet=True)
            
    except Exception as e:
        logger.error(f"Failed to initialize NLTK: {e}")

# Initialize NLTK at module load
initialize_nltk()

def preprocess_data(data, tokenizer):
    """
    Preprocess data using HuggingFace datasets for efficient GPU pipeline processing.
    """
    try:
        # Convert to Dataset if it's not already
        if not isinstance(data, Dataset):
            if isinstance(data, dict):
                dataset = Dataset.from_dict(data)
            else:
                raise ValueError("Input data must be either a Dataset or a dictionary")
        else:
            dataset = data

        def preprocess_function(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            """Process a batch of examples."""
            try:
                batch_size = len(examples['Content'])
                
                # Clean text with validation
                cleaned_texts = []
                for text in examples['Content']:
                    if not isinstance(text, str):
                        logger.warning(f"Non-string text encountered: {type(text)}")
                        cleaned_texts.append("")
                    else:
                        cleaned_texts.append(clean_text(text))
                
                # Extract IDs from keys with validation
                case_ids, doc_ids, page_nums = [], [], []
                for key in examples['Key']:
                    if not isinstance(key, str):
                        logger.warning(f"Invalid key format: {key}")
                        case_ids.append('unknown')
                        doc_ids.append('unknown')
                        page_nums.append('UNKNOWN')
                    else:
                        case_id, doc_id, page_num = extract_case_and_document_id(key)
                        case_ids.append(case_id)
                        doc_ids.append(doc_id)
                        page_nums.append(page_num)
                
                # Tokenize text with length validation
                tokenized = tokenizer(
                    cleaned_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors=None
                )
                
                # Ensure all lists are of the same length
                assert len(cleaned_texts) == len(case_ids) == len(doc_ids) == len(page_nums) == batch_size, \
                    "Inconsistent lengths in processed data"
                
                # Convert all outputs to lists or numpy arrays
                return {
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'cleaned_text': cleaned_texts,
                    'case_id': case_ids,
                    'document_id': doc_ids,
                    'page_number': page_nums,
                    'original_text': examples['Content'],
                    'Key': examples['Key']
                }
            except Exception as e:
                logger.error(f"Error in preprocess_function: {str(e)}")
                raise

        # Process the entire dataset in batches
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
            desc="Preprocessing documents",
            num_proc=4  # Use multiple processes for CPU operations
        )
        
        # Validate processed dataset
        required_columns = ['input_ids', 'attention_mask', 'cleaned_text', 'case_id', 
                          'document_id', 'page_number', 'original_text', 'Key']
        missing_columns = [col for col in required_columns if col not in processed_dataset.column_names]
        if missing_columns:
            raise ValueError(f"Missing columns in processed dataset: {missing_columns}")
        
        logger.info(f"Preprocessed {len(processed_dataset)} documents")
        return processed_dataset

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    try:
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return ""

def create_sentence_chunks(text: str, max_length: int = 512) -> List[str]:
    """Create overlapping chunks of sentences that fit within max_length."""
    try:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    except Exception as e:
        logger.error(f"Error creating sentence chunks: {str(e)}")
        return [text[:max_length]]

# Remove test code from module level
