import re
import nltk
import ssl
from nltk.tokenize import sent_tokenize
import pandas as pd
import logging
from utils import extract_case_and_document_id
import os

logger = logging.getLogger(__name__)

# Set NLTK data path to \src\punkt
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'punkt'))

# Print NLTK data path for debugging
logger.info(f"NLTK data path: {nltk.data.path}")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Check if punkt is available
if not nltk.data.find('tokenizers/punkt'):
    logger.warning("Punkt not found. Attempting to download.")
    try:
        nltk.download('punkt', download_dir=os.path.join(os.path.dirname(__file__), 'punkt'), quiet=True)
    except Exception as e:
        logger.error(f"Failed to download punkt: {e}")

def preprocess_data(df):
    def clean_text(text):
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_sentences(text):
        try:
            return sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Error in NLTK sentence tokenization: {str(e)}. Using fallback method.")
            return fallback_sentence_tokenize(text)

    # Extract case_id, document_id, and page_number
    df[['case_id', 'document_id', 'page_number']] = df['Key'].apply(lambda x: pd.Series(extract_case_and_document_id(x)))

    # Use .loc for assignments
    df.loc[:, 'cleaned_text'] = df['Content'].apply(clean_text)
    df.loc[:, 'sentences'] = df['cleaned_text'].apply(tokenize_sentences)
    return df

def fallback_sentence_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text)

# Test punkt tokenizer
test_text = "This is a test sentence. This is another one."
try:
    tokenized = sent_tokenize(test_text)
    logger.info(f"Punkt tokenizer test successful. Result: {tokenized}")
except Exception as e:
    logger.error(f"Punkt tokenizer test failed: {e}")
