import re
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK punkt: {str(e)}")

download_nltk_resources()

def preprocess_data(raw_data):
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(raw_data)
    
    # Ensure 'Content' column exists and rename it to 'text' if necessary
    if 'Content' in df.columns:
        df = df.rename(columns={'Content': 'text'})
    elif 'text' not in df.columns:
        raise ValueError("Input data does not contain 'Content' or 'text' column")

    # Apply cleaning and tokenization
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    try:
        df['sentences'] = df['cleaned_text'].apply(sent_tokenize)
    except LookupError:
        logger.warning("NLTK punkt tokenizer not available. Using fallback sentence splitting.")
        df['sentences'] = df['cleaned_text'].apply(fallback_sentence_tokenize)
    
    return df

def clean_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fallback_sentence_tokenize(text):
    # Simple fallback sentence tokenization
    return re.split(r'(?<=[.!?])\s+', text)
