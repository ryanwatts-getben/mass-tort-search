import yaml
import logging
from functools import wraps
import time
import itertools

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(log_level='INFO'):
    """Set up logging configuration."""
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
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

def chunk_text(text, max_length):
    """Split text into chunks of specified maximum length."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def flatten_list(nested_list):
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist]

def safe_get(dictionary, key, default=None):
    """Safely get a value from a dictionary."""
    return dictionary.get(key, default)

def validate_data(df, required_columns):
    """Validate that DataFrame contains required columns."""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def clean_text(text):
    """Clean text by removing special characters and extra whitespace."""
    import re
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def get_s3_file_list(s3_client, bucket, prefix):
    """Get list of files from S3 bucket with given prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj['Key'] for obj in response.get('Contents', [])]

def batch_generator(iterable, batch_size):
    """Generate batches from an iterable."""
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def load_search_criteria():
    """Load search criteria from config.yaml."""
    import yaml
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config['search_criteria']

def generate_report(results):
    """Generate a report from the final ranked results."""
    # For example, print the top 10 results
    for result in results[:10]:
        print(f"Document ID: {result[0]}")
        print(f"Score: {result[1]:.4f}")
        print(f"Text Snippet: {result[2]['text'][:200]}")  # Display first 200 characters
        print("-" * 80)

def get_s3_txt_files(s3_client, bucket, prefix):
    """Get list of .txt files from S3 bucket with given prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.txt')]
