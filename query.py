import os
import time
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Adjust as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='search.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI
openai_client = OpenAI()

# Initialize Pinecone client
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Constants
INDEX_NAME = 'lovely'
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072  # Must match the dimension used when creating the index
TOP_K = 5  # Number of top results to return

def get_embedding(text, model=EMBEDDING_MODEL):
    try:
        response = openai_client.embeddings.create(
            model=model,
            input=text
        )
        # Access the embedding data correctly
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def search_pinecone(query_embedding):
    """Performs a similarity search in the Pinecone index using the query embedding."""
    # Connect to the Pinecone index
    index = pc.Index(INDEX_NAME)
    
    # Perform similarity search
    try:
        logger.debug(f"Performing similarity search with top_k={TOP_K}")
        response = index.query(
            vector=query_embedding,
            top_k=TOP_K,
            include_values=False,
            include_metadata=True
        )
        logger.info(f"Search completed successfully")
        return response['matches']
    except Exception as e:
        logger.error(f"Error during search: {e}")
        return []

def main():
    # Define the query (modify this variable with your search query)
    query = "LymphomaC82.00"
    
    # Get embedding for the query
    query_embedding = get_embedding(query)
    if query_embedding is None:
        logger.error("Failed to get embedding for the query")
        return
    
    # Search Pinecone index
    results = search_pinecone(query_embedding)
    
    # Display the results
    for match in results:
        score = match['score']
        metadata = match['metadata']
        text_snippet = metadata.get('text', '[No text]')
        print(f"Score: {score}")
        print(f"Text snippet: {text_snippet}")
        print("-" * 50)

if __name__ == "__main__":
    main()
