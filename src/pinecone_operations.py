from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')  # Default to 'aws' if not specified
PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')  # Default to 'us-west-2' if not specified

# Initialize the Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load pre-trained BioBERT model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

def upload_to_pinecone(df):
    index_name = "medical-records"
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=768,  # BioBERT dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
    
    index = pc.Index(index_name)
    
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        vectors = [(str(row.name), row['vector'].tolist(), {'text': row['cleaned_text'][:1000]}) for _, row in batch.iterrows()]
        index.upsert(vectors=vectors)

def search_pinecone(search_criteria):
    index = pc.Index("medical-records")
    
    # Convert search criteria to vector (use the same vectorization process as documents)
    search_vector = vectorize_search_criteria(search_criteria)
    
    results = index.query(vector=search_vector, top_k=100, include_metadata=True)
    return results

def vectorize_search_criteria(criteria):
    """Vectorize the search criteria using BioBERT."""
    # Combine criteria into a single text
    combined_text = ' '.join([f"{key}: {value}" for key, value in criteria.items()])

    # Tokenize and get BERT embeddings
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding as the vector
    vector = outputs.last_hidden_state[0][0].numpy()
    return vector.tolist()  # Convert to list for Pinecone
