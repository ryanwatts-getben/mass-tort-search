from transformers import AutoTokenizer, AutoModel
import torch
from utils import ensure_consistent_dataframe, load_config
import logging

logger = logging.getLogger(__name__)

# Load pre-trained ClinicalBERT model
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

# Load configuration
config = load_config('config/config.yaml')
metadata_config = config.get('metadata', {})
TEXT_LIMIT = metadata_config.get('text_limit', 100)
ENTITY_LIMIT = metadata_config.get('entity_limit', 5)

def vectorize_documents(df):
    df = ensure_consistent_dataframe(df)
    vectors = []
    metadata_list = []
    
    for _, row in df.iterrows():
        # Get all entity types as extracted by ClinicalBERT
        entities = row['entities'] if isinstance(row['entities'], dict) else {}
        
        # Combine text and all entities for embedding
        combined_text = f"{row['cleaned_text']} {' '.join([' '.join(entities.get(key, [])) for key in entities])}"
        
        # Tokenize and get embeddings
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embedding
        vector = outputs.last_hidden_state[0][0].numpy()
        
        # Create metadata with dynamically extracted entity types
        metadata = {
            'text': row['cleaned_text'][:TEXT_LIMIT],
            'case_id': row['case_id'],
            'document_id': row['document_id'],
            'page_number': row['page_number'],
        }
        for entity_type, entity_list in entities.items():
            metadata[entity_type] = entity_list[:ENTITY_LIMIT]
        
        vectors.append(vector)
        metadata_list.append(metadata)
    
    # Use .loc for assignments
    df.loc[:, 'vector'] = vectors
    df.loc[:, 'metadata'] = metadata_list
    
    # Log any rows with unknown case_id or document_id
    unknown_rows = df[df['metadata'].apply(lambda x: x['case_id'] == 'unknown_case_id' or x['document_id'] == 'unknown_document_id')]
    if not unknown_rows.empty:
        logger.warning(f"Found {len(unknown_rows)} rows with unknown case_id or document_id")
        for _, row in unknown_rows.iterrows():
            logger.warning(f"Row index: {row.name}, case_id: {row['metadata']['case_id']}, document_id: {row['metadata']['document_id']}, page_number: {row['metadata']['page_number']}")
    
    return df
