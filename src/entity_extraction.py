from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import logging
from utils import load_config

logger = logging.getLogger(__name__)

# Load configuration
config = load_config('config/config.yaml')
model_config = config.get('models', {})

# Initialize ClinicalBERT model and tokenizer for NER
ner_model_name = model_config.get('ner', "samrawal/bert-base-uncased_clinical-ner")
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

# Initialize NER pipeline
device = 0 if torch.cuda.is_available() else -1
ner_pipeline = pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple",
    device=device
)

def extract_entities_from_text(text):
    """Extract entities from a single text string."""
    max_length = 512
    entities = {}

    # Split text into smaller chunks
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    all_ner_results = []
    for chunk in chunks:
        ner_results = ner_pipeline(chunk)
        all_ner_results.extend(ner_results)
        for entity in ner_results:
            ent_text = entity['word']
            ent_label = entity['entity_group']
            
            if ent_label not in entities:
                entities[ent_label] = []
            entities[ent_label].append(ent_text)

    # Remove duplicates and sort
    for key in entities:
        entities[key] = sorted(list(set(entities[key])))

    logger.info(f"Extracted entities: {entities}")
    logger.debug(f"Full NER results: {all_ner_results}")

    return entities

def extract_entities(batch):
    """Process the entire batch and extract entities from each row."""
    all_entities = []
    
    for _, row in batch.iterrows():
        try:
            entities = extract_entities_from_text(row['cleaned_text'])
            all_entities.append(entities)
            logger.info(f"Extracted entities for document {row.get('document_id', 'unknown')}: {entities}")
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            all_entities.append({})
    
    batch['entities'] = all_entities
    return batch
