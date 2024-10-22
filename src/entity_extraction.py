from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import logging

logger = logging.getLogger(__name__)

# Initialize ClinicalBERT model and tokenizer for NER
ner_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
ner_model = AutoModelForTokenClassification.from_pretrained("medicalai/ClinicalBERT")

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

    entities = {
        'icd10_codes': [],
        'symptoms': [],
        'lab_results': [],
        'other_conditions': [],
        'diagnostic_procedures': [],
        'treatment_options': [],
        'complications': []
    }

    # Split text into smaller chunks
    sentences = text.split('. ')
    current_chunk = ''
    
    for sentence in sentences:
        temp_chunk = current_chunk + sentence + '. '
        tokenized = ner_tokenizer(temp_chunk, return_tensors='pt')
        
        if tokenized.input_ids.shape[1] <= max_length:
            current_chunk = temp_chunk
        else:
            ner_results = ner_pipeline(current_chunk)
            for entity in ner_results:
                ent_text = entity['word']
                ent_label = entity['entity_group']
                
                if ent_label in entities:
                    entities[ent_label].append(ent_text)
            
            current_chunk = sentence + '. '

    # Process remaining text
    if current_chunk:
        ner_results = ner_pipeline(current_chunk)
        for entity in ner_results:
            ent_text = entity['word']
            ent_label = entity['entity_group']
            
            if ent_label in entities:
                entities[ent_label].append(ent_text)

    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities

def extract_entities(df):
    """Process the entire DataFrame and extract entities from each row."""
    # Create a list to store entities for each row
    all_entities = []
    
    # Process each row in the DataFrame
    for _, row in df.iterrows():
        try:
            # Extract entities from cleaned_text
            entities = extract_entities_from_text(row['cleaned_text'])
            all_entities.append(entities)
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            # Add empty entities if processing fails
            all_entities.append({
                'icd10_codes': [],
                'symptoms': [],
                'lab_results': [],
                'other_conditions': [],
                'diagnostic_procedures': [],
                'treatment_options': [],
                'complications': []
            })
    
    # Add the entities to the DataFrame
    df['entities'] = all_entities
    
    return df