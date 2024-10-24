from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import logging
from utils import load_config
from datasets import Dataset
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Load configuration
config = load_config('config/config.yaml')
model_config = config.get('models', {})

# Initialize ClinicalBERT model and tokenizer for NER
ner_model_name = model_config.get('ner', "samrawal/bert-base-uncased_clinical-ner")
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

def process_entities_batch(examples: Dict[str, List[Any]], pipeline) -> Dict[str, List[Any]]:
    """Process a batch of examples through NER pipeline."""
    try:
        # Get device
        device = next(pipeline.model.parameters()).device
        
        # Process texts through pipeline
        # Note: NER pipeline handles its own tokenization and padding
        # We only pass the texts and batch_size
        ner_results = pipeline(
            examples['cleaned_text'],
            batch_size=32
        )
        
        # Extract entities for each text
        entities_list = []
        for text_entities in ner_results:
            entities = {}
            for entity in text_entities:
                ent_label = entity['entity_group']
                ent_text = entity['word']
                
                if ent_text is not None:
                    if ent_label not in entities:
                        entities[ent_label] = []
                    entities[ent_label].append(ent_text)
                else:
                    logger.warning(f"Encountered None entity text for label {ent_label}")
            
            # Remove duplicates and sort
            for key in entities:
                entities[key] = sorted(list(set(entities[key])))
            
            # Convert all values to strings
            string_entities = {k: [str(v) for v in vals] for k, vals in entities.items()}
            entities_list.append(string_entities)
        
        # Add entities to examples
        return {
            **examples,
            'entities': entities_list
        }
        
    except Exception as e:
        logger.error(f"Error in batch entity processing: {str(e)}")
        return {**examples, 'entities': [{} for _ in range(len(examples['cleaned_text']))]}

def extract_entities(dataset: Dataset) -> Dataset:
    """Extract entities using the pipeline's support for datasets for better GPU utilization."""
    try:
        # Initialize NER pipeline with correct parameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ner_pipeline = pipeline(
            task="ner",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
            device=device,
            framework="pt"
        )

        # Process dataset directly through pipeline
        ner_results = ner_pipeline(dataset['cleaned_text'])

        # Process results into entities
        entities_list = []
        for text_entities in ner_results:
            entities = {}
            for entity in text_entities:
                ent_label = entity['entity_group'].lower()  # Convert to lowercase
                ent_text = entity['word']
                if ent_text is not None:
                    if ent_label not in entities:
                        entities[ent_label] = []
                    entities[ent_label].append(ent_text)
                else:
                    logger.warning(f"Encountered None entity text for label {ent_label}")
            
            # Remove duplicates and sort
            for key in entities:
                entities[key] = sorted(list(set(entities[key])))
            
            entities_list.append(entities)

        # Add entities to dataset
        dataset = dataset.add_column('entities', entities_list)
        return dataset

    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}")
        raise
