import spacy
import logging
from datasets import Dataset
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    raise

def extract_relationships(dataset: Dataset) -> Dataset:
    """Extract relationships using dataset operations."""
    try:
        def process_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            try:
                relationships = []
                
                # Validate input data
                if 'cleaned_text' not in examples:
                    raise ValueError("Missing cleaned_text in examples")
                if 'entities' not in examples:
                    raise ValueError("Missing entities in examples")
                
                # Process each text in the batch
                for text, entities in zip(examples['cleaned_text'], examples['entities']):
                    try:
                        # Validate text
                        if not isinstance(text, str):
                            logger.warning(f"Invalid text type: {type(text)}")
                            text = str(text) if text is not None else ""
                        
                        # Extract syntactic dependencies
                        doc = nlp(text[:1000000])  # Limit text length to prevent memory issues
                        deps = [(token.text, token.dep_, token.head.text) for token in doc]
                        
                        # Extract custom relationships
                        custom_rels = extract_custom_relationships(entities)
                        
                        relationships.append({
                            'syntactic': deps if deps else [],
                            'custom': custom_rels if custom_rels else []  # Ensure it's a list
                        })
                    except Exception as e:
                        logger.error(f"Error processing single text: {str(e)}")
                        relationships.append({'syntactic': [], 'custom': []})
                
                # Ensure relationships list matches batch size
                if len(relationships) != len(examples['cleaned_text']):
                    logger.error("Mismatch in relationships list length")
                    relationships.extend([{'syntactic': [], 'custom': []}] * 
                                      (len(examples['cleaned_text']) - len(relationships)))
                
                return {**examples, 'relationships': relationships}
                
            except Exception as e:
                logger.error(f"Error in process_batch: {str(e)}")
                return {
                    **examples,
                    'relationships': [{'syntactic': [], 'custom': []}] * len(examples['cleaned_text'])
                }
        
        # Process dataset in batches
        processed_dataset = dataset.map(
            process_batch,
            batched=True,
            batch_size=32,
            desc="Extracting relationships",
            num_proc=1  # SpaCy doesn't support multiprocessing well
        )
        
        # Validate output dataset
        if 'relationships' not in processed_dataset.column_names:
            raise ValueError("Relationship extraction failed to produce relationships column")
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error in relationship extraction: {str(e)}")
        raise

def extract_custom_relationships(entities: Dict) -> List[str]:
    """Extract custom relationships between entities."""
    try:
        relationships = []
        
        # Validate entities
        if not isinstance(entities, dict):
            logger.warning(f"Invalid entities type: {type(entities)}")
            return relationships  # Return empty list
        
        # Get all entity values in a list
        all_entities = []
        for category, items in entities.items():
            if isinstance(items, list):
                for item in items:
                    if item is not None:
                        all_entities.append((str(category), str(item)))
            else:
                logger.warning(f"Expected list for entities[{category}], got {type(items)}")
        
        # Create relationships between entities
        for i, (cat1, ent1) in enumerate(all_entities):
            for cat2, ent2 in all_entities[i+1:]:
                if ent1 != ent2:
                    relationship_type = determine_relationship_type(cat1, cat2)
                    relationship_str = f"{ent1}_{relationship_type}_{ent2}"
                    relationships.append(relationship_str)
        
        return relationships
        
    except Exception as e:
        logger.error(f"Error extracting custom relationships: {str(e)}")
        return []  # Return empty list on exception

def determine_relationship_type(category1: str, category2: str) -> str:
    """Determine the type of relationship between two entity categories."""
    try:
        # Convert categories to lowercase
        category1 = category1.lower()
        category2 = category2.lower()

        # Define relationship mapping
        relationship_map = {
            ('problem', 'test'): 'diagnosed_by',
            ('problem', 'treatment'): 'treated_by',
            ('problem', 'drug'): 'treated_with',
            ('problem', 'anatomy'): 'located_in'
        }

        # Get relationship type
        return relationship_map.get((category1, category2), 'related_to')

    except Exception as e:
        logger.error(f"Error determining relationship type: {str(e)}")
        return 'related_to'
