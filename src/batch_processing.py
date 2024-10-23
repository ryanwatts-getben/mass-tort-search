import torch
import logging
from preprocessing import preprocess_data
from entity_extraction import extract_entities
from relationship_extraction import extract_relationships
from vectorization import vectorize_dataset, initialize_vectorization_model
from utils import prepare_metadata
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from datasets import concatenate_datasets

logger = logging.getLogger(__name__)

def process_batch_parallel(batch, device, tokenizer, ner_model, vectorization_model):
    """Process a batch using dataset operations for efficient GPU utilization."""
    try:
        # Convert batch to dataset if necessary
        if isinstance(batch, dict):
            dataset = Dataset.from_dict(batch)
        elif isinstance(batch, pd.DataFrame):
            dataset = Dataset.from_pandas(batch)
        else:
            dataset = batch
            
        logger.debug(f"Processing batch of size: {len(dataset)}")
        
        # Process through pipeline stages
        try:
            # Preprocessing - ensure cleaned_text is created
            processed_dataset = preprocess_data(dataset, tokenizer)
            logger.debug("Preprocessing completed")
            
            # Verify cleaned_text exists
            if 'cleaned_text' not in processed_dataset.column_names:
                raise ValueError("Preprocessing did not generate cleaned_text field")
            
            # Entity Extraction
            processed_dataset = extract_entities(processed_dataset)
            logger.debug("Entity extraction completed")
            
            # Relationship Extraction
            processed_dataset = extract_relationships(processed_dataset)
            logger.debug("Relationship extraction completed")
            
            # Vectorization
            processed_dataset = vectorize_dataset(processed_dataset, vectorization_model)
            logger.debug("Vectorization completed")
            
            # Verify vector field exists
            if 'vector' not in processed_dataset.column_names:
                raise ValueError("Vectorization did not generate vector field")
            
            # Metadata Preparation - preserve vector field
            metadata_dataset = processed_dataset.map(
                prepare_metadata,
                batched=True,
                batch_size=32,
                desc="Preparing metadata",
                remove_columns=[]  # Don't remove any columns
            )
            
            # Ensure all required fields are present
            required_fields = ['vector', 'cleaned_text', 'case_id', 'document_id', 'page_number', 'metadata']
            missing_fields = [field for field in required_fields if field not in metadata_dataset.column_names]
            if missing_fields:
                raise ValueError(f"Missing required fields after processing: {missing_fields}")
            
            logger.info(f"Successfully processed batch of {len(metadata_dataset)} documents")
            return metadata_dataset
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return Dataset.from_dict({})

def process_batches_with_progress(batches, device, tokenizer, ner_model, vectorization_model):
    """Process multiple batches with progress tracking."""
    processed_datasets = []
    
    with tqdm(total=len(batches), desc="Processing batches") as pbar:
        for batch in batches:
            try:
                processed_batch = process_batch_parallel(
                    batch, 
                    device, 
                    tokenizer, 
                    ner_model, 
                    vectorization_model
                )
                if len(processed_batch) > 0:
                    # Verify vector field exists in processed batch
                    if 'vector' not in processed_batch.column_names:
                        logger.error("Vector field missing from processed batch")
                        continue
                    processed_datasets.append(processed_batch)
                pbar.update(1)
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
    
    if not processed_datasets:
        logger.warning("No batches were processed successfully")
        return Dataset.from_dict({})
    
    # Combine datasets and verify vector field is preserved
    combined_dataset = concatenate_datasets(processed_datasets)
    if 'vector' not in combined_dataset.column_names:
        logger.error("Vector field lost during dataset combination")
        raise ValueError("Vector field missing from combined dataset")
        
    return combined_dataset
