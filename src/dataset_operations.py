from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import logging
from typing import List, Dict, Any
import torch

logger = logging.getLogger(__name__)

def create_pipeline_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """Create a dataset ready for pipeline processing."""
    try:
        def tokenize_and_prepare(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            # Tokenize texts
            tokenized = tokenizer(
                examples['cleaned_text'],
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )

            # Combine with original data
            return {
                **tokenized,
                'cleaned_text': examples['cleaned_text'],
                'case_id': examples['case_id'],
                'document_id': examples['document_id'],
                'page_number': examples['page_number'],
                'Key': examples['Key']
            }

        # Apply preprocessing to entire dataset
        processed_dataset = dataset.map(
            tokenize_and_prepare,
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
            desc="Preparing dataset for pipeline"
        )

        return processed_dataset

    except Exception as e:
        logger.error(f"Error preparing pipeline dataset: {str(e)}")
        raise

def create_data_loader(dataset: Dataset, batch_size: int = 32) -> DataLoader:
    """Create DataLoader for batch processing."""
    try:
        # Set format for PyTorch
        dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'case_id', 'document_id', 'page_number', 'Key']
        )

        # Create DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return data_loader

    except Exception as e:
        logger.error(f"Error creating data loader: {str(e)}")
        raise

def process_pipeline_batch(batch: Dict[str, torch.Tensor], pipeline) -> Dict[str, Any]:
    """Process a batch through a pipeline efficiently."""
    try:
        # Move batch to correct device
        device = next(pipeline.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Run pipeline
        outputs = pipeline(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_dict=True
        )

        return {
            'outputs': outputs,
            'case_id': batch['case_id'],
            'document_id': batch['document_id'],
            'page_number': batch['page_number'],
            'Key': batch['Key']
        }

    except Exception as e:
        logger.error(f"Error processing pipeline batch: {str(e)}")
        raise

def combine_pipeline_outputs(processed_batches: List[Dataset]) -> Dataset:
    """Combine processed batches into a single Dataset."""
    try:
        # Concatenate Datasets
        combined_dataset = concatenate_datasets(processed_batches)
        return combined_dataset

    except Exception as e:
        logger.error(f"Error combining pipeline outputs: {str(e)}")
        raise
