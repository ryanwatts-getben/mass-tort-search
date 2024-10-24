from transformers import AutoTokenizer, AutoModel
import torch
from utils import load_config
import logging
from datasets import Dataset
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Load configuration
config = load_config('config/config.yaml')
model_config = config.get('models', {})
bert_model_name = model_config.get('bert', "medicalai/ClinicalBERT")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

def initialize_vectorization_model(device):
    """Initialize a separate model for vectorization."""
    try:
        model = AutoModel.from_pretrained(bert_model_name)
        model.to(device)
        model.eval()
        logger.info(f"Initialized vectorization model {bert_model_name} on {device}")
        return model
    except Exception as e:
        logger.error(f"Error initializing vectorization model: {str(e)}")
        raise

def vectorize_dataset(dataset: Dataset, model) -> Dataset:
    """Vectorize documents in a dataset using efficient batching."""
    try:
        device = next(model.parameters()).device

        def process_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            try:
                # Ensure we're using cleaned_text
                if 'cleaned_text' not in examples:
                    logger.error("Missing cleaned_text in examples")
                    raise ValueError("Dataset must contain cleaned_text field")

                # Combine text and entities
                combined_texts = []
                for idx, (text, entities) in enumerate(zip(examples['cleaned_text'], examples.get('entities', [{}] * len(examples['cleaned_text'])))):
                    try:
                        # Ensure text is a string
                        text = str(text) if text is not None else ""
                        
                        # Handle entities
                        if isinstance(entities, dict):
                            entity_strings = []
                            for ent_list in entities.values():
                                if isinstance(ent_list, list):
                                    entity_strings.extend(str(e) for e in ent_list)
                            entity_text = ' '.join(entity_strings)
                        else:
                            entity_text = ''
                        
                        combined_text = f"{text} {entity_text}".strip()
                        combined_texts.append(combined_text)

                        # Log the text being vectorized
                        logger.debug(f"Index {idx}: Combined text for vectorization: {combined_text[:100]}")

                    except Exception as e:
                        logger.error(f"Error combining text and entities at index {idx}: {str(e)}")
                        combined_texts.append("")

                # Tokenize with error handling
                try:
                    inputs = tokenizer(
                        combined_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(device)
                except Exception as e:
                    logger.error(f"Error in tokenization: {str(e)}")
                    raise

                # Get embeddings
                try:
                    with torch.no_grad():
                        outputs = model(**inputs)
                        vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        logger.debug(f"Generated vectors with shape: {vectors.shape}")
                except Exception as e:
                    logger.error(f"Error in model inference: {str(e)}")
                    raise

                # Validate vectors
                if vectors is None or len(vectors) == 0:
                    logger.error("No vectors generated in current batch.")
                    raise ValueError("Vectors are empty.")

                return {
                    **examples,
                    'vector': vectors.tolist()
                }
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                # Return empty vectors as fallback
                zero_vectors = [[0.0] * 768 for _ in range(len(examples['cleaned_text']))]
                return {
                    **examples,
                    'vector': zero_vectors
                }

        # Process dataset in batches
        vectorized_dataset = dataset.map(
            process_batch,
            batched=True,
            batch_size=32,
            desc="Vectorizing documents"
        )

        return vectorized_dataset

    except Exception as e:
        logger.error(f"Error in vectorization: {str(e)}")
        raise
