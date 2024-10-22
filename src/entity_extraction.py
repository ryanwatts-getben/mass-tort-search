from transformers import pipeline
from yake import KeywordExtractor
import spacy
import logging

logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize zero-shot classification pipeline
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialize YAKE keyword extractor
keyword_extractor = KeywordExtractor()

def extract_entities(df):
    entities = []
    for _, row in df.iterrows():
        # Apply spaCy NER
        doc = nlp(row['cleaned_text'])
        spacy_entities = [ent.text for ent in doc.ents]
        
        # Apply zero-shot classification for custom entity types
        custom_entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MEDICAL_CONDITION", "MEDICATION"]
        zero_shot_results = zero_shot_classifier(row['cleaned_text'], candidate_labels=custom_entity_types, multi_label=True)
        
        # Extract top entities from zero-shot classification
        zero_shot_entities = [label for label, score in zip(zero_shot_results['labels'], zero_shot_results['scores']) if score > 0.5]
        
        # Extract keywords using YAKE
        keywords = keyword_extractor.extract_keywords(row['cleaned_text'])
        yake_entities = [kw[0] for kw in keywords[:5]]  # Get top 5 keywords
        
        # Combine results
        combined_entities = list(set(spacy_entities + zero_shot_entities + yake_entities))
        entities.append(combined_entities)
    
    df['entities'] = entities
    return df
