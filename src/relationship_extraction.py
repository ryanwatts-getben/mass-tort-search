import spacy
import logging

logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_relationships(df):
    """Extract relationships from the DataFrame."""
    relationships = []
    
    for _, row in df.iterrows():
        doc = nlp(row['cleaned_text'])
        
        # Extract syntactic dependencies
        deps = [(token.text, token.dep_, token.head.text) for token in doc]
        
        # Extract custom relationships
        custom_rels = extract_custom_relationships(row['entities'])
        
        relationships.append({'syntactic': deps, 'custom': custom_rels})
    
    # Use .loc for assignment
    df.loc[:, 'relationships'] = relationships
    return df

def extract_custom_relationships(entities):
    """Extract custom relationships between entities."""
    relationships = []
    
    # Get all entity values in a list
    all_entities = []
    for category, items in entities.items():
        for item in items:
            all_entities.append((category, item))
    
    # Create relationships between entities
    for i, (cat1, ent1) in enumerate(all_entities):
        for cat2, ent2 in all_entities[i+1:]:
            if ent1 != ent2:
                relationship_type = determine_relationship_type(cat1, cat2)
                relationships.append((ent1, relationship_type, ent2))
    
    return relationships

def determine_relationship_type(category1, category2):
    """Determine the type of relationship between two entity categories."""
    if category1 == 'symptoms' and category2 == 'diagnostic_procedures':
        return 'diagnosed_by'
    elif category1 == 'symptoms' and category2 == 'treatment_options':
        return 'treated_by'
    elif category1 == 'icd10_codes' and category2 == 'complications':
        return 'may_lead_to'
    elif category1 == 'lab_results' and category2 == 'treatment_options':
        return 'indicates'
    else:
        return 'related_to'
