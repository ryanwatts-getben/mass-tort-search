import spacy

nlp = spacy.load("en_core_web_sm")

def extract_relationships(df):
    relationships = []
    for _, row in df.iterrows():
        doc = nlp(row['cleaned_text'])
        
        # Extract syntactic dependencies
        deps = [(token.text, token.dep_, token.head.text) for token in doc]
        
        # Apply custom rules (simplified example)
        custom_rels = extract_custom_relationships(row['entities'], row['cleaned_text'])
        
        relationships.append({'syntactic': deps, 'custom': custom_rels})
    
    df['relationships'] = relationships
    return df

def extract_custom_relationships(entities, text):
    """Extract custom relationships based on specific patterns."""
    relationships = []
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            # Simple co-occurrence in the same sentence
            if entity1 != entity2:
                relationships.append((entity1, 'related_to', entity2))
    return relationships
