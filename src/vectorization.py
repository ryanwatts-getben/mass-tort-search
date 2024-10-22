from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained ClinicalBERT model
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

def vectorize_documents(df):
    vectors = []
    metadata_list = []
    
    for _, row in df.iterrows():
        # Get all entity types separately
        entities = row['entities']
        
        # Combine text and all entities for embedding
        combined_text = f"{row['cleaned_text']} {' '.join([' '.join(entities[key]) for key in entities])}"
        relationships_text = ' '.join([f"{rel[0]} {rel[1]} {rel[2]}" for rel in row['relationships']['custom']])
        full_text = f"{combined_text} {relationships_text}"
        
        # Tokenize and get embeddings
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embedding
        vector = outputs.last_hidden_state[0][0].numpy()
        
        # Create metadata with separate entity types
        metadata = {
            'text': row['cleaned_text'][:1000],
            'case_id': row['case_id'],
            'document_id': row['document_id'],
            'icd10_codes': entities['icd10_codes'],
            'symptoms': entities['symptoms'],
            'lab_results': entities['lab_results'],
            'other_conditions': entities['other_conditions'],
            'diagnostic_procedures': entities['diagnostic_procedures'],
            'treatment_options': entities['treatment_options'],
            'complications': entities['complications']
        }
        
        vectors.append(vector)
        metadata_list.append(metadata)
    
    df['vector'] = vectors
    df['metadata'] = metadata_list
    return df