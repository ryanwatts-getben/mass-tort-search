from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BioBERT model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

def vectorize_documents(df):
    vectors = []
    for _, row in df.iterrows():
        # Combine text, entities, and relationships
        relationships_text = ' '.join([f"{rel[0]} {rel[1]} {rel[2]}" for rel in row['relationships']['custom']])
        combined_text = f"{row['cleaned_text']} {' '.join(row['entities'])} {relationships_text}"
        
        # Tokenize and get embeddings
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embedding
        vector = outputs.last_hidden_state[0][0].numpy()
        vectors.append(vector)
    df['vector'] = vectors
    return df
