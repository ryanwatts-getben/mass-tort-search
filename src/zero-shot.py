import os
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device

def load_model(device):
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)

def process_text(classifier, text, candidate_labels):
    result = classifier(text, candidate_labels)
    return result

def main():
    device = setup_gpu()
    classifier = load_model(device)

    text = "The patient has a history of hypertension and is currently taking lisinopril and amlodipine."
    candidate_labels = ["DISEASE", "MEDICATION"]

    result = process_text(classifier, text, candidate_labels)

    for label, score in zip(result['labels'], result['scores']):
        print(f"Result: {result}, Label: {label}, Score: {score:.4f}")

if __name__ == "__main__":
    main()