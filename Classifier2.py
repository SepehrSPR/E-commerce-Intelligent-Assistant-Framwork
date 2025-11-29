from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FaBERTSubClassifier:
    def __init__(self, model_path: str = ...):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.label_map = {
            0: 'ساده',
            1: 'پارامتری',
            2: 'غیر پارامتری'
        }

    def classify(self, query: str) -> str:
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_label = outputs.logits.argmax(-1).item()
        return self.label_map[predicted_label]
