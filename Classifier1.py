from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class FaBERTClassifier:
    def __init__(self, model_path: str = r"F:\Arshad\Payanname\Classifier1\withBertFamily\FaBert\fabert-sentence-classifier"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.label_map = {
            0: 'پشتیبانی سایت',
            1: 'راهنمایی در مورد کالا',
            2: 'جستجو',
            3: 'مقایسه'
        }

    def classify(self, query: str) -> str:
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_label = outputs.logits.argmax(-1).item()
        return self.label_map[predicted_label]
