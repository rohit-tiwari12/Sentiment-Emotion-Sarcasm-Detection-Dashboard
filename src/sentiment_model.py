# sentiment_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentModel:
    def __init__(self):
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item() + 1
        sentiment_map = {1:"Very Negative",2:"Negative",3:"Neutral",4:"Positive",5:"Very Positive"}
        return sentiment_map[pred], float(probs[0][pred-1])
