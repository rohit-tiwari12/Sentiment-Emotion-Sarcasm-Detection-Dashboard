# emotion_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmotionModel:
    def __init__(self):
        self.model_name = "bhadresh-savani/bert-base-go-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.labels = ["admiration","amusement","anger","annoyance","approval","caring","confusion",
                       "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
                       "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
                       "pride","realization","relief","remorse","sadness","surprise","neutral"]

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        top = torch.argmax(probs).item()
        return self.labels[top], float(probs[0][top])
