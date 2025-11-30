import pickle
import os
import re

class SarcasmModel:
    def __init__(self):

        # Path to local sarcasm model
        model_path = os.path.join(os.path.dirname(__file__), "sarcasm_classifier.pkl")
        vectorizer_path = os.path.join(os.path.dirname(__file__), "sarcasm_vectorizer.pkl")

        # Load local ML model & vectorizer
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def clean(self, text):
        text = re.sub(r"[^a-zA-Z ]", "", text)
        return text.lower().strip()

    def predict(self, text):
        cleaned = self.clean(text)
        x = self.vectorizer.transform([cleaned])
        pred = self.model.predict(x)[0]
        proba = max(self.model.predict_proba(x)[0])
        
        label = "Sarcastic" if pred == 1 else "Not Sarcastic"
        return label, float(proba)
