# FastAPI backend
from fastapi import FastAPI
from pydantic import BaseModel
from src.sentiment_model import SentimentModel
from src.emotion_model import EmotionModel
from src.sarcasm_model import SarcasmModel
from src.preprocess import preprocess_pipeline

app = FastAPI()
sentiment = SentimentModel()
emotion = EmotionModel()
sarcasm = SarcasmModel()

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict(payload: TextInput):
    text = preprocess_pipeline(payload.text)
    senti, senti_prob = sentiment.predict(text)
    emo, emo_prob = emotion.predict(text)
    sarc, sarc_prob = sarcasm.predict(text)
    return {
        "sentiment": senti,
        "sentiment_confidence": senti_prob,
        "emotion": emo,
        "emotion_confidence": emo_prob,
        "sarcasm": sarc,
        "sarcasm_confidence": sarc_prob
    }
