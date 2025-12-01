# FastAPI backend
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import httpx
import feedparser

from src.sentiment_model import SentimentModel
from src.emotion_model import EmotionModel
from src.sarcasm_model import SarcasmModel
from src.preprocess import preprocess_pipeline

app = FastAPI()

# Load your ML models
sentiment = SentimentModel()
emotion = EmotionModel()
sarcasm = SarcasmModel()


# ============================================================
# 1️⃣ TEXT PREDICTION ENDPOINT (works for manual text)
# ============================================================

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
        "sentiment_confidence": float(senti_prob),
        "emotion": emo,
        "emotion_confidence": float(emo_prob),
        "sarcasm": sarc,
        "sarcasm_confidence": float(sarc_prob),
    }


# ============================================================
# 2️⃣ REAL-TIME GOOGLE NEWS SCRAPER (FREE + VERY STABLE)
# ============================================================

@app.get("/news/")
def news_search(query: str = "india", limit: int = 10):
    try:
        # Google News RSS feed (free & no key)
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

        feed = feedparser.parse(rss_url)

        posts = []

        for entry in feed.entries[:limit]:
            text = entry.title

            clean = preprocess_pipeline(text)

            senti, senti_prob = sentiment.predict(clean)
            emo, emo_prob = emotion.predict(clean)
            sarc, sarc_prob = sarcasm.predict(clean)

            posts.append({
                "text": text,
                "link": entry.link,
                "sentiment": senti,
                "emotion": emo,
                "sarcasm": sarc,
                "sentiment_confidence": float(senti_prob),
                "emotion_confidence": float(emo_prob),
                "sarcasm_confidence": float(sarc_prob)
            })

        return posts

    except Exception as e:
        return {"error": str(e)}
