import streamlit as st
import requests
import pandas as pd
import altair as alt

# ============================
#   UI HEADER
# ============================

st.set_page_config(
    page_title="AI Sentiment–Emotion–Sarcasm Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Real-Time Sentiment • Emotion • Sarcasm Detection Dashboard")

st.markdown("""
This dashboard analyzes text using your machine-learning models:

- **Sentiment Analysis** → Positive / Neutral / Negative  
- **Emotion Detection** → Joy, Anger, Sadness, Fear, Surprise, Love  
- **Sarcasm Detection** → Detects sarcastic or misleading tone  
- **Confidence Scores** → Probability-based insights  
- **Real-Time Charts** → Trend analytics  
- **Live Monitoring (News)** → Fetch latest Google News titles  

Use the input box or fetch live data to visualize AI-powered insights instantly.
""")
st.divider()

# ============================
#   API URLs
# ============================

API_PREDICT = "http://127.0.0.1:8000/predict/"
API_NEWS = "http://127.0.0.1:8000/news/"

# ============================
#   USER TEXT INPUT SECTION
# ============================

st.subheader("📝 Manual Text Analysis")

text = st.text_area("Enter your text here:", height=120)

if st.button("Analyze Text"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        try:
            response = requests.post(API_PREDICT, json={"text": text}).json()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Sentiment", response["sentiment"],
                          f"{response['sentiment_confidence']:.2f}")

            with col2:
                st.metric("Emotion", response["emotion"],
                          f"{response['emotion_confidence']:.2f}")

            with col3:
                st.metric("Sarcasm", response["sarcasm"],
                          f"{response['sarcasm_confidence']:.2f}")

        except Exception as e:
            st.error(f"❌ API Error: {e}")

st.divider()

# ============================
#   REAL-TIME NEWS SECTION
# ============================

st.subheader("📰 Real-Time Google News Monitoring")

query = st.text_input("Enter news topic or keyword (example: India, technology, AI):", "india")
limit = st.slider("Number of News Articles", 5, 50, 20)

def get_news_data():
    try:
        response = requests.get(API_NEWS, params={"query": query, "limit": limit})
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"❌ Error fetching news: {e}")
        return pd.DataFrame()

if st.button("Fetch Live News"):
    df = get_news_data()

    if df.empty:
        st.warning("No news articles found or backend is not running.")
    else:
        st.success(f"Fetched {len(df)} news articles")

        # SHOW TABLE
        st.dataframe(df)

        # ============================
        #   SENTIMENT CHART
        # ============================

        if "sentiment" in df.columns:
            st.subheader("📈 Sentiment Distribution")

            sentiment_counts = df["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment", "count"]

            chart = alt.Chart(sentiment_counts).mark_bar().encode(
                x="sentiment",
                y="count",
                color="sentiment"
            )
            st.altair_chart(chart, use_container_width=True)

        # ============================
        #   EMOTION CHART
        # ============================

        if "emotion" in df.columns:
            st.subheader("🎭 Emotion Distribution")

            emotion_counts = df["emotion"].value_counts().reset_index()
            emotion_counts.columns = ["emotion", "count"]

            chart2 = alt.Chart(emotion_counts).mark_bar().encode(
                x="emotion",
                y="count",
                color="emotion"
            )
            st.altair_chart(chart2, use_container_width=True)

        # ============================
        #   SARCASM CHART
        # ============================

        if "sarcasm" in df.columns:
            st.subheader("🤨 Sarcasm Levels")

            sarcasm_counts = df["sarcasm"].value_counts().reset_index()
            sarcasm_counts.columns = ["sarcasm", "count"]

            chart3 = alt.Chart(sarcasm_counts).mark_bar().encode(
                x="sarcasm",
                y="count",
                color="sarcasm"
            )
            st.altair_chart(chart3, use_container_width=True)
