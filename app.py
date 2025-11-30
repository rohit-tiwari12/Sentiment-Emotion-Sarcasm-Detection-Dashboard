import streamlit as st
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict/"

st.title("Sentiment Analysis Dashboard")
st.write("Enter text below and click **Analyze** to get predictions.")

# Input text box
text = st.text_area("Enter your text here:", height=150)

# Analyze button
if st.button("Analyze"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        try:
            # Send request to backend API
            response = requests.post(API_URL, json={"text": text})

            if response.status_code == 200:
                result = response.json()

                st.subheader("Prediction Results")
                st.write(f"**Sentiment:** {result.get('sentiment')}")
                st.write(f"**Emotion:** {result.get('emotion')}")
                st.write(f"**Sarcasm:** {result.get('sarcasm')}")

                # Wordcloud visualization
                st.subheader("Word Cloud")
                wc = WordCloud(width=500, height=300, background_color="white").generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

            else:
                st.error(f"❌ API Error: {response.text}")

        except Exception as e:
            st.error(f"🔥 Failed to connect to API: {e}")

