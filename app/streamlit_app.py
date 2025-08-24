import streamlit as st
import pickle
import re
import string

# -----------------------------
# Preprocessing function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

# -----------------------------
# Load model and vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    with open("../models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("../models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis of Social Media Posts")
st.write("Enter a tweet or social media post and the model will predict its sentiment.")

st.markdown("<br>", unsafe_allow_html=True)  # Add spacing

tweet_input = st.text_area("Enter Text:", height=150)

st.markdown("<br>", unsafe_allow_html=True)  # Add spacing

if st.button("Predict Sentiment"):
    if tweet_input.strip() == "":
        st.warning("Please enter some text to predict!")
    else:
        cleaned = clean_text(tweet_input)
        features = vectorizer.transform([cleaned])
        pred = model.predict(features)[0]

        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(pred, "Unknown")

        # Set colors and icons
        if sentiment.lower() == "positive":
            bg_color = "#e6ffed"
            border_color = "green"
            text_color = "green"
            icon = "😊"
        elif sentiment.lower() == "negative":
            bg_color = "#ffe6e6"
            border_color = "red"
            text_color = "red"
            icon = "😠"
        else:  # Neutral
            bg_color = "#e6f0ff"
            border_color = "blue"
            text_color = "blue"
            icon = "😐"

        # Modern card-style display
        st.markdown(
            f"""
            <div style='
                background-color: {bg_color};
                padding: 25px;
                border-radius: 15px;
                border: 2px solid {border_color};
                box-shadow: 4px 4px 12px rgba(0,0,0,0.1);
                max-width: 600px;
            '>
                <p style='font-size:20px; color:black; margin:0;'><strong>Predicted Sentiment:</strong></p>
                <p style='font-size:28px; color:{text_color}; margin:5px 0 0 0;'>{icon} {sentiment}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing after card
