
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load IMDb word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load trained RNN model
model = load_model("simple_RNN.h5")

# Helper functions
def preprocess_text(text):
    encoded = [word_index.get(word, 2) for word in text.lower().split()]
    padded = pad_sequences([encoded], maxlen=500, padding='pre')
    return padded

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)[0][0]
    sentiment = "positive" if prediction > 0.5 else "negative"
    return sentiment, prediction

# Streamlit UI
st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.write("Enter a movie review and get sentiment prediction (positive/negative).")

user_review = st.text_area("Enter your review:")

if st.button("Predict"):
    if user_review.strip() != "":
        sentiment, score = predict_sentiment(user_review)
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.3f}")
    else:
        st.warning("⚠️ Please enter a review text.")
