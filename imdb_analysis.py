import streamlit as st
import pickle  # Save model 
from sklearn.feature_extraction.text import TfidfVectorizer


# Load pre-trained model and vectorizer
model = pickle.load(open('review_model_log.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# App title
st.title("IMDb Movie Review Sentiment Analysis")

st.image("./img/imdb_img3.jpg", caption="IMDB Review", use_column_width=True)
# Input text box
user_input = st.text_area("Enter a movie review:")

if st.button("Analyze"):
    if user_input:
        processed_input = vectorizer.transform([user_input])
        prediction = model.predict(processed_input)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to analyze.")
