import streamlit as st
from keras.models import load_model
from keras.preprocessing import sequence
import pickle
import numpy as np

# Function to load models and tokenizers
def load_models():
    try:
        # Load LSTM model
        lstm_model = load_model('lstm_model_v2.h5')

        # Load Naive Bayes model
        with open('nb_classifier.pkl', 'rb') as f:
            nb_model = pickle.load(f)

        # Load Tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        # Load Count Vectorizer
        with open('count_vect.pkl', 'rb') as f:
            count_vect = pickle.load(f)

        # Load TFIDF Transformer
        with open('tfidf_transformer.pkl', 'rb') as f:
            tfidf_transformer = pickle.load(f)

        return lstm_model, nb_model, tokenizer, count_vect, tfidf_transformer

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# Load the models and tokenizers
lstm_model, nb_model, tokenizer, count_vect, tfidf_transformer = load_models()

# Function to classify text using LSTM model
def classify_text_lstm(input_text):
    if tokenizer is None or lstm_model is None:
        st.error("Models are not loaded properly. Please check the files.")
        return None

    try:
        input_seq = tokenizer.texts_to_sequences([input_text])
        input_pad = sequence.pad_sequences(input_seq, maxlen=150)
        lstm_prediction = lstm_model.predict(input_pad).ravel()[0]
        lstm_result = "Toxic" if lstm_prediction > 0.5 else "Non-toxic"
        return lstm_result
    except Exception as e:
        st.error(f"LSTM error: {str(e)}")
        return None

# Function to classify text using Naive Bayes model
def classify_text_nb(input_text):
    if count_vect is None or nb_model is None or tfidf_transformer is None:
        st.error("Models are not loaded properly. Please check the files.")
        return None

    try:
        input_counts = count_vect.transform([input_text])
        input_tfidf = tfidf_transformer.transform(input_counts)
        nb_prediction = nb_model.predict(input_tfidf)[0]
        nb_result = "Toxic" if nb_prediction == 1 else "Non-toxic"
        return nb_result
    except Exception as e:
        st.error(f"Naive Bayes error: {str(e)}")
        return None

# Streamlit interface
st.title("Toxic Comment Classifier")
st.markdown("Enter the text you want to classify:")
user_input = st.text_area("")

model_selection = st.selectbox("Select the classification model:", ("LSTM", "Naive Bayes"))

if st.button("Classify"):
    if user_input:
        if model_selection == "Naive Bayes":
            result = classify_text_lstm(user_input)
        elif model_selection == "LSTM":
            result = classify_text_nb(user_input)

        if result is not None:
            st.markdown("### Results")
            st.markdown(f"**Prediction:** {result}", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to classify.")
