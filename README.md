# toxic-comment-classifier
Detecting toxic comments using LSTM in Python
# Toxic Comment Classifier using LSTM

This project is a web-based application that classifies user comments as **toxic** or **non-toxic** using **Natural Language Processing (NLP)** and **LSTM (Long Short-Term Memory)** neural networks.

The model has been pre-trained and integrated into a simple web interface using Streamlit. Users can enter a comment into the app and instantly get a prediction on whether the comment is toxic or not.

---

## 🚀 Features

- Real-time text classification using a pre-trained LSTM model
- Web interface built with Streamlit
- Simple and fast inference without retraining
- Ideal for detecting harmful or abusive comments

---

## 🛠 Technologies & Libraries Used

- Python
- TensorFlow
- Pandas
- Streamlit
- NLTK

---

## 📦 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/toxic-comment-classifier.git
   cd toxic-comment-classifier
## Run the app
   streamlit run app.py
   -A web interface will open in your browser. Enter a comment and click the "Classify" button to get the result!
   
✨ Future Improvements
   -Support for multi-label classification (e.g., insult, threat, etc.)
   -Better handling of slang or abbreviations
   -Deploy the app online (e.g., on Heroku or Streamlit Cloud)
