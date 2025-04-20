import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import re
from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def main():
    st.title("Toxic Comment Classification")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        train_data = pd.read_csv(uploaded_file)
    
        # Make the data lowercase
        train_data["comment_text"] = train_data["comment_text"].str.lower()

        def cleaning(data):
            clean_column = re.sub('<.*?>', ' ', str(data))
            clean_column = re.sub('[^a-zA-Z0-9\.]+',' ', clean_column)       
            tokenized_column = word_tokenize(clean_column)
            return tokenized_column

        train_data["cleaned"] = train_data["comment_text"].apply(cleaning)

        lemmatizer = WordNetLemmatizer()
        def lemmatizing(data):
            my_data = data["cleaned"]
            lemmatized_list = [lemmatizer.lemmatize(word) for word in my_data]
            return (lemmatized_list)
        
        train_data["lemmatized"] = train_data.apply(lemmatizing, axis = 1)

        train_data["comment_text"] = train_data["lemmatized"]
        train = train_data[["comment_text"]]
        train_labels = train_data[["toxic"]]

        # 2. Use train_test_split to split into train/test
        comment_train, comment_test, labels_train, labels_test = train_test_split(train, train_labels, test_size = 0.2, random_state=42)
        labels_train = np.ravel(labels_train)
        labels_test = np.ravel(labels_test)

        # 3. CountVectorizer
        count_vect = CountVectorizer()
        comment_train_counts = count_vect.fit_transform(comment_train.comment_text.astype(str))

        tfidf_transformer = TfidfTransformer()
        comment_train_tfidf = tfidf_transformer.fit_transform(comment_train_counts)

        # 5 Train a classifier
        clf = MultinomialNB().fit(comment_train_tfidf, labels_train)

        comment_test_new_counts = count_vect.transform(comment_test.comment_text.astype(str))
        comment_test_new_tfidf = tfidf_transformer.transform(comment_test_new_counts)

        # 6 Train LSTM Model
        num_words = 20000
        max_len = 150
        emb_size = 128

        tok = Tokenizer(num_words = num_words, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tok.fit_on_texts(list(comment_train.comment_text.astype(str)))

        comment_train2 = tok.texts_to_sequences(comment_train.comment_text.astype(str))
        comment_test2 = tok.texts_to_sequences(comment_test.comment_text.astype(str))

        comment_train2 = sequence.pad_sequences(comment_train2, maxlen = max_len)
        comment_test2 = sequence.pad_sequences(comment_test2, maxlen = max_len)

        inp = Input(shape = (max_len, ))
        layer = Embedding(num_words, emb_size)(inp)
        layer = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout = 0.15))(layer)
        layer = GlobalMaxPool1D()(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(50, activation = 'relu')(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(1, activation = 'sigmoid')(layer)
        model = Model(inputs = inp, outputs = layer)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

        model.fit(comment_train2, labels_train, batch_size = 512, epochs = 1, validation_split = 0.2, validation_data = (comment_test2, labels_test))

        # 6 Prediction:
        prediction_nb = clf.predict(comment_test_new_tfidf)
        prediction_lstm = (model.predict(comment_test2).ravel()>0.5)+0 

        cm_nb = metrics.confusion_matrix(labels_test, prediction_nb)
        cm_lstm = confusion_matrix(labels_test, prediction_lstm)

        st.write("Naive Bayes Confusion Matrix")
        st.write(cm_nb)

        st.write("LSTM Confusion Matrix")
        st.write(cm_lstm)

        st.write("Naive Bayes Accuracy:")
        st.write(np.mean(prediction_nb == labels_test))

        st.write("LSTM Accuracy:")
        st.write(np.mean(prediction_lstm == labels_test))

        # Show classification reports
        st.write("Naive Bayes Classification Report:")
        st.text(metrics.classification_report(labels_test, prediction_nb))

        st.write("LSTM Classification Report:")
        st.text(metrics.classification_report(labels_test, prediction_lstm))

if __name__ == '__main__':
    main()
