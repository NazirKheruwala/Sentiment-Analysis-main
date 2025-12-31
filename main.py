import tensorflow as tf
import streamlit as st
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

from tensorflow.keras.models import load_model

st.title("Sentiment Analysis System")

# Load the Keras model
loaded_model = load_model('oxymoron.keras') #load the new model here
corpus_file = "corpus.txt"
with open(corpus_file, 'r', encoding='utf-8') as file:
    corpus = [line.strip() for line in file]

voc_size = 5000
ps = PorterStemmer()
tokenizer = Tokenizer(num_words=voc_size)
tokenizer.fit_on_texts(corpus)

comment = st.text_input("Enter your comment: ", "This is the default value for spam detection")
st.write("You entered: ", comment)

comment = re.sub('[^a-zA-Z]', ' ', comment)
comment = comment.lower()
comment = comment.split()
comment = [ps.stem(word) for word in comment if not word in stopwords.words('english')]
comment = ' '.join(comment)

# Convert the comment to a sequence of integers
comment_sequence = tokenizer.texts_to_sequences([comment])
padded_input = pad_sequences(comment_sequence, padding='pre', maxlen=20)

# Use the trained model to predict if the comment is spam or not
prediction = loaded_model.predict(padded_input)


if st.button('Predict'):
    if prediction < 0.4955:
        result_text = "<div style='text-align: center; color: green; font-size: 20px;'>This looks like a positive sentiment.</div>"
    else:
        result_text = "<div style='text-align: center; color: red; font-size: 20px;'>This looks like a negative sentiment.</div>"

    st.markdown(result_text, unsafe_allow_html=True)