import streamlit as st 
import liga
from liga import Graph
from liga import Vertex
import lstm
from lstm import LSTMBaseline

def evaluate_liga(input_text):
    predicted_lang = liga.predict(input_text)
    return '''The input text ***%s*** is in ***%s***''' % (input_text,predicted_lang)

def evaluate_lstm(input_text):
    predicted_lang = lstm.test_one_sample(input_text)
    return '''The input text ***%s*** is in ***%s***''' % (input_text,predicted_lang)

st.title("""
Language Identification
""")

st.write(
    """
    Enter any text to detect a language!\n 
    *Supported Languages: English, German, French, Dutch, Spanish, Italian*
    """
)

model_choice = st.radio("Select the type of model you would like: ",("LIGA","LSTM"))

sentence = st.text_input('Input your sentence here:') 

if sentence:
    if model_choice == "LIGA":
        st.write(evaluate_liga(sentence))
    elif model_choice == 'LSTM':
        st.write(evaluate_lstm(sentence))