import streamlit as st
import joblib


model=joblib.load("sentiment-model.pkl")

senti_labels={1:"Positive",0:"Negative"}

st.title("Sentiment Analysis")
user_input=st.text_area("Enter your text here:")
if st.button("Predict"):
    predict_sen=model.predict([user_input])[0]
    print(predict_sen)
    pred_senti_lab=senti_labels[predict_sen]
    st.info(f"Predicted Sentiment: {pred_senti_lab}")
