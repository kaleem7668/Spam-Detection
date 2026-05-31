import streamlit as st
import pandas as pd
import  joblib


model = joblib.load("NB_spam.pkl")
vectorizer = joblib.load("vectorizer_spam.pkl")


st.set_page_config(page_title="Spam Detector")
st.title("📩 Spam Detection App")

st.write("Enter a message to check if it's Spam or Not Spam")

input_sms = st.text_area("Message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        data = vectorizer.transform([input_sms])
        result = model.predict(data)[0]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

