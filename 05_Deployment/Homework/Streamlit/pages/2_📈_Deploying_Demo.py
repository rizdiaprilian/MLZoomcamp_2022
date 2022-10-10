import pickle
import streamlit as st
import time
import numpy as np
import shutil
import joblib

st.set_page_config(page_title="Deploying Demo", page_icon="📈")

st.markdown("# Deploying Demo")
st.sidebar.header("PlotDeployingting Demo")
st.write(
    """This demo illustrates a combination of deploying with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

def load_models(model_file):
    model = joblib.load(model_file)
    return model

def load_dictvectorizer(dv_file):
    dv = joblib.load(dv_file)
    return dv

uploaded_model_file = st.file_uploader("Choose a file", accept_multiple_files=True)
if uploaded_model_file is not None:
    # Upload model file firstly
    model = load_models(uploaded_model_file[0])
    # Then upload dictvect file
    dv = load_dictvectorizer(uploaded_model_file[1])

# Give new customer for prediction
customer = {
    "customerid": "XXXG-00W0",
    "gender": "male",
    "seniorcitizen": 1,
    "partner": "yes",
    "dependents": "no",
    "tenure": 21,
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "no",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check", 
    "monthlycharges": 30.55,
    "totalcharges": 43.12
}

# Transform input dict readable for prediction
X = dv.transform([customer])

probability = model.predict_proba(X)[0,1]

st.write("Churn Probability: ", probability.round(3))