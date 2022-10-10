import streamlit as st
import time
import numpy as np
from main_app import process_data, data_splitting, save_model
from engine import dict_vect_transform, classification, make_prediction, metric_calculation

st.set_page_config(page_title="Training Demo", page_icon="📈")

st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

df = process_data()
df_train, y_train, df_val, y_val, df_test, y_test = data_splitting(df)

classifier = st.sidebar.selectbox("Classifier", (
        "Random Forest",
        "Support Vector Machine", 
        "Logistic Regression",
        "Gradient Boosting"))

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

dv, df_train = dict_vect_transform(df_train, categorical, numerical)

algo_learning = classification(
    classifier, df_train, y_train
    )

y_predict, y_prob = make_prediction(dv, algo_learning, df_val, categorical, numerical)

precision, recall, fscore, accuracy, tnr, npv, conf_matrix_array = metric_calculation(y_val, y_predict)

st.write("Accuracy: ", accuracy.round(3))
st.write("Precision: ", precision.round(3))
st.write("Recall: ", recall.round(3))
st.write("F1-Score: ", fscore.round(3))
st.write("Specificity: ", tnr.round(3))
st.write("Negative Prediction Value: ", npv.round(3))