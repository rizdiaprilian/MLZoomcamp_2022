import streamlit as st

st.set_page_config(
    page_title="Heart Failure",
    layout="wide",
    initial_sidebar_state="expanded",
)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

import psycopg2
from psycopg2 import Error

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score

import plotly.express as px
import plotly.figure_factory as ff
# from classification_engine import classification
from classifier_curves import eval_curves
from engine import classification, make_prediction
from engine import LearnerName, LearnerInfo, Logistic_Regression



#---------------------------------#
# Sidebar - Specify parameter settings
with st.sidebar.header('1. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 
                        value=0.20, min_value=0.10, max_value=0.50, step=0.05)

def building_models(df):
    X = df.loc[:,:"time"]
    y = df.loc[:,["DEATH_EVENT"]]

    y = np.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42, stratify=y)

    cat_columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    num_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                     'platelets', 'serum_creatinine', 'serum_sodium', 'time']

    ## Transformer Pipeline
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', dtype=np.int))
    ])

    ## Column Transformer
    preprocessor = ColumnTransformer([
        ('numeric', num_transformer, num_columns),
        ('categoric', cat_transformer, cat_columns),
    ])

    ## Apply Column Transformer
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    ## Label Encoding
    y_transformer = LabelEncoder()
    y_train = y_transformer.fit_transform(y_train)
    y_test = y_transformer.transform(y_test)

    return X_train, y_train, X_test, y_test

def main():
    st.sidebar.header('User Input Features')
    sns.set_style('darkgrid')

    df = pd.read_csv('normalized_data.csv')

#     param_dic = {
#     "user":"postgres",
#     "password":"xxxg00w0",
#     "host":"127.0.0.1",
#     "port":"5432",
#     "database":"heart_failure"
# }
#     conn = connect(param_dic)

#     # Connect to the database
#     column_names = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
# 				"high_blood_pressure","platelets", "serum_creatinine", "serum_sodium",
# 				"sex", "smoking", "time", "DEATH_EVENT"]

#     # Execute the "SELECT *" query
#     df = postgresql_to_dataframe(conn, """SELECT * FROM heart_failure;""", column_names)
    
    # df = pd.read_csv('normalized_data.csv', sep=',')
    cat_cols = list(filter(lambda x: x if len(df[x].unique()) <= 3 else None, df.columns))

    df[cat_cols] = df[cat_cols].astype('category')

    
    X_train, y_train, X_test, y_test = building_models(df)

    classifier = st.sidebar.selectbox("Classifier", ("Random Forest", "Support Vector Machine", "Logistic Regression"))
    
    y_predict, y_prob = classification(classifier, X_train, y_train, X_test, y_test)
    precision, recall, fscore, accuracy, tnr, npv, conf_matrix_array = make_prediction(y_test, y_predict)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
    auc = roc_auc_score(y_test, y_prob[:,1]).round(3)


    st.title('Classification Performances on Dataset Heart Failure')


    st.write('The model performance of the heart failure is depicted.')

    st.write("Accuracy: ", accuracy.round(3))
    st.write("Precision: ", precision.round(3))
    st.write("Recall: ", recall.round(3))
    st.write("F1-Score: ", fscore.round(3))
    st.write("Specificity: ", tnr.round(3))
    st.write("Negative Prediction Value: ", npv.round(3))

    fig, fig2, fig3 = eval_curves(fpr, tpr, auc, y_test, y_prob, conf_matrix_array)    

    with st.expander("ROC-AUC and Precision-Recall"):
        col3, col4= st.columns(2)
        with col3 :
            st.subheader("ROC AUC")
            st.plotly_chart(fig)
        
        with col4:
            st.subheader("Precision Recall")
            st.plotly_chart(fig2)

    with st.expander("Confusion Matrix"):
        st.subheader("Confusion Matrix")

        st.plotly_chart(fig3)


if __name__ == '__main__':
    main()