import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

import plotly.express as px
import plotly.figure_factory as ff
from classifier_curves import eval_curves



st.set_page_config(
    page_title="Customer Churn",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")


#---------------------------------#
# Sidebar - Specify parameter settings
with st.sidebar.header('1. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 
                        value=0.20, min_value=0.10, max_value=0.50, step=0.05)

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

def process_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.TotalCharges = df.TotalCharges.fillna(0)

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    string_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
        
    df.churn = (df.churn == 'yes').astype(int)
    return df

def train(df_train, y_train, C=1.0): 
    ## Modify format from training dataframe to dictionary
    train_dict = df_train[categorical + numerical].to_dict(orient='records')

    ## Using DictVectorizer to create one-hot encoding for categorical variables
    dv = DictVectorizer(sparse=False)

    ## Fitting DictVectorizer on training dictionary
    dv.fit(train_dict)
    X_train = dv.transform(train_dict)

    ## Fitting Logistic Regression with solver 'liblinear' to training set
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=1)
    model.fit(X_train, y_train)

    return dv, model

def predict(df_val, dv, model):
    ## Modify format from validation dataframe to dictionary
    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    ## Predict model on validation set, returning probabilities
    y_pred_prob = model.predict_proba(X_val)

    return y_pred_prob

def data_splitting(df):
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)

    y_train = df_train.churn.values
    y_val = df_val.churn.values
    y_test = df_test.churn.values

    del df_train['churn']
    del df_val['churn']
    del df_test['churn']

    return df_train, y_train, df_val, y_val, df_test, y_test

def pickle_model(model):
    """Pickle the model inside bytes. In our case, it is the "same" as 
    storing a file, but in RAM.
    """
    f = io.BytesIO()
    pickle.dump(model, f)
    return f

def save_model(model):
    # save the model to disk
    filename = 'finalized_model.sav'
    joblib.dump(model, filename)

def save_dictvectorizer(dv):
    # save the dictionary vectorizer to disk
    filename = 'dictvec.pkl'
    joblib.dump(dv, filename)

def main():
    st.sidebar.header('User Input Features')
    sns.set_style('darkgrid')

    df = process_data()

    df_train, y_train, df_val, y_val, df_test, y_test = data_splitting(df)

    dv, model = train(df_train, y_train, C=1.0)
    y_pred_prob = predict(df_val, dv, model)

    fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob[:, 1])
    auc = roc_auc_score(y_val, y_pred_prob[:, 1]).round(3)

    st.title('Classification Performances on Customer Churn')

    st.write('Depicting Generalization Performances.')

    

    # fig, fig2, fig3 = eval_curves(fpr, tpr, auc, y_test, y_prob, conf_matrix_array)
    fig, fig2 = eval_curves(fpr, tpr, auc, y_val, y_pred_prob)     

    with st.expander("ROC-AUC and Precision-Recall"):
        col3, col4= st.columns(2)
        with col3 :
            st.subheader("ROC AUC")
            st.plotly_chart(fig)
        
        with col4:
            st.subheader("Precision Recall")
            st.plotly_chart(fig2)

    # with st.expander("Confusion Matrix"):
    #     st.subheader("Confusion Matrix")

    #     st.plotly_chart(fig3)
    data = pickle_model(model)
    st.download_button("Download .pkl file", data=data, file_name="model_C.pkl")

    save_dictvectorizer(dv,)


if __name__ == '__main__':
    main()