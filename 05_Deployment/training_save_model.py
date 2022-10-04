#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import sys

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
            'phoneservice', 'multiplelines', 'internetservice',
            'onlinesecurity', 'onlinebackup', 'deviceprotection',
            'techsupport', 'streamingtv', 'streamingmovies',
            'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

def read_dataframe(file: str):
    """
    Read csv file and load it as a Pandas DataFrame
    """
    df = pd.read_csv(file)
    return df

def preprocessing(df: pd.DataFrame):
    """
    Modify format for consistency and fill missing values
    """
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.TotalCharges = df.TotalCharges.fillna(0)

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    string_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
        
    df.churn = (df.churn == 'yes').astype(int)
    return df

def split_dataframe(df: pd.DataFrame):
    """
    Split dataframe into three sets: train, validation, test
    Produce target arrays for three sets
    """
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)

    y_train = df_train.churn.values
    y_val = df_val.churn.values
    y_test = df_test.churn.values

    del df_train['churn']
    del df_val['churn']
    del df_test['churn']

    return df_train_full, df_test, y_train, y_test


## Function train
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

# Function predict
def predict(df_val, dv, model):
    ## Modify format from validation dataframe to dictionary
    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    ## Predict model on validation set, returning probabilities
    y_pred = model.predict_proba(X_val)[:, 1]

    return y_pred


def train_running(df_train_full, df_test, y_test, C = 1.0, n_splits = 5):
    """
    Train the model on data folds and evaluate it with validation set
    Produce AUC score
    """

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print(f'C={C} {np.mean(scores):.3f} +- {np.std(scores):.3f}')


    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_test, dv, model)

    auc = roc_auc_score(y_test, y_pred)

    return dv, model, auc

def save_model(dv, model, C):
    output_file = f"model_C={C}.bin"

    with open(output_file, "wb") as f_out:
        pickle.dump((dv, model), f_out)


def main():
    C = float(sys.argv[1])
    n_splits = int(sys.argv[2])
    df = read_dataframe('Telco-Customer-Churn.csv')
    df = preprocessing(df)
    df_train_full, df_test, y_train, y_test = split_dataframe(df)

    dv, model, auc = train_running(df_train_full, df_test, y_test, C = C, n_splits = n_splits)
    save_model(dv, model, C = C)

if __name__ == '__main__':
    main()