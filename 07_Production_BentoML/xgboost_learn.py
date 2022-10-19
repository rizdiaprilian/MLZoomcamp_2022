import bentoml

import pandas as pd
import numpy as np
import pickle
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

from matplotlib import pyplot as plt


def read_data(file: str) -> pd.DataFrame:
    """Read data from given file
        Argument: file: name of the file to be load. The format is csv
        Return: df: Pandas Dataframe 
    """

    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()
    return df

def feature_mapping(df: pd.DataFrame) -> pd.DataFrame:
    status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
    }


    home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
    }

    marital_values = {
        1: 'single',
        2: 'married',
        3: 'widow',
        4: 'separated',
        5: 'divorced',
        0: 'unk'
    }

    records_values = {
        1: 'no',
        2: 'yes',
        0: 'unk'
    }

    job_values = {
        1: 'fixed',
        2: 'partime',
        3: 'freelance',
        4: 'others',
        0: 'unk'
    }

    df.status = df.status.map(status_values)
    df.home = df.home.map(home_values)
    df.marital = df.marital.map(marital_values)
    df.records = df.records.map(records_values)
    df.job = df.job.map(job_values)

    return df

def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing missing values
        Argument: df: Pandas DataFrame
                column: selected column to be used in find and remove missing rows
        Return: df: Pandas Dataframe after rows removal
    """
    for c in ['income', 'assets', 'debt']:
        df[c] = df[c].replace(to_replace=99999999, value=np.nan)
    df = df[df.status != 'unk'].reset_index(drop=True)
    return df

def split_data(df: pd.DataFrame):
    """Prepare train and test data with splitting
        Argument: df: Pandas DataFrame
                size: fraction split
    """
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = (df_train.status == 'default').astype('int').values
    y_val = (df_val.status == 'default').astype('int').values
    y_test = (df_test.status == 'default').astype('int').values

    del df_train['status']
    del df_val['status']
    del df_test['status']

    training_splits = {'df_val' : df_val, 
                     'y_val' : y_val, 
                     'df_test' : df_test, 
                     'y_test' : y_test}
    pickle.dump(training_splits, open('training_splits.p', 'wb'))

    return df_train, y_train, df_val, y_val

def dictVectorizer(df_train, df_val):
    """Calling DictVectorizer for encoding purpose. Input in dataframe must be
        transformed to dict form
        Argument: df_train: Pandas DataFrame
                df_val: Pandas DataFrame
        Return: X_train: Pandas DataFrame
                X_val: Pandas DataFrame
    """

    dict_train = df_train.fillna(0).to_dict(orient='records')
    dict_val = df_val.fillna(0).to_dict(orient='records')
    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(dict_train)
    X_val = dv.transform(dict_val)

    dv_dict = {'dv' : dv}
    pickle.dump(dv_dict, open('dv_vectorizer.p', 'wb'))

    return X_train, X_val, dv

def xgboost_matrix(X_train, X_val, y_train, y_val, dv):
    """Prepare matrices of train and validation data for training purpose
        Argument: X_train: Pandas DataFrame
                X_val: Pandas DataFrame
                y_train: np.array
                y_val: np.array
                dv: DictVectorizer
    """
    features = dv.get_feature_names_out()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    return dtrain, dval

def train_model(dtrain, dval):
    """Fit a XGBoost model to train matrix with specified parameters.
    Watchlist is provided to track how XGBoost makes its progress over iterations
        Argument: dtrain: XGB matrix
                dval: XGB matrix
    """
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}

    xgb_params = {
        'eta': 0.3, 
        'max_depth': 6,
        'min_child_weight': 1,
        
        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(
        params=xgb_params, 
        dtrain=dtrain, 
        num_boost_round=100,
        verbose_eval=5, 
        evals=watchlist,
        evals_result=evals_result)

    return model, evals_result

def parse_xgb_output(evals_result):
    """
    Parse output of xgb
    """
    columns = ['num_iter', 'train_auc', 'val_auc']
    train_aucs = list(evals_result['train'].values())[0]
    val_aucs = list(evals_result['val'].values())[0]

    df_scores = pd.DataFrame(
        list(zip(
            range(1, len(train_aucs)),
            train_aucs,
            val_aucs
        )), columns=columns)
    return df_scores

def main():
    # Load training data set
    df = read_data("CreditScoring.csv")
    df = feature_mapping(df)
    df = preprocessing_data(df)
    df_train, y_train, df_val, y_val = split_data(df)
    X_train, X_val, dv = dictVectorizer(df_train, df_val)

    dtrain, dval = xgboost_matrix(X_train, X_val, y_train, y_val, dv)

    model, evals_result = train_model(dtrain, dval)

    df_scores = parse_xgb_output(evals_result)
    bentoml.xgboost.save_model("credit_scoring_xgboost", model,
                        custom_objects={
                            "dictVectorizer": dv
                            })


if __name__ == "__main__":
    main()