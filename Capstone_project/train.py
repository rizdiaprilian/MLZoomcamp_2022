import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import bentoml
import xgboost as xgb
from xgboost import XGBClassifier
from enum import Enum
import sys

def read_data(data: str) -> pd.DataFrame:
    """
    Reading csv data record and load it into
    a pandas dataframe
    """
    df = pd.read_csv(data, sep=',')
    return df

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modifying data structures to the shape that 
    machine learning can learn and recognize better context/features.
    """
    cat_cols = list(filter(lambda x: x if len(df[x].unique()) <= 2 else None, df.columns))
    df[cat_cols] = df[cat_cols].astype('category')

    # Mapping values to change from numeric into boolean
    boolean_values = {
        1: 'yes',
        0: 'no'
    }

    sex_values = {
        1: 'male',
        0: 'female'
    }

    for x in cat_cols:
        if x != "sex":
            df[x] = df[x].map(boolean_values)
        else:
            df[x] = df[x].map(sex_values)
    return df

def data_splitting(df: pd.DataFrame):
    """
    Divide a whole dataframe into train-test pairs of features and targets.
    """
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = (X_train.DEATH_EVENT == 'yes').astype('int').values
    y_test = (X_test.DEATH_EVENT == 'yes').astype('int').values

    del X_train['DEATH_EVENT']
    del X_test['DEATH_EVENT']

    return X_train, y_train, X_test, y_test

def extract_feature(X_train, X_test):
    """
    Extract features in a format supported by machine learning algorithms from 
    given train-test features. DictVectorizer is used here to convert feature arrays
    to Numpy representations.
    """
    dict_train = X_train.to_dict(orient='records')
    dict_test = X_test.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(dict_train)
    X_test = dv.transform(dict_test)

    return X_train, X_test, dv

class LearnerName(str, Enum):
    decision_tree = "Decision Tree"
    randomforest = "Random Forest"
    xgboost_tree = "XGBoost"

def create_model(classifier):
    if classifier == "Decision Tree":
        return  DecisionTreeClassifier(criterion="gini", max_depth=30, min_samples_leaf=0.1,
                                  min_samples_split=0.3, 
                                  class_weight="balanced", random_state=42)
    elif classifier == "Random Forest":
        return  RandomForestClassifier(oob_score=True, 
                                        class_weight="balanced", n_jobs=-1, 
                                        n_estimators=140,
                                        max_depth=5,
                                        max_samples=0.6,
                                        random_state=42)
    elif classifier == "XGBoost":
        return XGBClassifier()

def train_sklearn(model, *data_input):
    """
    Run training on a chosen model with sklearn that will build itself
    based on labeled data.
    Note that only either DecisionTreeClassifier or RandomForestClassifier
    must be chosen as for model fitting.
    """
    X_train, y_train, X_test, y_test = data_input
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    print(roc_auc_score(y_test, y_pred))
    return model

def train_xgboost(model, *data_input):
    """
    Run training on a chosen model with xgboost that will build itself
    based on labeled data.
    """
    X_train, y_train, X_test, y_test = data_input
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    watchlist = [(dtrain, 'train'), (dtest, 'test')]

    xgb_params = {
        'eta': 0.5, 
        'max_depth': 11,
        'min_child_weight': 9.0,
        'colsample_bytree': 0.65,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'seed': 1,
        'verbosity': 1,
    }
    del model
    model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=100,
                  verbose_eval=5, evals=watchlist)

    y_pred = model_xgb.predict(dtest)
    print(roc_auc_score(y_test, y_pred))
    return model_xgb

def save_bentoml(model, dv, classifier):
    """
    Build a machine learning package from a trained model and 
    a feature extractor after training finishes.
    """
    string_id = "_".join(classifier.lower().split())
    if classifier == "XGBoost":
        bentoml.xgboost.save_model(f"heart_failure_{string_id}", model,
                        custom_objects={
                            "dictVectorizer": dv
                            },
                            signatures={
                                "predict": {
                                    "batchable": False,
                                    "batch_dim": 0,
                                }
                            }
                    )      
    else:
        bentoml.sklearn.save_model(f"heart_failure_{string_id}", model,
                        custom_objects={
                            "dictVectorizer": dv
                            },
                        signatures = {
                            "predict_proba": 
                                {"batchable": False}}
                          )

MODEL_MODULES = {
            "sklearn": train_sklearn,
            "xgboost": train_xgboost
        }


def train(model, modules: str, *data_input):
    return MODEL_MODULES[modules](model, *data_input)


def main():
    df = read_data('heart_failure_clinical_records_dataset.csv')
    df = preprocessing(df)
    X_train, y_train, X_test, y_test = data_splitting(df)
    X_train, X_test, dv = extract_feature(X_train, X_test)
    # Decision Tree, Random Forest, or XGBoost
    classifier = sys.argv[1] 
    modules = sys.argv[2]
    model = create_model(classifier)
    model_trained = train(model, modules, X_train, y_train, X_test, y_test)
    save_bentoml(model_trained, dv, classifier)

if __name__ == '__main__':
    main()