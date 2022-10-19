import bentoml
import pandas as pd
import pickle
import logging
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report
logging.basicConfig(level=logging.WARN)

def load_model(tags: str):
    return  bentoml.xgboost.load_model(tags)

def load_data() -> pd.DataFrame:
    df_val, y_val, df_test, y_test = pickle.load(open('training_splits.p', 'rb')).values()

    return df_val, y_val, df_test, y_test

def load_dictvectorizer():
    model_ref = bentoml.xgboost.get("credit_scoring_xgboost:revrrckpcs3es7fs")
    return model_ref.custom_objects["dictVectorizer"]

def transform_matrix(dv, df_val, df_test):
    dict_val = df_val.fillna(0).to_dict(orient='records')
    dict_test = df_test.fillna(0).to_dict(orient='records')
    X_val = dv.transform(dict_val)
    X_test = dv.transform(dict_test)

    return X_val, X_test

def xgboost_matrix(X_val, X_test, y_val, y_test, dv):
    """Prepare matrices of validation and test data for test purpose
        Argument: 
                X_val: Pandas DataFrame
                X_test: Pandas DataFrame
                y_val: np.array
                y_test: np.array
                dv: DictVectorizer
    """
    features = dv.get_feature_names_out()
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

    return dval, dtest

def metric_measure(xgb_model, dtest, y_test):
    predicted_y = xgb_model.predict(dtest)
    # report = classification_report(y_test, predicted_y)
    rocauc = roc_auc_score(y_test, predicted_y)
    return rocauc

def main():
    df_val, y_val, df_test, y_test = load_data()
    dv = load_dictvectorizer()
    print(dv)
    X_val, X_test = transform_matrix(dv, df_val, df_test)
    dval, dtest = xgboost_matrix(X_val, X_test, y_val, y_test, dv)
    xgb_model = load_model("credit_scoring_xgboost:revrrckpcs3es7fs")
    rocauc = metric_measure(xgb_model, dtest, y_test)
    print(rocauc)


if __name__ == "__main__":
    main()