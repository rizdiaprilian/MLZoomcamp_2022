import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import shap
import mlflow
from mlflow.models import infer_signature
import xgboost
import seaborn as sns
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

def loading_data() -> pd.DataFrame:
    data = 'CreditScoring.csv'

    df = pd.read_csv(data)
    df.columns = df.columns.str.lower()

    return df

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
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

    df.marital = df.marital.map(marital_values)
    df.home = df.home.map(home_values)
    df.records = df.records.map(records_values)
    df.status = df.status.map(status_values)
    df.job = df.job.map(job_values)

    for c in ['income', 'assets', 'debt']:
        df[c] = df[c].replace(to_replace=99999999, value=np.nan)
    
    df = df[df.status != 'unk'].reset_index(drop=True)

    return df

def prepare(df: pd.DataFrame, test_size:float):
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

    # Filling missing values with 0
    dict_train = df_train.fillna(0).to_dict(orient='records')
    dict_val = df_val.fillna(0).to_dict(orient='records')

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(dict_train)
    X_val = dv.transform(dict_val)

    # model_train = pd.DataFrame(X_train, columns=list(dv.get_feature_names_out()))
    # model_val = pd.DataFrame(X_val, columns=list(dv.get_feature_names_out()))

    return dv, X_train, X_val, y_train, y_val

def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

# def train():
#     # Fit an XGBoost binary classifier on the training data split
#     model_optimised = xgboost.XGBClassifier(**optimized_params).fit(model_train, y_train)

#     # Create a model_optimised signature
#     signature = infer_signature(model_val, model_optimised.predict(model_val))

#     # # Build the Evaluation Dataset from the test set
#     eval_data = model_val.copy()
#     eval_data["label"] = y_val

#     with mlflow.start_run() as run:
#         # Log the baseline model_optimised to MLflow
#         mlflow.sklearn.log_model(model_optimised, "model_optimised", signature=signature)
#         model_uri = mlflow.get_artifact_uri("model_optimised")

#         # Evaluate the logged model_optimised
#         result = mlflow.evaluate(
#             model_uri,
#             eval_data,
#             targets="label",
#             model_type="classifier",
#             evaluators=["default"],
#         )



def main():
    TEST_SIZE = 0.2
    
    df = loading_data()
    df = preprocessing(df)
    dv, X_train, X_val, y_train, y_val = prepare(df, TEST_SIZE)

    model_train = pd.DataFrame(X_train, columns=list(dv.get_feature_names_out()))

    features = dv.get_feature_names_out()
    dtrain = xgboost.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgboost.DMatrix(X_val, label=y_val, feature_names=features)

    experiment_id = mlflow.create_experiment("Churn")

    def objective(trial):
        with mlflow.start_run(nested=True):
            # Define hyperparameters
            params = {
                'objective': 'binary:hinge',
                'eval_metric': 'auc',
                'random_state': 23,
                'n_estimators': 200,
                'reg_alpha': trial.suggest_float('reg_alpha', 1E-10, 1E-5),
                'reg_lambda': trial.suggest_float('reg_lambda', 1E-10, 1E-5),
                'num_leaves': trial.suggest_int('num_leaves', 150, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
                'max_depth': trial.suggest_int('max_depth', 10,50),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                'min_child_weight': trial.suggest_int('min_child_samples', 1, 20),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 10)
            }

            # Train XGBoost model
            bst = xgboost.train(params, dtrain)
            preds = bst.predict(dval)
            roc_auc = roc_auc_score(y_val, preds)

            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("roc_auc", roc_auc)

        return roc_auc

    # Initiate the parent run and call the hyperparameter tuning child run logic
    with mlflow.start_run(experiment_id=experiment_id, run_name="first_attempt", nested=True):
        # Initialize the Optuna study
        study = optuna.create_study(direction="minimize")

        # Execute the hyperparameter optimization trials.
        # Note the addition of the `champion_callback` inclusion to control our logging
        study.optimize(objective, n_trials=500, callbacks=[champion_callback])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_metric", study.best_value)

        # Log tags
        mlflow.set_tags(
            tags={
                "project": "Churn Project",
                "optimizer_engine": "optuna",
                "model_family": "xgboost",
                "feature_set_version": 1,
            }
        )

        # Log a fit model instance
        model = xgboost.train(study.best_params, dtrain)

        artifact_path = "model_optimised"

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path=artifact_path,
            input_example=model_train.iloc[[0]],
            metadata={
                "model_data_version": 1
                },
        )

        # Get the logged model uri so that we can load it from the artifact store
        model_uri = mlflow.get_artifact_uri(artifact_path)

        # # Evaluate the logged model_optimised
        # result = mlflow.evaluate(
        #     model_uri,
        #     eval_data,
        #     targets="label",
        #     model_type="classifier",
        #     evaluators=["default"],
        # )

    

if __name__ =="__main__":
    main()