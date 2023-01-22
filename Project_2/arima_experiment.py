import sys
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from functools import partial

import mlflow
import mlflow.statsmodels
import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from ets_engine import extract_param_count_hwes, calculate_errors, exp_smoothing_bayesian

plt.style.use('fivethirtyeight')

def load_data():
    data = pd.read_csv(
        "Internet_sales_UK_preprocessed.csv",
        parse_dates=["date"],
        index_col=["date"],
    )

    int_col = list(data.select_dtypes("int").columns)
    float_col = list(data.select_dtypes("float").columns)
    data[int_col] = data[int_col].astype('int16')
    data[float_col] = data[float_col].astype('float32')

    data['Log_KPC4'] = np.log(data['KPC4'])
    data['Log_KPB8'] = np.log(data['KPB8'])

    kpc4_log_diff = data['Log_KPC4'].diff()
    kpc4_log_diff = kpc4_log_diff.dropna()
    kpb8_log_diff = data['Log_KPB8'].diff()
    kpb8_log_diff = kpb8_log_diff.dropna()
    return data

def splitting_data(data, split_date):
    train = data.loc[data.index < split_date]
    test = data.loc[data.index >= split_date]

    # the target variable
    y_train = train["KPC4"].copy()
    y_test = test["KPC4"].copy()

    # remove raw time series from predictors set
    X_train = train.drop(['KPC4','KPB8','KPB8_lag_1', 'KPB8_lag_3',
                        'KPB8_lag_6', 'KPB8_lag_12',
                        'KPB8_window_3_mean', 'KPB8_window_3_std',
                                'KPB8_window_6_mean', 'KPB8_window_6_std'], axis=1)
    X_test = test.drop(['KPC4','KPB8','KPB8_lag_1', 'KPB8_lag_3',
                        'KPB8_lag_6', 'KPB8_lag_12',
                        'KPB8_window_3_mean', 'KPB8_window_3_std',
                        'KPB8_window_6_mean', 'KPB8_window_6_std'], axis=1)

    return X_train, y_train, X_test, y_test

def arima_bayesian(endog, train, test, selected_hp_values):
    output = {}
    arima_model = ARIMA(endog=endog, 
                               trend=selected_hp_values['model']['trend'],
                                order=(2,0,0),
                               enforce_stationarity=selected_hp_values['model']['enforce_stationarity'],
                               concentrate_scale=selected_hp_values['model']['concentrate_scale']
                                              )

    arima_fit = arima_model.fit(
                            cov_type=selected_hp_values['fit']['cov_type'],
                         )

    forecast = arima_fit.predict(train.index[-1], test.index[-1])
    output['model'] = arima_fit
    output['forecast'] = forecast[1:]
    return output

def hwes_minimization_function(selected_hp_values, endog, train, test, loss_metric):
    model_results = arima_bayesian(endog, train, test, selected_hp_values)
    errors = calculate_errors(test, model_results['forecast'], extract_param_count_hwes(selected_hp_values))

    return {'loss': errors[loss_metric], 'status': STATUS_OK}


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    np.random.seed(40)
    date = float(sys.argv[1]) if len(sys.argv) > 1 else "2019-01-01"
    data = load_data()

    X_train, y_train, X_test, y_test = splitting_data(data, date)

    endog=y_train
    exog=X_train

    ### Define hyperparameter space to be explored and tested 
    hpopt_space = {
        'model': {
            'trend': hp.choice('trend', ["n","c","t","ct"]), 
            'enforce_stationarity': hp.choice('enforce_stationarity', [True, False]),
            'concentrate_scale': hp.choice('concentrate_scale', [True, False])
        },
        'fit': {
            'cov_type': hp.choice('cov_type', ["oim","approx","robust","none"]),
        }
    }

    param_count = extract_param_count_hwes(hpopt_space)
    ### Set the given experiment as the active experiment
    mlflow.set_experiment("internet_sales_index")
    ### Starting MLFlow execution
    with mlflow.start_run():
        ### Begin optimization process
        best_result = fmin(partial(hwes_minimization_function,
                            endog=endog,  
                            train=y_train, 
                            test=y_test,
                            loss_metric='mse' ### switch whichever to 'mse','rmse','mae','mape'
                            ), 
                    space=hpopt_space, 
                    algo=tpe.suggest, 
                    max_evals=90, 
                    trials=Trials(),
                    )

        ### Modify output structure from optimization process to form similar to dict hpopt_space
        best_result = hyperopt.space_eval(hpopt_space, best_result)

        ### Logging optimized parameter result into MLFlow
        mlflow.log_param("concentrate_scale", best_result['model']['concentrate_scale'])
        mlflow.log_param("enforce_stationarity", best_result['model']['enforce_stationarity'])
        mlflow.log_param("trend", best_result['model']['trend'])

        mlflow.log_param("cov_type", best_result['fit']['cov_type'])
        
        ### Building ARIMA using optimized parameter
        endog=y_train
        model_results = arima_bayesian(endog, y_train, y_test, best_result)

        ### Produce dicts containing errors made from model with optimized parameters
        error_scores = calculate_errors(y_test, model_results['forecast'], param_count)
        
        ### Logging error result into MLFlow
        mlflow.log_metric("mae", error_scores['mae'])
        mlflow.log_metric("mape", error_scores['mape'])
        mlflow.log_metric("mse", error_scores['mse'])
        mlflow.log_metric("rmse", error_scores['rmse'])
        mlflow.log_metric("aic", error_scores['aic'])
        mlflow.log_metric("bic", error_scores['bic'])
        mlflow.log_metric("explained_var", error_scores['explained_var'])
        mlflow.log_metric("r2", error_scores['r2'])

    ### MLFlow execution ends here
    mlflow.end_run() 