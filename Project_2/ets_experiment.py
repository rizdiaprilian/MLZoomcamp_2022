import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score 
from math import sqrt
from functools import partial


import mlflow
import mlflow.statsmodels
import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

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



def extract_param_count_hwes(config):
    return len(config['model'].keys()) + len(config['fit'].keys())

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def aic(n, mse, param_count):
    return n * np.log(mse) + 2 * param_count

def bic(n, mse, param_count):
    return n * np.log(mse) + param_count * np.log(n)

def calculate_errors(y_true, y_pred, param_count):
    # create a dictionary to store all of the metrics
    error_scores = {}
    pred_length = len(y_pred)
    try: 
        mse = mean_squared_error(y_true, y_pred)
    except ValueError:
        mse = 1e12
    try:
        error_scores['mae'] = mean_absolute_error(y_true, y_pred)
    except ValueError:
        error_scores['mae'] = 1e12
    error_scores['mape'] = mape(y_true, y_pred)
    error_scores['mse'] = mse
    error_scores['rmse'] = sqrt(mse)
    error_scores['aic'] = aic(pred_length, mse, param_count)
    error_scores['bic'] = bic(pred_length, mse, param_count)
    try:
        error_scores['explained_var'] = explained_variance_score(y_true, y_pred)
    except ValueError:
        error_scores['explained_var'] = -1e4
    try:
        error_scores['r2'] = r2_score(y_true, y_pred)
    except ValueError:
        error_scores['r2'] = -1e4
    
    return error_scores

def exp_smoothing_bayesian(train, test, selected_hp_values):
    output = {}
    exp_smoothing_model = ExponentialSmoothing(train,
                               trend=selected_hp_values['model']['trend'],
                               seasonal=selected_hp_values['model']['seasonal'],
                               damped_trend=selected_hp_values['model']['damped_trend'],
                               initialization_method=None
                                              )

    exp_fit = exp_smoothing_model.fit(smoothing_level=selected_hp_values['fit']['smoothing_level'],
                        smoothing_trend=selected_hp_values['fit']['smoothing_trend'],
                          smoothing_seasonal=selected_hp_values['fit']['smoothing_seasonal'],
                          damping_trend=selected_hp_values['fit']['damping_trend'],
                          method=selected_hp_values['fit']['method'],
                          remove_bias=selected_hp_values['fit']['remove_bias']
                         )

    forecast = exp_fit.predict(train.index[-1], test.index[-1])
    output['model'] = exp_fit
    output['forecast'] = forecast[1:]
    return output

def hwes_minimization_function(selected_hp_values, train, test, loss_metric):
    model_results = exp_smoothing_bayesian(train, test, selected_hp_values)
    errors = calculate_errors(test, model_results['forecast'], extract_param_count_hwes(selected_hp_values))

    return {'loss': errors[loss_metric], 'status': STATUS_OK}


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    np.random.seed(40)
    date = float(sys.argv[1]) if len(sys.argv) > 1 else "2019-01-01"
    data = load_data()

    X_train, y_train, X_test, y_test = splitting_data(data, date)

    ### For testing purpose before heading to optimization experiment
    # space_ets = {
    #     'model': {
    #         'trend': 'add',
    #         'seasonal': 'mul',
    #         'damped_trend': True
    #     },
    #     'fit': {
    #         'smoothing_level': 0.5,
    #         'smoothing_trend': 0.5,
    #         'smoothing_seasonal': 0.5,
    #         'damping_trend': 0.5,
    #         'method': "L-BFGS-B" ,
    #         'remove_bias': True
    #     }
    # }
    # model_results = exp_smoothing_bayesian(y_train, y_test, space_ets)
    # param_count = extract_param_count_hwes(space_ets)
    # print(f"param count is {param_count}")
    # errors = calculate_errors(y_test, model_results['forecast'], param_count)
    # print(errors)

    ### Define hyperparameter space to be explored and tested 
    hpopt_space = {
        'model': {
            'trend': hp.choice('trend', ['add', 'mul']),
            'seasonal': hp.choice('seasonal', ['add', 'mul']),
           
            'damped_trend': hp.choice('damped', [True, False])
        },
        'fit': {
            'smoothing_level': hp.uniform('smoothing_level', 0.01, 0.99),
            'smoothing_trend': hp.uniform('smoothing_trend', 0.01, 0.99),
            'smoothing_seasonal': hp.uniform('smoothing_seasonal', 0.01, 0.99),
            'damping_trend': hp.uniform('damping_trend', 0.01, 0.99),
            'method': hp.choice('method', ["L-BFGS-B" , "TNC", "SLSQP","Powell", 
                            "trust-constr", "basinhopping", "least_squares" ]),
            'remove_bias': hp.choice('remove_bias', [True, False])
        }
    }

    param_count = extract_param_count_hwes(hpopt_space)
    ### Set the given experiment as the active experiment
    mlflow.set_experiment("internet_sales_index")
    ### Starting MLFlow execution
    with mlflow.start_run():
        ### Begin optimization process
        best_result = fmin(partial(hwes_minimization_function, 
                          train=y_train, 
                          test=y_test,
                          loss_metric='mse' ### switch whichever to 'mse','rmse','mae','mape'
                         ), 
                  space=hpopt_space, 
                algo=tpe.suggest, 
                  max_evals=30, 
                  trials=Trials(),
                 )

        ### Modify output structure from optimization process to form similar to dict hpopt_space
        best_result = hyperopt.space_eval(hpopt_space, best_result)

        ### Logging optimized parameter result into MLFlow
        mlflow.log_param("seasonal", best_result['model']['seasonal'])
        mlflow.log_param("trend", best_result['model']['trend'])
        mlflow.log_param("damped_trend", best_result['model']['damped_trend'])

        mlflow.log_param("smoothing_level", best_result['fit']['smoothing_level'])
        mlflow.log_param("smoothing_trend", best_result['fit']['smoothing_trend'])
        mlflow.log_param("smoothing_seasonal", best_result['fit']['smoothing_seasonal'])
        mlflow.log_param("damping_trend", best_result['fit']['damping_trend'])
        mlflow.log_param("method", best_result['fit']['method'])
        mlflow.log_param("remove_bias", best_result['fit']['remove_bias'])

        ### Building Exponential Smooth using optimized parameter
        model_results = exp_smoothing_bayesian(y_train, y_test, best_result)

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


