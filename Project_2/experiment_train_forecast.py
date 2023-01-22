import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score 
from math import sqrt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

import mlflow
import mlflow.sklearn
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


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_predictions(y_true, y_pred, time_series_name, value_name, plot_size=(10, 7)):
    # dictionary for currying
    validation_output = {} 
    
    # full error metrics suite as shown in listing 6.6
    error_values = calculate_errors(y_true, y_pred)
    
    # store all of the raw values of the errors
    validation_output['errors'] = error_values
    
    # create a string to populate a bounding box with on the graph
    text_str = '\n'.join((
        'mae = {:.3f}'.format(error_values['mae']),
        'mape = {:.3f}'.format(error_values['mape']),
        'mse = {:.3f}'.format(error_values['mse']),
        'rmse = {:.3f}'.format(error_values['rmse']),
        'explained var = {:.3f}'.format(error_values['explained_var']),
        'r squared = {:.3f}'.format(error_values['r2']),
    )) 
    with plt.style.context(style='seaborn'):
        fig, axes = plt.subplots(1, 1, figsize=plot_size)
        axes.plot(y_true, 'b-', label='Test data for {}'.format(time_series_name))
        axes.plot(y_pred, 'r-', label='Forecast data for {}'.format(time_series_name))
        axes.legend(loc='upper left')
        axes.set_title('Raw and Predicted data trend for {}'.format(time_series_name))
        axes.set_ylabel(value_name)
        axes.set_xlabel(y_true.index.name)
        
        # create an overlay bounding box so that all of our metrics are displayed on the plot
        props = dict(boxstyle='round', facecolor='oldlace', alpha=0.5)
        axes.text(0.05, 0.9, text_str, transform=axes.transAxes, fontsize=12, verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()
    return validation_output

def calculate_errors(y_true, y_pred):
    # create a dictionary to store all of the metrics
    error_scores = {}
    # Here is populated dictionary with various metrics
    mse = mean_squared_error(y_true, y_pred)
    error_scores['mae'] = mean_absolute_error(y_true, y_pred)
    error_scores['mape'] = mape(y_true, y_pred)
    error_scores['mse'] = mse
    error_scores['rmse'] = sqrt(mse)
    error_scores['explained_var'] = explained_variance_score(y_true, y_pred)
    error_scores['r2'] = r2_score(y_true, y_pred)
    
    return error_scores


def objective(params):
    # params = {
    #           'alpha': params['alpha'], 
    #          'tol': params['tol']
    #          }

    # lr = Lasso(**params)

    params = {
                'alpha': params['alpha'], 
                'solver': params['solver'],
                'tol': params['tol'],
                'fit_intercept': params['fit_intercept']
             }
    lr = Ridge(**params)
    score = cross_val_score(lr, X_train, y_train,
            scoring="r2", cv=5).mean()

    return {'loss': score, 'status': STATUS_OK }


# #### ARIMA
def arima_(train, test):
    train_arima, test_arima = train['Log_KPC4'], test['Log_KPC4']

    history = [x for x in train_arima]
    predictions = []

    for t in range(len(test_arima)):
        model = ARIMA(history, order=(1,1,2))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(float(yhat))
        
        obs = test_arima[t]
        history.append(obs)
        
        print('predicted = %f, expected = %f' % (np.exp(yhat), np.exp(obs)))


    predictions_series = pd.Series(predictions, index = test_arima.index)
    arima_log_score = plot_predictions(test_arima, predictions_series, 
                                "ARIMA Model", "Logarithmic Index Sales per Week")


    arima_score = plot_predictions(test['KPC4'], np.exp(predictions_series), 
                                "ARIMA Model", "Index Sales per Week")

    return model_fit


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    np.random.seed(40)
    date = float(sys.argv[1]) if len(sys.argv) > 1 else "2019-01-01"
    data = load_data()

    X_train, y_train, X_test, y_test = splitting_data(data, date)
    
    ### Lasso search space
    # search_space = {   
    #     'alpha': hp.uniform("alpha", 0.1, 0.9),
    #     'tol': hp.uniform("tol", 0.0001, 0.001),
    # }

    ### Ridge search space
    search_space = {   
        'alpha': hp.uniform("alpha", 0.0001, 0.1),
        'solver': hp.choice("solver", ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "auto"]),
        'tol': hp.uniform("tol", 0.0001, 0.1),
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
    }

    mlflow.set_experiment("internet_sales_index")

    with mlflow.start_run():
        best_result = fmin(
                fn=objective, 
                space=search_space,
                algo=tpe.suggest,
                trials=Trials(),
                max_evals=70)

        # mlflow.log_param("alpha", best_result['alpha'])
        # mlflow.log_param("tol", best_result['tol'])

        # lassomodel = Lasso(**best_result)
        # lassomodel.fit(X_train, y_train)
        # y_pred = lassomodel.predict(X_test)

        mlflow.log_param("alpha", best_result['alpha'])
        mlflow.log_param("solver", best_result['solver'])
        mlflow.log_param("tol", best_result['tol'])
        mlflow.log_param("fit_intercept", best_result['fit_intercept'])


        best_result_ridge = hyperopt.space_eval(search_space, best_result)
        ridgemodel = Ridge(**best_result_ridge)
        ridgemodel.fit(X_train, y_train)
        y_pred = ridgemodel.predict(X_test)

        ### Calculate regression
        error_scores = calculate_errors(y_test, y_pred)

        mlflow.log_metric("mae", error_scores['mae'])
        mlflow.log_metric("mape", error_scores['mape'])
        mlflow.log_metric("mse", error_scores['mse'])
        mlflow.log_metric("rmse", error_scores['rmse'])
        mlflow.log_metric("explained_var", error_scores['explained_var'])
        mlflow.log_metric("r2", error_scores['r2'])
    mlflow.end_run()
    X_test['Prediction'] = y_pred
    plot_predictions(y_test, X_test['Prediction'], "Lasso Linear Model", "Index Sales per Week")
