import os, sys
import numpy as np
import pandas as pd 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pred_app import load_data, splitting_data
from statsmodels.tsa.vector_ar.var_model import VAR

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

def exp_smooth(train, test, selected_hp_values):
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

def main():
    date = float(sys.argv[1]) if len(sys.argv) > 1 else "2019-01-01"
    data = load_data()

    X_train, y_train, X_test, y_test = splitting_data(data, date)
    print(np.asarray(X_train)[:3])
    # model = VAR(endog=np.asarray(X_train))
    # model_fit = model.fit()
    # prediction = model_fit.forecast(model_fit.y, steps=len(X_test))
    # print(prediction)

if __name__ == "__main__":
    main()