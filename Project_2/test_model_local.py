import os, sys
import numpy as np
import pandas as pd 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from ets_experiment import load_data, splitting_data
from ets_experiment import exp_smoothing_bayesian, calculate_errors, extract_param_count_hwes

import pickle

def main():
    np.random.seed(40)
    date = float(sys.argv[1]) if len(sys.argv) > 1 else "2019-01-01"
    data = load_data()

    X_train, y_train, X_test, y_test = splitting_data(data, date)
    best_result = {
        'model': {
            'trend': 'mul',
            'seasonal': 'add',
            'damped_trend': False
        },
        'fit': {
            'smoothing_level': 0.019176,
            'smoothing_trend': 0.03367,
            'smoothing_seasonal': 0.98565,
            'damping_trend': 0.061314,
            'method': "trust-constr",
            'remove_bias': False
        }
    }
    param_count = extract_param_count_hwes(best_result)
    model_results = exp_smoothing_bayesian(y_train, y_test, best_result)
    error_scores = calculate_errors(y_test, model_results['forecast'], param_count)
    # print(error_scores)

    # with open('holt_winter_model.pickle', 'wb') as handle:
    #     pickle.dump(model_results['model'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_test['Prediction'] = model_results['model'].forecast(len(y_test))

    d = {'ground_truth': y_test, 'pred': X_test['Prediction']}
    chart_data = pd.DataFrame(data=d)

    print(chart_data)



if __name__ == "__main__":
    main()