"""
Main class of forecasting package.
Here will be the implementation of the forecasting API of the project.
"""

from Methods import analyze_ts, forecast, get_ts, methods, errors
from data_gen.util import get_pod_name
import pandas as pd
import numpy as np
import sys
import os
import pathlib

def main():

    # Print menu to user - He can choose one after the other the method, metric etc.
    """
    method = method_menu()
    metric = metric_menu()
    error_metric = error_metric_menu()
    application = application_menu()
    granularity = gran_menu()
    compress_method = cm_menu()
    """

    result, error = forecast(data_path, method, application, metric, error_metric, granularity, compress_method)






    """df = pd.read_csv('../data/csv/seconds/pod_container_cpu_usage_sum_2022-05-06_09-18-36_28h/0.csv')
    ts = get_ts(df)
    analyze_ts(ts)
    if len(sys.argv) > 1:
        forecast(ts, sys.argv[1])
    else:
        method = 'fbprhophet'               # Choose any other method from the methods in methods.
        forecast(ts, method)
    """

    # Todo: Make a function that gets the following parameters: (data_path, forecasting_method, application, metric,
    #                                                            error_metric, Granularity, compress_method)
    #  and returns the forecast of this time-series, and the loss (according to the error-metric given).


def total_error(path, forecasting_method, metric, application, error_metric, granularity, compress_method='mean',
             test_len=0.2):
    """
    Perform the forecasting method 'forecasting_method' on all the data of 'application', and
    metric 'metric' (traversing the data-directory). Resmaple the data to 'granularity', by compressing the data (resample) with method
    'compress_method'.
    :param path: Path to the data.
    :type path: str.
    :param forecasting_method: One of the methods in Methods.methods
    :type forecasting_method: str.
    :param metric: 'CPU' or 'memory' metric.
    :type metric: str.
    :param application: One of the applications from util.cpu_legal_pod_names or util.memory_legal_pod_names (also
    respective to 'metric'.
    :type application: str.
    :param error_metric: The error_metric we wish to use. One of Method.error_metrics.
    :type error_metric: str.
    :param granularity: The granularity to which the data will be transfered to. One of: {'w', 'd', 'h', 's'}
    which correspond to {week, day, hour, second}.
    :type granularity: str.
    :param compress_method: How to compress data (which is in seconds defaultly).
    One of {'mean', 'sum', 'first', 'last', 'max', 'min'}.
    :type compress_method: str.
    :param test_len: Length of test-period. For example 0.2 if we want 20% of the length of the original time-series.
    :type test_len: float.
    :return: Return the forecast of the time-series, and the error from the real values.
    :rtype: (ts, error).
    """
    tot_err = 0.0
    contents = pathlib.Path('../data/csv/seconds')
    for path1 in contents.iterdir():
        if not str(path1).__contains__(metric):
            continue            # Not the desired metric.
        contents2 = pathlib.Path(path1)
        # Check if the directory is of metric 'metric'. If not --> Continue.
        for path2 in contents2.iterdir():
            # Read csv file, check if it is of application 'application'. If not --> Continue.
            df = pd.read_csv(path2)
            if not get_pod_name(df)[:len(application)] == application:
                continue        # Not the desired application.
            ts = get_ts(df)
            # Resample for correct granularity
            resample_ts(ts, granularity, compress_method)
            _, err = forecast_ts(ts, forecasting_method, error_metric)
            tot_err += err
    return tot_err



def forecast_ts(ts, forecasting_method, error_metric):
    # Initialize forecaster
    forecaster = methods[forecasting_method]

    return pred_ts, error


def forecast(path, forecasting_method, error_metric, test_len):
    # TBD
    return None


def resample_ts(ts, granularity, compress_method):
    if compress_method == 'mean':
        ts = ts.astype(float).resample(granularity).mean()
    elif compress_method == 'sum':
        ts = ts.astype(float).resample(granularity).sum()
    elif compress_method == 'first':
        ts = ts.astype(float).resample(granularity).first()
    elif compress_method == 'last':
        ts = ts.astype(float).resample(granularity).last()
    elif compress_method == 'max':
        ts = ts.astype(float).resample(granularity).max()
    elif compress_method == 'min':
        ts = ts.astype(float).resample(granularity).min()
    elif compress_method == 'median':
        ts = ts.astype(float).resample(granularity).median()











if __name__ == '__main__':
    main()
































