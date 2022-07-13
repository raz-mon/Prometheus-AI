"""
Main class of forecasting package.
Here will be the implementation of the forecasting API of the project.
"""

from Methods import analyze_ts, forecast, get_ts, methods, error_metrics, metrics
from data_gen.util import get_pod_name
from menu import metric_menu, application_menu, method_menu, gran_menu, cm_menu, error_metric_menu, plot_one_menu, \
    test_len_menu
from data_gen.util import cpu_legal_pod_names, memory_legal_pod_names
from data_gen.Thanos_conn import leg
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm

def main():
    # Print menu to user, and get user-inputs.
    metric = metric_menu()
    application = application_menu(metric)
    methods = method_menu()
    test_len = test_len_menu()
    granularity = gran_menu()
    compress_method = cm_menu()
    error_metric = error_metric_menu()
    plot_one = plot_one_menu()

    if (plot_one == 'Yes'):
        ts = find_ts(metric, application, granularity, compress_method)
        forecast(ts, methods, application, error_metric, test_len, True, True)
    elif type(methods) == list:
        errs = {}
        for method in methods:
            errs[method] = total_error(method, metric, application, error_metric, granularity, compress_method, test_len)
        for method in errs:
            print(f'Error for method {method}: {errs[method]}')
    else:
        error = total_error(methods, metric, application, error_metric, granularity, compress_method, test_len)
        print(f'Total error: {error}')

def total_error(forecasting_method, metric, application, error_metric, granularity, compress_method='mean',
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
    # contents = pathlib.Path('../data/csv/seconds')
    contents = pathlib.Path('../test_dir (delete)')     # Remove this, use previous line (for testing).
    for path1 in tqdm(contents.iterdir(), colour='blue', bar_format='{l_bar}{bar}{r_bar}'):
        if leg(metric) not in str(path1):
            continue                             # Not the desired metric.
        contents2 = pathlib.Path(path1)
        # Check if the directory is of metric 'metric'. If not --> Continue.
        for path2 in contents2.iterdir():
            if 'problem_indexes' in str(path2):
                continue
            # Read csv file, check if it is of application 'application'. If not --> Continue.
            df = pd.read_csv(path2)
            pod_name = get_pod_name(df)[:len(application)]
            is_application = (pod_name == application)
            if not is_application:
                continue
            ts = get_ts(df)
            # Resample for correct granularity
            ts = resample_ts(ts, granularity, compress_method)
            ts = ts.fillna(method="ffill")
            pred, err = forecast(ts, forecasting_method, application, error_metric, test_len, False, False)
            tot_err += err
    return tot_err


def forecast_path(path, forecasting_method, error_metric, test_len, plot=False, print_err=False):
    df = pd.read_csv(path)
    ts = get_ts(df)
    pred, err = forecast(ts, forecasting_method, error_metric, test_len, plot, print_err)
    return pred, err


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
    return ts


def find_ts(metric, application, granularity, compress_method):
    """Return first time-series of metric 'metric', application 'application', resampled to granularity 'granilarity'
    with compress method 'compress method'"""
    ts = None
    found = False
    count = 0
    contents = pathlib.Path('../data/csv/seconds')
    for path1 in contents.iterdir():
        if leg(metric) not in str(path1):
            continue
        contents2 = pathlib.Path(path1)
        for path2 in contents2.iterdir():
            if 'problem_indexes' in str(path2):
                continue
            df = pd.read_csv(path2)
            pod_name = get_pod_name(df)[:len(application)]
            is_application = (pod_name == application)
            if not is_application:
                continue
            count = count + 1
            if count < 30:
                continue
            ts = get_ts(df)
            # Resample for correct granularity
            ts = resample_ts(ts, granularity, compress_method)
            ts = ts.fillna(method="ffill")
            found = True
            break
        if found:
            break
    if ts is None:
        raise Exception(f'No time-series found for metric {metric}, application {application} in our data-base!')
    return ts


def tot_err_all_apps_for_metric(forecasting_method, metric, error_metric, granularity, compress_method='mean',
                test_len=0.2):
    """Print the error recieved for every application, for a specific metric, forecasting method, granularity,
    compressing method, test-length and error metric."""
    errs = {}
    for application in cpu_legal_pod_names:
        errs[application] = total_error(forecasting_method, metric, application, error_metric, granularity, compress_method, test_len)
    for app, err in errs.items():
        print(f'Error for application {app}: {err}')
    return errs


def compare_methods(metric, application, error_metric, granularity, compress_method='mean', test_len=0.2):
    """Compare the errors received by different forecasting methods for a given metric, application, granularity,
    compressing method, test-length and error metric"""
    errs = {}
    for method in methods:
        errs[method] = total_error(method, metric, application, error_metric, granularity, compress_method, test_len)
    for method, err in errs.items():
        print(f'Error for method {method}: {err}')
    # Plot histogram of errors (bar for each method)
    plt.style.use('ggplot')
    plt.bar(list(errs.keys()), errs.values(), color='r', width=0.3)
    plt.ylabel('error')
    plt.title(f'Error per method')
    plt.show()
    return errs




if __name__ == '__main__':
    main()
























