import pandas as pd
import pathlib
from util import clean_and_date_df, get_pod_name, get_last_ts, get_first_ts, pod_name_no_rand, aggregate_results_df, \
    get_df, cpu_legal_pod_names, memory_legal_pod_names
import os

"""Aggregation of results of the same applications, from the data polled from Prometheus"""

# for app_name in util.legal_pod_names:

def aggregate_app(path, app_name, metric):
    """Aggregate the results of an application, defined by its name.
    For now, this iterates over test_dir. To be changed to the main data directory.
    path is supposed to be '../data/csv/seconds', but can be changed for debugging purposes.
    """
    agg_df = None
    contents = pathlib.Path(path).iterdir()
    num_of_files = 0    # Number of query-results in which the application returned a result.
    tot_num_of_files = 0    # Total number of files
    for path_ in contents:
        if not str(path_).__contains__(metric):
            continue
        contents2 = pathlib.Path(path_).iterdir()
        tot_num_of_files += 1

        for path2 in contents2:

            df1 = pd.read_csv(path2)
            # if pod_name_no_rand == pod_name       # This is what is should be.
            # Todo: Replace second condition with the one above when it works fine.
            if (not str(path2).__contains__('problem_indexes')) and (get_pod_name(df1)[:len(app_name)] == app_name):
                df1 = clean_and_date_df(df1)
                df1 = df1.astype(float).resample("h").mean()
                if agg_df is None:
                    num_of_files += 1
                    agg_df = df1
                    # Todo: Replace break with wanted behaviour (ask Ilya).
                    break
                else:
                    num_of_files += 1
                    agg_df = aggregate_results_df(agg_df, df1)
                    break
    # Todo: Fill the missing data-points (mean of last or copy last value).
    file_name = f'../data/Resampled_hour_aggregated/{app_name}_{metric}_res-h_agg.csv'
    if not agg_df is None:
        agg_df.to_csv(file_name)        # Save to .csv file.
    # Print statistics
    print(f'--------------------------------------------------------------------------------------')
    print(f'Application {app_name} was found in ({num_of_files}/{tot_num_of_files}) query results')
    print(f'Generated file {file_name}')

# for app_name in memory_legal_pod_names:
#     aggregate_app('../data/csv/seconds', app_name, 'memory_usage')

def aggregate_all_pods(metric):
    """
    Aggregate all of the results for each pod.
    :param metric: The metric to aggregate, either 'memory_usage' or 'cpu_usage'.
    :type metric: str
    """
    if metric == 'memory_usage':
        for app_name in memory_legal_pod_names:
            aggregate_app('../data/csv/seconds', app_name, 'memory_usage')
    elif metric == 'cpu_usage':
        for app_name in cpu_legal_pod_names:
            aggregate_app('../data/csv/seconds', app_name, 'cpu_usage')

def from_dated_to_timestamp_csv(path):
    """From datetime index to seconds (timestamp - the original form)"""
    df = pd.read_csv(path)
    df.index = df['time']
    df = df['values']
    df.index = pd.DatetimeIndex(df.index)
    df.index = [d.timestamp() for d in df.index]
    df.to_csv(f'{path[:-4]}_seconds.csv')
    return df






"""
Running data-aggregating code for 'test_dir':

aggregate_app('../test_dir', 'cloud-credential-operator', 'cpu_usage')
aggregate_app('../test_dir', 'cloud-credential-operator', 'memory_usage')
"""



"""
Running from_dated_to_timestamp_csv

from_dated_to_timestamp_csv('../aggregated.csv')
df = pd.read_csv('../aggregated_seconds.csv')
df = clean_and_date_df(df)
df.to_csv('another_one_dated.csv')
from_dated_to_timestamp_csv('../another_one_dated.csv')

"""









