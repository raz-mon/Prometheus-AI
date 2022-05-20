import pandas as pd
import pathlib
from util import clean_and_date_df, get_pod_name, get_last_ts, get_first_ts, pod_name_no_rand, aggregate_results_df
# import util
import os

"""Aggregation of results of the same applications, from the data polled from Prometheus"""

# for app_name in util.legal_pod_names:

def aggregate_app(app_name):
    """Aggregate the results of an application, defined by its name.
    For now, this iterates over test_dir. To be changed to the main data directory.
    """
    agg_df = None
    contents = pathlib.Path('../test_dir').iterdir()
    for path in contents:
        contents2 = pathlib.Path(path).iterdir()

        for path2 in contents2:
            # print(path2)
            df1 = pd.read_csv(path2)
            # if pod_name_no_rand == pod_name       # This is what is should be..
            if (not str(path2).__contains__('problem_indexes')) and (get_pod_name(df1)[:len(pod_name)] == pod_name):
                df1 = clean_and_date_df(df1)
                if agg_df is None:
                    print(f'first aggregation!!')
                    agg_df = df1
                    break
                else:
                    print(f'Another aggregation!!')
                    agg_df = aggregate_results_df(agg_df, df1)
                    break

    agg_df.to_csv('../aggregated.csv')
# aggregate_app('cloud-credential-operator')




def from_dated_to_timestamp_csv(path):
    """From datetime index to seconds (timestamp - the original form)"""
    df = pd.read_csv(path)
    df.index = df['time']
    df = df['values']
    df.index = pd.DatetimeIndex(df.index)
    df.index = [d.timestamp() for d in df.index]
    df.to_csv(f'{path[:-4]}_seconds.csv')
    return df
# from_dated_to_timestamp_csv('../aggregated.csv')
# df = pd.read_csv('../aggregated_seconds.csv')
# df = clean_and_date_df(df)
# df.to_csv('another_one_dated.csv')
# from_dated_to_timestamp_csv('../another_one_dated.csv')




















