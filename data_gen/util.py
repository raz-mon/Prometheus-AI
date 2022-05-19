import numpy as np
import pandas as pd
from main import OPERATE_FIRST_TOKEN, THANOS_URL
from Thanos_conn import ThanosConnect
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from prometheus_api_client.utils import parse_datetime

"""Aggregation of results of the same application"""

def clean_and_date_df(df):
    """Clean data-frame from non-relevant columnes (metric-data), change index to be the time in datetime form"""
    new_df = df.copy()
    new_df.index = new_df['time']
    new_df.index = pd.to_datetime(new_df.index, unit='s', utc=True)
    new_df = new_df['values']
    return new_df


def get_first_ts(df):
    """Get first Timestamp of the data-frame"""
    return np.array(df.index)[0]


def get_last_ts(df):
    """Get last Timestamp of the data-frame"""
    return np.array(df.index)[-1]


legal_pod_names = ['klusterlet', 'cloud-credential-operator'] # And many more.. need to write them all down..

"""
General flow:
  for pod_name in legal_pod_names:
    f1 = pd.DataFrame(['time', 'values'])   # Make new data-frame, in which we aggregate the results (not sure of syntax).
    for directory in directories:        #  (in increasing dates order - must make sure of that)
        for file in directory:
            if pod_name_no_rand(file) == pod_name:
                aggregate_results(f1, file)     # Make sure you COPY the file, do not write on at, or MOVE from it!
"""

def agg_for_app(app_name):
    """Aggregate the results saved in directory in 'path', of the pod name 'pod_name'"""
    """
    for directory in path:
        for file in directory:
            if pod_name_no_rand(file) == pod_name:
                aggregate_results(f1, file)     # MAKE SURE you COPY the file, do not write on at, or MOVE from it!
    """
    return 1    # TBD.

def aggregate_results(f1_path, f2_path):
    """Aggregate the results of the two files (.csv files)
    Returns a new data-frame, that starts with the first data-frame, and is continued by the second data-frame, with no
    intersection.
    """
    df1 = clean_and_date_df(pd.read_csv(f1_path))
    df2 = clean_and_date_df(pd.read_csv(f2_path))
    return df1.append(df2[get_last_ts(df1)])


def pod_name_no_rand(file_path):
    df = pd.read_csv(file_path)
    pod_name = df['metric data'].loc[3][5:]
    for legal_pod_name in legal_pod_names:
        if legal_pod_name == pod_name[len(legal_pod_name)]:
            return legal_pod_name
    print(f'pod name not found!')
    return 'not_found!'




























def append_simple_test():
    df1 = pd.read_csv('../data/csv/seconds/pod_container_cpu_usage_sum_2022-05-16_08-55-42_28h/1.csv')
    df1 = clean_and_date_df(df1)
    df1.plot()
    plt.show()
    # fig = go.Figure()

    # fig.add_trace(go.Scatter(x=df1.index, y=df1))

    df2 = pd.read_csv('../data/csv/seconds/pod_container_cpu_usage_sum_2022-05-17_08-38-29_25h/1.csv')
    df2 = clean_and_date_df(df2)
    df2.plot()
    plt.show()

    # fig.add_trace(go.Scatter(x=df2.index, y=df2))

    app = append_tss(df1, df2)
    app.plot()
    plt.show()

    # fig.add_trace(go.Scatter(x=app.index, y=app))
    # fig.update_layout(
    #     title="Pod memory usage: sum", xaxis_title="Time", yaxis_title="Bytes"
    # )

append_simple_test()

