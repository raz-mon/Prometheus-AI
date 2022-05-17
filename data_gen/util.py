import numpy as np
import pandas as pd
from main import OPERATE_FIRST_TOKEN, THANOS_URL
from Thanos_conn import ThanosConnect
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from prometheus_api_client.utils import parse_datetime


def clean_and_date_df(df):
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


def append_tss(df1, df2):
    return df1.append(df2.loc[get_last_ts(df1):])


def append_test_simple():
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



