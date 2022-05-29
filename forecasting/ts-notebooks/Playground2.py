from prometheus_api_client import PrometheusConnect  # noqa: F401
from prometheus_api_client.metric_range_df import (  # noqa: F401
    MetricRangeDataFrame,
)
from datetime import timedelta, datetime  # noqa: F401
import pandas as pd
from sklearn import linear_model
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from scipy.stats.distributions import chi2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings


"""
metric_df = pd.read_csv('../data/csv/dated/pod_container_cpu_usage_sum_2022-05-13_08-47-47_28h/9.csv')
metric_df.index = metric_df['time']
metric_df = metric_df['values']
# print(metric_df)
# ts = metric_df.resample('min').mean()
metric_df.index = pd.to_datetime(metric_df.index, unit='s', utc=True)
"""




metric_df = pd.read_csv('../data/csv/seconds/pod_container_cpu_usage_sum_2022-05-13_08-47-47_28h/9.csv')
df = metric_df[['time', 'values']]
print(df)
df.index = df['time']
print(df)
df = df['values']
# print(df)
df.index = pd.to_datetime(df.index, unit="s", utc=True)
# print(df)
ts = df.astype(float).resample("min").mean()
# print(ts)
# ts2 = df.resample("h").mean()
# print(ts2)

# sns.set()
# ts.plot(figsize=(15, 10))
# plt.title("Visualize time series")
# plt.ylabel("Node memory active bytes")
# plt.show()





















