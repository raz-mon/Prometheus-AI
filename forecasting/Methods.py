"""
A module containing forecasting methods.
Taking inspiration from the following notebooks:
    - https://github.com/aicoe-aiops/time-series/blob/master/notebooks/ts-3-concepts.ipynb
    - https://github.com/aicoe-aiops/time-series/blob/master/notebooks/ts-4-forecasting.ipynb

** A separate module will be written for the Deep-Learning methods which we will use. This is for the more 'classical'
   ones.**
"""

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from scipy.stats.distributions import chi2
import numpy as np
import itertools
import warnings


warnings.filterwarnings("ignore")

def visualize_ts(ts, ylabel, title):
    """Visualize time series"""
    sns.set()
    ts.plot(figsize=(10, 5))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()


def white_noise(ts):
    """Generate some white noise"""
    ## Create a wn time series with mean, std, length same as our loaded data
    wn = np.random.normal(loc=ts.mean(), scale=ts.std(), size=len(ts))
    pd.DataFrame(wn).plot(figsize=(10, 5))
    plt.title("Visualize white noise")
    plt.ylabel("value")
    plt.xlabel("time")
    plt.show()


def random_walk(ts):
    """Random walk, in the same length as ts"""
    ## Randomly choose from -1, 0, 1 for the next step
    random_steps = np.random.choice(a=[-1, 0, 1], size=(len(ts), 1))
    rw = np.concatenate([np.zeros((1, 1)), random_steps]).cumsum(0)
    pd.DataFrame(rw).plot(figsize=(10, 5))
    plt.title("Visualize random walk")
    plt.ylabel("value")
    plt.xlabel("time")
    plt.show()


def get_ts(df):
    ts = df
    ts.index = pd.to_datetime(ts.index, unit='s', utc=True)
    return ts['values']


# TODO: Keep implementing more forecasting methods --> Write forecasting API.
#  Can be really nice to also use some RNNs and LSTMs.


df = pd.read_csv('../data/csv/seconds/pod_container_cpu_usage_sum_2022-05-06_09-18-36_28h/0.csv')
ts = get_ts(df)
visualize_ts(ts, 'vals', 'ts')
random_walk(ts)
white_noise(ts)





































