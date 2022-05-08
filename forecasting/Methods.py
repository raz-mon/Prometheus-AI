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

"""Visualize time series"""
metric_df = pd.read_pickle("../data/raw/ts.pkl")
ts = metric_df["value"].astype(float).resample("min").mean()
sns.set()
ts.plot(figsize=(15, 10))
plt.title("Visualize time series")
plt.ylabel("Node memory active bytes")
plt.show()


"""Generate some white noise"""
## Create a wn time series with mean, std, length same as our loaded data
wn = np.random.normal(loc=ts.mean(), scale=ts.std(), size=len(ts))
pd.DataFrame(wn).plot(figsize=(15, 10))
plt.title("Visualize white noise")
plt.ylabel("value")
plt.xlabel("time")
plt.show()


"""Random walk"""
## Randomly choose from -1, 0, 1 for the next step
random_steps = np.random.choice(a=[-1, 0, 1], size=(len(ts), 1))
rw = np.concatenate([np.zeros((1, 1)), random_steps]).cumsum(0)
pd.DataFrame(rw).plot(figsize=(15, 10))
plt.title("Visualize random walk")
plt.ylabel("value")
plt.xlabel("time")
plt.show()



# class Linear(object):







































