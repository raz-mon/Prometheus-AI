"""
A module containing forecasting methods.
** A separate module will be written for the Deep-Learning methods which we will use. This is for the more 'classical'
   ones.**
"""

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# class Linear(object):










































