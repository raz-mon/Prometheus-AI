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
# from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa import SARIMAX
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

def get_ts(df):
    ts = df
    ts.index = pd.to_datetime(ts.index, unit='s', utc=True)
    return ts['values']


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


def Dickey_Fuller_test(ts):
    dft = adfuller(ts)
    print(
        f"p value {round(dft[1], 4)}",
        f"\n Test statistic {round(dft[0], 4)}",
        f"\n Critical values {dft[4]}",
    )


def seasonality_additive(ts):
    plt.rc("figure", figsize=(15, 10))
    sd_add = seasonal_decompose(ts, model="additive", period=30)
    sd_add.plot()  # Image manipulation doesn't work here.
    plt.show()


def seasonality_multiplicative(ts):
    plt.rc("figure", figsize=(15, 10))
    sd_add = seasonal_decompose(ts, model="multiplicative", period=30)
    sd_add.plot()  # Image manipulation doesn't work here.
    plt.show()


def AC(ts):
    sgt.plot_acf(ts, lags=50, zero=False)
    ## We give zero=False since correlation of a time series with itself is always 1
    plt.rc("figure", figsize=(15, 10))
    plt.ylabel("Coefficient of correlation")
    plt.xlabel("Lags")
    plt.show()


def PAC(ts):
    sgt.plot_pacf(ts, zero=False, method="ols")
    plt.rc("figure", figsize=(15, 10))
    plt.ylabel("Coefficient of correlation")
    plt.xlabel("Lags")
    plt.show()


def Linear_Regression_forecast(ts):
    train = ts[0: int(len(ts) * 0.8)]
    test = ts[int(len(ts) * 0.8):]
    train_time = [i + 1 for i in range(len(train))]
    test_time = [i + 8065 for i in range(len(test))]

    LinearRegression_train = pd.DataFrame(train)
    LinearRegression_test = pd.DataFrame(test)
    LinearRegression_train["time"] = train_time
    LinearRegression_test["time"] = test_time

    # Plot data (train and test in different colors)

    plt.figure(figsize=(16, 5))
    plt.plot(LinearRegression_train["values"], label="Train")
    plt.plot(LinearRegression_test["values"], label="Test")
    plt.legend(loc="best")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks(rotation=90)
    plt.show()


    # Training the algorithm
    lr = linear_model.LinearRegression()
    lr.fit(
        LinearRegression_train[["time"]], LinearRegression_train["values"].values
    )

    # Intercept
    print("Intercept :", lr.intercept_)
    # Coeffiecient of x : Slope
    print("Coefficient of x :", lr.coef_)


def exponential_smoothing_forecast(ts):
    train = ts[0: int(len(ts) * 0.8)]
    test = ts[int(len(ts) * 0.8):]
    train_time = [i + 1 for i in range(len(train))]
    test_time = [i + 8065 for i in range(len(test))]
    ses_train = pd.DataFrame(train)

    # Try autofit
    ses_model = SimpleExpSmoothing(ses_train["values"])
    ses_model_autofit = ses_model.fit(optimized=True, use_brute=True)
    print(ses_model_autofit.summary())


def exp_smoothing_opt_alpha_forecast(ts):
    train = ts[0: int(len(ts) * 0.8)]
    test = ts[int(len(ts) * 0.8):]
    train_time = [i + 1 for i in range(len(train))]
    test_time = [i + 8065 for i in range(len(test))]
    ses_train = pd.DataFrame(train)
    ses_model = SimpleExpSmoothing(ses_train["values"])

    # Try to optimize the coefficient by finding minimum AIC.
    min_ses_aic = 99999999
    for i in np.arange(0.01, 1, 0.01):
        ses_model_alpha = ses_model.fit(
            smoothing_level=i, optimized=False, use_brute=False
        )
        # You can print to see all the AIC values
        # print(' SES {} - AIC {} '.format(i,ses_model_alpha.aic))
        if ses_model_alpha.aic < min_ses_aic:
            min_ses_aic = ses_model_alpha.aic
            min_aic_ses_model = ses_model_alpha
            min_aic_alpha_ses = i

    print("Best Alpha : ", min_aic_alpha_ses)
    print("Best Model : \n")
    min_aic_ses_model.summary()


def holt_forecast(ts):
    # since optimization is intensive, we are sampling for this method
    # ts_holt = metric_df["value"].astype(float).resample("30min").mean()
    ts_holt = ts
    train = ts_holt[0: int(len(ts_holt) * 0.8)]
    test = ts_holt[int(len(ts_holt) * 0.8):]
    des_train = pd.DataFrame(train)

    # Try out autofit model and see what alpha and beta values are.
    des_model = Holt(des_train["values"])
    des_model_autofit = des_model.fit(optimized=True, use_brute=True)
    print(des_model_autofit.summary())


def holt_opt_alpha(ts):
    ts_holt = ts
    train = ts_holt[0: int(len(ts_holt) * 0.8)]
    test = ts_holt[int(len(ts_holt) * 0.8):]
    des_train = pd.DataFrame(train)
    des_model = Holt(des_train["values"])

    # Try to optimize the coefficient:
    min_des_aic = 99999
    for i in np.arange(0.01, 1, 0.01):
        for j in np.arange(0.01, 1.01, 0.01):
            des_model_alpha_beta = des_model.fit(
                smoothing_level=i,
                smoothing_slope=j,
                optimized=False,
                use_brute=False,
            )
            # You can print to see all the AIC values
            # print(' DES {} - AIC {} '.format(i,des_model_alpha_beta.aic))
            if des_model_alpha_beta.aic < min_des_aic:
                min_des_aic = des_model_alpha_beta.aic
                min_aic_des_model = des_model_alpha_beta
                min_aic_alpha_des = i
                min_aic_beta_des = j

    print("Best Alpha : ", min_aic_alpha_des)
    print("Best Beta : ", min_aic_beta_des)
    print("Best Model : \n")
    print(min_aic_des_model.summary())


def AR1(ts):
    # Should use the lag that gets the highest PACF coeficient.
    model_ar_1 = ARIMA(ts, order=(1, 0, 0))
    results_ar_1 = model_ar_1.fit()
    print(results_ar_1.summary())
    return results_ar_1


def AR2(ts):
    # Should use the lag that gets the highest PACF coeficient.
    model_ar_2 = ARIMA(ts, order=(2, 0, 0))
    results_ar_2 = model_ar_2.fit()
    print(results_ar_2.summary())
    return results_ar_2


def AR5(ts):
    # Should use the lag that gets the highest PACF coeficient.
    model_ar_5 = ARIMA(ts, order=(5, 0, 0))
    results_ar_5 = model_ar_5.fit()
    print(results_ar_5.summary())
    return results_ar_5


# LLR Test
def llr_test(res_1, res_2, df=1):
    l1, l2 = res_1.llf, res_2.llf
    lr = 2 * (l2 - l1)
    p = chi2.sf(lr, df).round(3)
    result = "Insignificant"
    if p < 0.005:
        result = "Significant"
    return p, result


def AR_auto(ts):
    # Automatically find the best order for the AR method.
    llr = 0
    p = 1
    results = ARIMA(ts, order=(p, 0, 0)).fit()
    while llr < 0.05:
        results_prev = results
        p += 1
        results = ARIMA(ts, order=(p, 0, 0)).fit()
        llr, _ = llr_test(results_prev, results)
        print(p, llr)


def analyze_resid3(ts):
    resid = ARIMA(ts, (3, 0, 0)).fit().resid
    resid.plot()
    plt.ylabel("Residuals")
    plt.rc("figure", figsize=(15, 10))
    plt.title("Visualizing residuals")
    print(resid.mean(), resid.var())
    plt.show()


# TODO: Keep implementing more forecasting methods (got to ARMA)--> Write forecasting API.
#  Can be really nice to also use some RNNs and LSTMs.


df = pd.read_csv('../data/csv/seconds/pod_container_cpu_usage_sum_2022-05-06_09-18-36_28h/0.csv')
ts = get_ts(df)
# visualize_ts(ts, 'vals', 'ts')
# random_walk(ts)
# white_noise(ts)
# Dickey_Fuller_test(ts)
# seasonality_additive(ts)
# seasonality_multiplicative(ts)
# AC(ts)
# PAC(ts)
# Linear_Regression_forecast(ts)
# exponential_smoothing_forecast(ts)
# exp_smoothing_opt_alpha_forecast(ts)
# holt_forecast(ts)
# holt_opt_alpha(ts)
# AR1(ts)
# AR2(ts)
# AR5(ts)
# AR_auto(ts)
# analyze_resid3(ts)            # Doesn't work at the moment.


































