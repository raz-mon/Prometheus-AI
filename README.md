# <ins>Prometheus-AI</ins>
Applying classical and Deep-Learning algorithms to time-series data, collected through Prometheus (Thanos).

## <ins>Project goals</ins>
In this project, we wish to write an API that performs the following tasks:
 * Poll data from Prometheuse constantly. On this data we will use our classical and Deep-Learning algorithms.
 * Organize the data in an orderly fasion on the local computer running the queries (see file-paths before running).
 * Offer the user to pick the following parameters when performing forecasting and error calculation (goodness of a forecasting method on the data-set):
   * `Metric` (cpu or memory in this case).
   * `Application` (from the applications monitored - see list in data_get/utils.py).
   * `Forecasting method` (See implemented methods in forecasting/Methods.py).
   * `Test segment length` (proportional to whole data-length).
   * `Granularity` of the data.
   * `Compression method` used to aggregate data-samples to the wanted granularity.
   * `Error metric`.

Other than these options, the user can also choose if he wants to:
  * Traverse the whole data-set, aggregating the error of a specific metric and application using a specific forecasting method.
  * Traverse the data-set, until finding one of the results received for specific metric and application, and plotting the forecast (and train period) of that data-sample (this option will also print the error).

All these options are chosen by the user via the menu-interface that is printed to the user (see forecasting/main_fore.py).

## <ins>Introduction and motivation</ins>
Time-series (ts) forecasting has been a major field interested and invested in by the industry for the last 10-20 years. In years prior to 2019, mainly classical forecasting algorithms were used (e.g., ARIMA models, exponential smoothing etc.). 
In 2019, a brakethrough was achieved for forecasting using Deep-Learning methods (See [DeepAR](https://arxiv.org/abs/1704.04110)). From then an on, forecasting via Deep-Learning methods gained more and more attention, resulting in some very good models used today.
In this project we wish to implement some of the classical methods, along with some Deep-Learning methods, and apply them to a data-set of time-series we will aggregate throughout the project timeline. 

## <ins>How to run the code:</ins>
From the root directory, run

```
python forecasting/main_fore.py
```

in order to run the menu-code from which you can choose the forecasting parameters. Executing this module will also run the forecasting method chosen on the data chosen, and plot the result or aggregate the loss respectively to the choice of 'plot one?'.

There are many supplamentary methods you can run in order to manually control the data inserted, the forecasting method parameters etc.

Some more execution options - `TBD`.


## <ins>Useful links:</ins>

### Theory:
1. **[DeepAR](https://arxiv.org/abs/1704.04110)**
2. **[Comparing Classical and Machine Learning Algorithms for Time Series Forecasting](https://machinelearningmastery.com/findings-comparing-classical-and-machine-learning-methods-for-time-series-forecasting/)**
3. **[An overview of time series forecasting models](https://towardsdatascience.com/an-overview-of-time-series-forecasting-models-a2fa7a358fcb)**
4. **[Forecasting: Principles and Practice](https://otexts.com/fpp2/)**


### <ins>Notebooks</ins>
1. **[Fetching metrics data](https://github.com/aicoe-aiops/time-series/blob/master/notebooks/ts-1-fetching-metrics.ipynb)**
2. **[Manipulating and visualizing metrics data](https://github.com/aicoe-aiops/time-series/blob/master/notebooks/ts-2-visualization.ipynb)**
3. **[Concepts](https://github.com/aicoe-aiops/time-series/blob/master/notebooks/ts-3-concepts.ipynb)**
4. **[Forecasting](https://github.com/aicoe-aiops/time-series/blob/master/notebooks/ts-4-forecasting.ipynb)**

### <ins>Time series data manipulation and visualization</ins>
1. **[pandas library.](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)**
2. **[Modern pandas.](https://tomaugspurger.github.io/modern-7-timeseries)**
3. **[Working with time-series.](https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html)**
4. **[Time Series Analysis in Python](https://www.youtube.com/playlist?list=PLtIY5kwXKny91_IbkqcIXuv6t1prQwFhO)**


  
### <ins>Jupyter Hub</ins>
1. **[Jupyterhub](https://jupyterhub-opf-jupyterhub.apps.smaug.na.operate-first.cloud/hub/spawn-pending/raz-mon)**

### <ins>Forecasting algorithms & Implementations</ins>
1. **[DeepAR example](https://github.com/zhykoties/TimeSeries)**
 
