"""
Main class of forecasting package.
Here will be the implementation of the forecasting API of the project.
"""

from Methods import analyze_ts, forecast, get_ts
import pandas as pd
import sys

def main():
    df = pd.read_csv('../data/csv/seconds/pod_container_cpu_usage_sum_2022-05-06_09-18-36_28h/0.csv')
    ts = get_ts(df)
    analyze_ts(ts)
    if len(sys.argv) > 1:
        forecast(ts, sys.argv[1])
    else:
        method = 'fbprhophet'               # Choose any other method from the methods in methods.
        forecast(ts, method)



















if __name__ == '__main__':
    main()
































