import pandas as pd
from matplotlib import pyplot as plt
from prometheus_api_client import PrometheusApiClientException
from datetime import timedelta
from config import OPERATE_FIRST_TOKEN, THANOS_URL
from thanos_api_client import ThanosConnect
import datetime


def show_graph(metric_df):
    x = metric_df['date']
    y = metric_df['value']
    y_label = metric_df['__name__'][0] if not metric_df['__name__'].empty else ""
    plt.scatter(x, y, s=2.0)
    plt.locator_params(axis="y", nbins=20)
    plt.xlabel('Date GMT+0')
    plt.ylabel(y_label)
    plt.show()


def export_to_csv(metric_df, csv_path: str, start_time: datetime):
    pivot = metric_df.pivot(index='date', columns=set(metric_df.columns) - {'timestamp', 'date', 'value'})['value']
    r_pivot = pivot.resample(timedelta(seconds=60), label='left', closed='right', origin=start_time).last()
    r_pivot.to_csv(csv_path, header=True)


def start_preprocessing(label_config: dict, start_time: datetime, end_time: datetime, step: int):
    """
    Saves the desired Prometheus data according to the label_config into a csv file based on the csv_path
    :param label_config: Metric names we query the server upon
    :param start_time: The query's start time
    :param end_time: The query's end time
    :param step: the resolution between two samples.
    """
    tc = ThanosConnect(THANOS_URL, OPERATE_FIRST_TOKEN)
    clusters = tc.query_label_values(label='cluster')
    if label_config['cluster'] not in clusters:
        raise Exception(f'Cluster {label_config["cluster"]} not found. Try one of: {clusters}')
    jobs = tc.query_label_values(label='job', cluster=label_config["cluster"])
    if label_config['job'] not in jobs:
        raise Exception(f'Cluster {label_config["job"]} not found in cluster {label_config["cluster"]}. Try one of: {jobs}')

    try:
        metric_data = tc.range_query(metric_name="NooBaa_BGWorkers_nodejs_external_memory_bytes", label_config=label_config, start_time=start_time, end_time=end_time, step=step)
    except PrometheusApiClientException as e:
        print(f"this is my exception {e}")
        if e.args[0] != 'HTTP Status Code 400 (b\'exceeded maximum resolution of 11,000 points per timeseries. Try decreasing the query resolution (?step=XX)\')':
            raise e
        chunk_size = timedelta(seconds=step)
        metric_data = tc.get_metric_range_data(metric_name=tc.build_query(label_config),
                                               start_time=start_time,
                                               end_time=end_time,
                                               chunk_size=chunk_size)
    if not metric_data:
        raise Exception("Got empty results")

    metric_df = tc.metric_data_to_df(metric_data)
    metric_df['date'] = pd.to_datetime(metric_df['timestamp'], origin='unix', unit='s')

    return metric_df


# usage example:
if __name__ == '__main__':
    # initialize arguments
    _csv_path = 'noobaa-mgmt.csv'
    _label_config = {'cluster': 'moc/smaug', 'job': 'noobaa-mgmt'}
    _start_time = datetime.datetime.strptime('2022-03-21 10:00:00', "%Y-%m-%d %H:%M:%S")
    _end_time = datetime.datetime.strptime('2022-03-29 14:00:00', "%Y-%m-%d %H:%M:%S")
    _step = 60  # seconds
    # fetch the data
    _metric_df = start_preprocessing(_label_config, _start_time, _end_time, _step)
    # export to csv
    export_to_csv(_metric_df, _csv_path, _start_time)
    # show graph
    show_graph(_metric_df)
