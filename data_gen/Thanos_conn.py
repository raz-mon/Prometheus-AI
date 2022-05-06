import traceback

from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prometheus_api_client.utils import parse_datetime
import os
import plotly.express as px
import plotly.graph_objects as go


class ThanosConnect(PrometheusConnect):
    def __init__(self, url, op_token):
        super().__init__(url=url,
                         headers={"Authorization": f"Bearer {op_token}"},
                         disable_ssl=False)

    def get_data(self, metric_name, start_time, end_time, date_time_for_file, time_back, label_config=None, path=None, file_type='csv',
                 show_fig=False):
        """
        Get data from Prometheus, according to the following parameters:
        :param time_back: How much time back from the current time is the request (query).
        :type time_back: str.
        :param date_time_for_file: Date and time of query - ready to be inserted to file-name (i.e., no ':' inside).
        :type date_time_for_file: str.
        :param metric_name: Metric name.
        :type metric_name: str.
        :param label_config: More detail about the specific metric desired.
        :type label_config: dict.
        :param start_time: Start time of the received data.
        :type start_time: datetime object.
        :param end_time: End time of the received data.
        :type end_time: datetime object.
        :param path: Path to save data in.
        :type path: str.
        :param file_type: file type to save the data (.csv, .xlsx).
        :type file_type: str.
        :param show_fig: Show the data plotted (boolean).
        :type show_fig: bool.
        :return: The return value of get_metric_range_data (from Prometheus API).
        """
        dat = self.get_metric_range_data(metric_name,
                                         label_config=label_config,
                                         start_time=start_time,
                                         end_time=end_time
                                         )
                                         # Other optional ways:
                                         # start_time= (datetime.now() - datetime.timedelta(hours=8)),
                                         # end_time=datetime.now()
        print(f'Got {len(dat)} results for metric {metric_name}.')

        # Save data (Make directory of metric data if does not exist, and save each result received by its file name).
        metric_name_leg = leg(metric_name)
        # current_date_time_str = str(parse_datetime("now"))
        # date_time_for_file = current_date_time_str[:10] + '_' + current_date_time_str[11:13] + \
        #                      '-' + current_date_time_str[14:16] + '-' + current_date_time_str[17:19]
        data_path_seconds = f"../data/csv/seconds/{metric_name_leg}_{date_time_for_file}_{time_back}"
        data_path_dates = f'../data/csv/dated/{metric_name_leg}_{date_time_for_file}_{time_back}'
        os.mkdir(data_path_dates)
        os.mkdir(data_path_seconds)

        problem_indexes = []            # list for the indexes of the files with problems.
        for i in range(len(dat)):
            if good_result(dat[i]['metric']['pod']):
                save_data(dat[i], data_path_seconds, data_path_dates, str(label_config), i, file_type, problem_indexes)
            else:
                print(f"bad result: {dat[i]['metric']['pod']}")

        # Save the indexes of the files with problems (mostly some size issue).
        df = pd.DataFrame(np.array(problem_indexes))
        df.to_csv(f'{data_path_seconds}/problem_indexes.{file_type}')
        df.to_csv(f'{data_path_dates}/problem_indexes.{file_type}')

        # Show and save the figures of the data, according to show_fig, path (see specification).
        plot_data(show_fig, dat, metric_name, label_config, path)

        return dat


def save_data(data, data_path_seconds, data_path_dates, label_config, i, file_type='csv', problem_indexes=[]):
    """
    Save the received data.
    :param data: The data to save (one of the data-series received from the request).
    :param data_path_seconds: Path for saving data with seconds.
    :param data_path_dates: Path for saving data with dates.
    :param label_config: More details on the metric.
    :param i: The index of this time-series data (in received array).
    :param file_type: Type of file to save the data in (.csv, .xlsx).
    :return: Void.
    """

    # df = MetricRangeDataFrame(data)
    d1 = {'time': [x[0] for x in data['values']], 'values': [x[1] for x in data['values']]}
    d2 = {'metric data': [f'{item[0]}: {item[1]}' for item in data['metric'].items()] +
                         ['' for i in range(len(data['values']) - len(data['metric'].items()))]}
    d1.update(d2)
    try:
        pd.DataFrame(d1).to_csv(f'{data_path_seconds}/{i}.{file_type}')
        # Save a .csv file with dates instead of integers.
        dates_df = pd.DataFrame(d1)
        dates_df['time'] = pd.to_datetime(dates_df['time'], unit='s', utc=True)
        dates_df.to_csv(f'{data_path_dates}/{i}.{file_type}')
    except:
        print(f'problem occured with data{i}.')
        problem_indexes += [i]
        # traceback.print_exc()



def plot_data(show, dat, metric_name, label_config=None, path=None):
    """
    Show figurs of the data received.
    :param show: Show data, or only save it.
    :type show: Boolean.
    :param dat: Array of all received data from the request.
    :param metric_name: Metric name.
    :param label_config: More details of the data.
    :param path: Save the figures in path 'path', if no path given - don't save.
    :return: Nothing (void).
    """
    # Todo: Add option to visualize the data with go (plotly.graph_objects) or px (plotly.express).

    # Make directory for this query.
    if not (path is None):
        os.mkdir(path)

    # Plot data, and save all figures if path != None.
    if show or (path is not None):
        for i in range(len(dat)):
            data = dat[i]
            x0 = data['values'][0][0]    # Start time (to normalize others)
            xs = [(x[0] - x0) for x in data['values']]
            ys = [float(x[1]) for x in data['values']]
            plt.title(metric_name)
            plt.ylabel('value')
            plt.xlabel('time[s]')
            plt.plot(xs, ys)
            if not (path is None):
                plt.savefig(path + f'/{i}.png')
        # if show=True -> Show all figures after creating them.
        if show:
            plt.show()

        # Conversion to real date-time:
        # m1_df = MetricRangeDataFrame(data[i])
        # m1_df.index = pd.to_datetime(m1_df.index, unit="s", utc=True)


def leg(s: str):
    """
    Remove all appearances of ':' in string s, so it can be inserted to the name of the file we save.
    :param s: String to manipulated, in order for it to be safely part of a path of a saved file.
    :type s: str.
    :return: The manipulated safe-to-save string.
    :rtype: str.
    """
    ret = s
    while (ret.find(':')) >= 0:
        ind = ret.find(':')
        ret = ret[0:ind] + '_' + ret[ind + 1:len(ret)]
    return ret


def current_time_for_file():
    """
    Returns the current time and date, in a fashion that can be inserted to a path of a saved file.
    :return: Current date and time, in legal-to-save fashion.
    :rtype: str.
    """
    current_date_time_str = str(parse_datetime("now"))
    date_time_for_file = current_date_time_str[:10] + '_' + current_date_time_str[11:13] + \
                         '-' + current_date_time_str[14:16] + '-' + current_date_time_str[17:19]
    return date_time_for_file


def good_result(pod_name):
    # Todo: Add other Kubernetes-internal pods.
    bad_result_pod_names = ['apiserver', 'cluster-manager', 'etcd', 'coredns']
    # If one of the components of 'bad_result_pod_names' appears as the beginning of the pod name -> Return false.
    # Otherwise return true;
    for p_name in bad_result_pod_names:
        if len(pod_name) >= len(p_name):
            if pod_name[0:len(p_name)] == pod_name:
                return False
    return True











