from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ThanosConnect(PrometheusConnect):
    def __init__(self, url, op_token):
        super().__init__(url=url,
                         headers={"Authorization": f"Bearer {op_token}"},
                         disable_ssl=False)

    def get_data(self, metric_name, start_time, end_time, label_config=None, path=None, file_type='csv',
                 show_fig=False):
        """
        Get data from Prometheus, according to the following parameters:
        :param metric_name: Mecric name.
        :param label_config: More detail about the specific metric desired.
        :param start_time: Start time of the received data.
        :param end_time: End time of the received data.
        :param path: Path to save data in.
        :param file_type: file type to save the data (.csv, .xlsx).
        :param show_fig: Show the data plotted (boolean).
        :return: The return value of get_metric_range_data (from Prometheus API).
        """
        # Todo: Add option to visualize the data with go (plotly.graph_objects).
        dat = self.get_metric_range_data(metric_name,
                                         label_config=label_config,
                                         start_time=start_time,
                                         end_time=end_time
                                         )
                                         # Other optional ways:
                                         # start_time= (datetime.now() - datetime.timedelta(hours=8)),
                                         # end_time=datetime.now()
        print(f'Got {len(dat)} results.')

        # Save data (Make directory of metric data if does not exist, and save each result received by its file name.
        for i in range(len(dat)):
            save_data(dat[i], metric_name, str(label_config), i, file_type)
            # Todo: Check if 'str(label_config)' really does what you want
            #  (accumulated string of the label_config items).

        # Show the data in a figure (plot).
        if show_fig:
            plot_data(dat, metric_name, label_config, path)


def save_data(data, metric_name, label_config, i, file_type='csv'):
    """
    Save the data received.
    :param data: The data to save (one of the data-series received from the request).
    :param metric_name: Metric name.
    :param label_config: More details on the metric.
    :param i: The index of this time-series data (in received array).
    :param file_type: Type of file to save the data in (.csv, .xlsx).
    :return: True on success, False otherwise.
    """
    """ Save the data to the directory /metric_name/label_config_i.file_type"""
    df = MetricRangeDataFrame(data)

    metric_name = leg(metric_name)

    df.to_csv(f'../data/csv/{metric_name}_{str(label_config)}_{i}.{file_type}')

    df_with_dates = MetricRangeDataFrame(data)
    df_with_dates.index = pd.to_datetime(df_with_dates.index, unit="s", utc=True)
    df_with_dates.to_csv(f'../data/csv/dated/{metric_name}_{str(label_config)}_dates_{i}.{file_type}')




def plot_data(dat, metric_name, label_config=None, path=None):
    """
    Show figurs of the data received.
    :param dat: Array of all received data from the request.
    :param metric_name: Metric name.
    :param label_config: More details of the data.
    :param path: Save the figures in path 'path', if no path given - don't save.
    :return: Nothing (void).
    """
    for i in range(len(dat)):
        data = dat[i]
        xs = [(x[0]) for x in data['values']]
        ys = [int(x[1]) for x in data['values']]
        plt.ylabel('value')
        plt.xlabel('time[s]')
        plt.plot(xs, ys)
        if not (path is None):
            plt.savefig(path + f'_{i}.png')
        plt.show()

        # Conversion to real date-time:
        # m1_df = MetricRangeDataFrame(data[i])
        # m1_df.index = pd.to_datetime(m1_df.index, unit="s", utc=True)


def leg(s: str):
    ret = s
    while (ret.find(':')) >= 0:
        ind = ret.find(':')
        ret = ret[0:ind] + '_' + ret[ind + 1:len(ret)]
    return ret












