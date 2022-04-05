from prometheus_api_client import PrometheusConnect, PrometheusApiClientException
from typing import Sequence
from pandas import DataFrame
import datetime


class ThanosConnect(PrometheusConnect):
    def __init__(self, url, op_token):
        super().__init__(url=url,
                         headers={"Authorization": f"Bearer {op_token}"},
                         disable_ssl=False)

    @staticmethod
    def build_query(labels_config: dict, metric_name: str = ''):
        if labels_config:
            label_list = [f'{key}="{val}"' for key, val in labels_config.items()]
            return f'{metric_name}{{{",".join(label_list)}}}'
        else:
            return ''

    """
    :returns a query result over the selected metrics based on the given params - a list of metrics 
    :arguments:
    start_time - Start time for the range of the query over the selected metrics
    end_time - End time for the range of the query over the selected metrics
    step - The steps in time between each sample to align the range according to the step's resolution.
    metric_name - The metrics that we want to get from our sensor and we query upon. Can have more than one metric value 
    label_config - the labels of the metrics we query upon. (Additional parameters for filtering the query) 
    :reference https://prometheus.io/docs/prometheus/latest/querying/api/#range-queries
    """

    def range_query(self, start_time: datetime, end_time: datetime, step: int, metric_name='',
                    label_config: dict = None):
        query = self.build_query(label_config, metric_name)
        metric_data = self.custom_query_range(query=query,
                                              start_time=start_time,
                                              end_time=end_time,
                                              step=str(step))
        return metric_data

    """
    :returns all possible values of for a given label
    :reference: https://prometheus.io/docs/prometheus/latest/querying/api/#querying-label-values 
    """

    def query_label_values(self, label: str, cluster: str = None):
        if cluster:
            match = {'match[]': self.build_query({'cluster': cluster})}
        else:
            match = None
        response = self._session.get(
            '{url}/api/v1/label/{label}/values'.format(url=self.url, label=label),
            params=match,
            verify=self.ssl_verification,
            headers=self.headers,
        )
        if response.status_code == 200:
            data = response.json()["data"]
        else:
            raise PrometheusApiClientException(
                "HTTP Status Code {} ({!r})".format(response.status_code, response.content)
            )

        return data

    @staticmethod
    def metric_data_to_df(data):
        def _get_values(metric: dict):
            return metric["values"] if "values" in metric else [metric["value"]]

        if data is not None:
            # if just a single json instead of list/set/other sequence of jsons,
            # treat as list with single entry
            if not isinstance(data, Sequence):
                data = [data]

            # unpack metric
            unpack = [{**i["metric"], **{"timestamp": val[0], "value": val[1]}}
                      for i in data for val in _get_values(i)]
            return DataFrame(unpack)
