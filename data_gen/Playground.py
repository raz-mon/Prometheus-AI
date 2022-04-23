from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
from prometheus_api_client.utils import parse_datetime
import pandas as pd
from main import OPERATE_FIRST_TOKEN, THANOS_URL
from old.thanos_api_client import ThanosConnect
import matplotlib.pyplot as plt
from prometheus_api_client import PrometheusApiClientException
from datetime import timedelta
import datetime
# from old.preprocess_thanos import export_to_csv, show_graph


# Todo: Find some data-series that are continuous as can be, and write them down, so can talk to Ilya about which we
#  actually want.

# Todo: Add visualization options as proposed in
#  https://github.com/aicoe-aiops/time-series/blob/master/notebooks/ts-2-visualization.ipynb.

# Todo: Should probably add resampling. See link above for nice resampling scheme and implementations (quite amazing).

than = ThanosConnect(THANOS_URL, OPERATE_FIRST_TOKEN)

dat = than.get_metric_range_data('node_memory_Active_bytes',
                                 # label_config={'cluster': 'moc/smaug'},
                                 start_time=parse_datetime("50h"),
                                 end_time=parse_datetime("now")
                                 )
                                 # Other optional ways:
                                 # start_time= (datetime.now() - datetime.timedelta(hours=8)),
                                 # end_time=datetime.now()































"""
print(f'Got {len(dat)} results!')
print(dat)
print(dat[0])
print((dat[0])['values'])

metric1 = dat[0]                        # This is only the first metric!!! Out of many!
# x0 = metric1['values'][0][0]
# print(x0)
# xs = [(x[0]-x0) for x in metric1['values']]
xs = [(x[0]) for x in metric1['values']]
ys = [int(x[1]) for x in metric1['values']]

print(xs)
print(ys)
plt.ylabel('value')
plt.xlabel('time[s]')
plt.plot(xs, ys)
# plt.savefig('data/test1/9.png')
plt.show()

m1_df = MetricRangeDataFrame(metric1)
m1_df.index = pd.to_datetime(m1_df.index, unit="s", utc=True)


# m1_df.to_csv('t1.csv')
# m1_df.to_excel('t1.xlsx')
"""

# print(prom.all_metrics())

# print(prom.custom_query(query="prometheus_http_requests_total"))

"""
start_time = parse_datetime("2d")
end_time = parse_datetime("now")
chunk_size = timedelta(days=1)

metric_data = prom.get_metric_range_data(
    "up{cluster='moc/smaug'}",  # this is the metric name and label config
    start_time=start_time,
    end_time=end_time,
    chunk_size=chunk_size,
)

df = ThanosConnect.metric_data_to_df(metric_data)
df['date'] = pd.to_datetime(df['timestamp'], origin='unix', unit='s')
export_to_csv(df, "test1.csv", datetime.strptime('2022-03-21 10:00:00', "%Y-%m-%d %H:%M:%S"))
# show graph
show_graph(df)
# print(metric_data)
"""


