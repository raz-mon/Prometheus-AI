from prometheus_api_client.utils import parse_datetime
from Thanos_conn import ThanosConnect, leg, current_time_for_file, good_result
import numpy as np
import pandas as pd

# token-gen url:  https://www.operate-first.cloud/apps/content/observatorium/thanos/thanos_programmatic_access.html
OPERATE_FIRST_TOKEN = "sha256~jrKn3bjXy5pa-R3wjraS81zJBHFoe9aFKAaj7wTX5Kc"
THANOS_URL = "https://thanos-query-frontend-opf-observatorium.apps.smaug.na.operate-first.cloud"

# Run a simple request, for fetching a data-series, plotting it and saving it in data/metric/label_conf.csv
conn = ThanosConnect(THANOS_URL, OPERATE_FIRST_TOKEN)

# Now get some data, and see that you can save\show it nicely. --> Work on.
date_time_for_file = current_time_for_file()
time_back = "28h"


metric_name = 'pod:container_memory_usage_bytes:sum'
data = conn.get_data(metric_name,
                     start_time=parse_datetime(time_back),
                     end_time=parse_datetime("now"),
                     date_time_for_file=date_time_for_file,
                     time_back=time_back,
                     # label_config= {'pod': },
                     # path=f'../data/png/{leg(metric_name)}_{time_back}_{date_time_for_file}',
                     show_fig=False
                     )
print(f'succeeded to download memory results, moving on to cpu...')

metric_name = 'pod:container_cpu_usage:sum'

data2 = conn.get_data(metric_name,
                      start_time=parse_datetime(time_back),
                      end_time=parse_datetime("now"),
                      date_time_for_file=date_time_for_file,
                      time_back=time_back,
                      # label_config= {'pod': },
                      # path=f'../data/png/{leg(metric_name)}_{time_back}_{date_time_for_file}',
                      show_fig=False
                      )













"""
# Check that filtering really occures.

def get_filtered_pod_names():
    metric_name = 'pod:container_cpu_usage:sum'
    dat = conn.get_metric_range_data(metric_name,
                                     # label_config=label_config,
                                     start_time=parse_datetime("28h"),
                                     end_time=parse_datetime("now"),
                                     )
    print(f'Got {len(dat)} results.')

    pods = []
    for res in dat:
        if good_result(res['metric']['pod']):
            pods += [res['metric']['pod']]
    print(f'len(filtered pods): {len(pods)}')
    # print(pods)

    df = pd.DataFrame(np.array(pods))
    df.to_csv('pod_names/pod_names_cpu_usage_filtered.csv')

get_filtered_pod_names()

def get_pod_names():
    metric_name = 'pod:container_cpu_usage:sum'
    dat = conn.get_metric_range_data(metric_name,
                                     # label_config=label_config,
                                     start_time=parse_datetime("28h"),
                                     end_time=parse_datetime("now"),
                                     )
    print(f'Got {len(dat)} results.')

    pods = []
    for res in dat:
        pods += [res['metric']['pod']]
    print(f'len(unfiltered pods): {len(pods)}')
    # print(pods)

    df = pd.DataFrame(np.array(pods))
    df.to_csv('pod_names/pod_names_cpu_usage_not_filtered.csv')

get_pod_names()
"""















