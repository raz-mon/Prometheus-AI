from prometheus_api_client.utils import parse_datetime
from Thanos_conn import ThanosConnect, leg, current_time_for_file
import numpy as np
import pandas as pd

# token-gen url:  https://www.operate-first.cloud/apps/content/observatorium/thanos/thanos_programmatic_access.html
OPERATE_FIRST_TOKEN = "sha256~fyf0kwJxEVe73PbB_jcsrKnNK7gzvrxb7g3YVN6tsBY"
THANOS_URL = "https://thanos-query-frontend-opf-observatorium.apps.smaug.na.operate-first.cloud"

# Run a simple request, for fetching a data-series, plotting it and saving it in data/metric/label_conf.csv
conn = ThanosConnect(THANOS_URL, OPERATE_FIRST_TOKEN)
# Now get some data, and see that you can save\show it nicely. --> Work on.

metric_name = 'pod:container_memory_usage_bytes:sum'
# metric_name = 'pod:container_cpu_usage:sum'

date_time_for_file = current_time_for_file()
data = conn.get_data(metric_name,
                     start_time=parse_datetime("2d"),
                     end_time=parse_datetime("now"),
                     date_time_for_file=date_time_for_file,
                     # label_config= ,
                     # path=f'../data/png/{leg(metric_name)}_{date_time_for_file}',
                     show_fig=False
                     )


"""
dat = conn.get_metric_range_data(metric_name,
                                 # label_config=label_config,
                                 start_time = parse_datetime("12h"),
                                 end_time = parse_datetime("now"),
                                 )
print(f'Got {len(dat)} results.')

pods = []
for res in dat:
    pods += [res['metric']['pod']]
print(f'len(pods): {len(pods)}')
print(pods)

df = pd.DataFrame(np.array(pods))
df.to_csv('pod_names_cpu_usage.csv')
"""





















