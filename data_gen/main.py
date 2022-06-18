from prometheus_api_client.utils import parse_datetime
from data_gen.Thanos_conn import ThanosConnect, leg, current_time_for_file, good_result
import pandas as pd
import numpy as np

# token-gen url:  https://www.operate-first.cloud/apps/content/observatorium/thanos/thanos_programmatic_access.html
OPERATE_FIRST_TOKEN = "sha256~gmImH6zcn7BSEekNP08L9ns3ytbqInfNnp9R4zqJoME"
THANOS_URL = "https://thanos-query-frontend-opf-observatorium.apps.smaug.na.operate-first.cloud"


def main():
    # Initialize connection Object.
    conn = ThanosConnect(THANOS_URL, OPERATE_FIRST_TOKEN)
    date_time_for_file = current_time_for_file()                # Current time in datetime protocol.
    time_back = "28h"                                           # How much time back does the query go.


    # Memory query
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

    print(f'\n---------------------------------------------------------------------------------\n'
          f'Succeeded to download memory results, moving on to cpu...\n'
          f'---------------------------------------------------------------------------------\n')


    # Cpu-usage query
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


if __name__ == '__main__':
    main()


















"""
# Check that filtering really occures.
def get_filtered_pod_names(metric, filtered):
    conn = ThanosConnect(THANOS_URL, OPERATE_FIRST_TOKEN)
    dat = conn.get_metric_range_data(metric,
                                     # label_config=label_config,
                                     start_time=parse_datetime("28h"),
                                     end_time=parse_datetime("now"),
                                     )
    print(f'Got {len(dat)} results.')

    pods = []
    for res in dat:
        if good_result(res['metric']['pod']) and filtered:
            pods += [res['metric']['pod']]
    print(f'len(after filtering): {len(pods)}')
    # print(pods)

    df = pd.DataFrame(np.array(pods))
    if filtered:
        df.to_csv(f'pod_names/pod_names_{metric[4:-4]}_filtered.csv')
    else:
        df.to_csv(f'pod_names/pod_names_{metric[4:-4]}_not-filtered.csv')


# get_filtered_pod_names('pod:container_memory_usage_bytes:sum', True)
# get_filtered_pod_names('pod:container_memory_usage_bytes:sum', False)
# get_filtered_pod_names('pod:container_cpu_usage:sum', True)
get_filtered_pod_names('pod:container_cpu_usage:sum', False)
"""













