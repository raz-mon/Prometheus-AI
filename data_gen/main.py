from prometheus_api_client.utils import parse_datetime

from Thanos_conn import ThanosConnect, leg

# token-gen url:  https://www.operate-first.cloud/apps/content/observatorium/thanos/thanos_programmatic_access.html
OPERATE_FIRST_TOKEN = "sha256~lYeTfpFnPW8yxoCHiIm_4T7KDJH-p_UKXtEM8UpMymE"
THANOS_URL = "https://thanos-query-frontend-opf-observatorium.apps.smaug.na.operate-first.cloud"

# Run a simple request, for fetching a data-series, plotting it and saving it in data/metric/label_conf.csv
conn = ThanosConnect(THANOS_URL, OPERATE_FIRST_TOKEN)
# Now get some data, and see that you can save\show it nicely. --> Work on.

metric_label = 'cluster:capacity_cpu_cores:sum'

conn.get_data(metric_label,
              start_time=parse_datetime("1w"),
              end_time=parse_datetime("now"),
              path=f'../data/png/{leg(metric_label)}.png',
              show_fig=True
              )