from prometheus_api_client.utils import parse_datetime
from Thanos_conn import ThanosConnect, leg, current_time_for_file

# token-gen url:  https://www.operate-first.cloud/apps/content/observatorium/thanos/thanos_programmatic_access.html
OPERATE_FIRST_TOKEN = "sha256~wdiYIuz2E8MFvmJnG3WEVO0gfQFvMA9Avqs2L7V3Geo"
THANOS_URL = "https://thanos-query-frontend-opf-observatorium.apps.smaug.na.operate-first.cloud"

# Run a simple request, for fetching a data-series, plotting it and saving it in data/metric/label_conf.csv
conn = ThanosConnect(THANOS_URL, OPERATE_FIRST_TOKEN)
# Now get some data, and see that you can save\show it nicely. --> Work on.

metric_label = 'pod:container_cpu_usage:sum'
date_time_for_file = current_time_for_file()
data = conn.get_data(metric_label,
                     start_time=parse_datetime("1d"),
                     end_time=parse_datetime("now"),
                     date_time_for_file=date_time_for_file,
                     label_config={'cluster':"emea/balrog", 'namespace':"open-cluster-management-agent", 'pod':"klusterlet-work-agent-5c866c8d9b-6pzxb", 'prometheus':"openshift-monitoring/k8s", 'tenant_id':"emea-balrog"},
                     path=f'../data/png/{leg(metric_label)}_{date_time_for_file}',
                     show_fig=False
                     )


