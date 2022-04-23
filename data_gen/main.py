from prometheus_api_client.utils import parse_datetime
from Thanos_conn import ThanosConnect, leg, current_time_for_file

# token-gen url:  https://www.operate-first.cloud/apps/content/observatorium/thanos/thanos_programmatic_access.html
OPERATE_FIRST_TOKEN = "sha256~OkKamOMHXx4vYe3Rp4EfsWKZSXGVh8gmChGmsaU3I3M"
THANOS_URL = "https://thanos-query-frontend-opf-observatorium.apps.smaug.na.operate-first.cloud"

# Run a simple request, for fetching a data-series, plotting it and saving it in data/metric/label_conf.csv
conn = ThanosConnect(THANOS_URL, OPERATE_FIRST_TOKEN)
# Now get some data, and see that you can save\show it nicely. --> Work on.

metric_label = 'NooBaa_Endpoint_process_cpu_seconds_total'
date_time_for_file = current_time_for_file()
data = conn.get_data(metric_label,
                     start_time=parse_datetime("1d"),
                     end_time=parse_datetime("now"),
                     date_time_for_file=date_time_for_file,
                     label_config={'cluster':"moc/smaug", 'endpoint':"metrics", 'instance':"10.129.0.40:7004", 'job':"s3", 'namespace':"openshift-storage", 'pod':"noobaa-endpoint-84799dddf5-j6f6m", 'prometheus':"openshift-monitoring/k8s", 'service': "s3", 'tenant_id': "moc-smaug"},
                     path=f'../data/png/{leg(metric_label)}_{date_time_for_file}',
                     show_fig=False
                     )


