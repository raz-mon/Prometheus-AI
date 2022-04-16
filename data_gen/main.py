from Thanos_conn import ThanosConnect, save_data, plot_data

# token-gen url:  https://www.operate-first.cloud/apps/content/observatorium/thanos/thanos_programmatic_access.html
OPERATE_FIRST_TOKEN = "sha256~Y1nh6z8Gb_7cbcYMMGp9MMjPcduAwpOABDmUTuVU7rY"
THANOS_URL = "https://thanos-query-frontend-opf-observatorium.apps.smaug.na.operate-first.cloud"

# Run a simple request, for fetching a data-series, plotting it and saving it in data/metric/label_conf.csv
conn = ThanosConnect(THANOS_URL, OPERATE_FIRST_TOKEN)
# Now get some data, and see that you can save\show it nicely. --> Work on.