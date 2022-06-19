from Methods import methods, error_metrics, metrics
from data_gen.util import cpu_legal_pod_names, memory_legal_pod_names

granularities = ['s', 'min', '10min', '30min', 'h', 'D', 'W', 'M', 'Y']          # [second, hour, week, month, year]
compress_methods = ['mean', 'max', 'min', 'last', 'first', 'median', 'sum']

# from menu import metric_menu, application_menu, method_menu, gran_menu, cm_menu, error_metric_menu

def metric_menu():
    print(f'Please choose a metric:')
    for ind, metric in enumerate(metrics.keys()):
        print(f'{ind}) {metric}')
    choice = input('')
    return metrics[list(metrics.keys())[int(choice)]]


def application_menu(metric):
    print(f'Please choose an application:')
    if metric == 'pod:container_memory_usage_bytes:sum':
        for ind, name in enumerate(memory_legal_pod_names):
            print(f'{ind}) {name}')
        choice = input('')
        return memory_legal_pod_names[int(choice)]
    elif metric == 'pod:container_cpu_usage:sum':
        for ind, name in enumerate(cpu_legal_pod_names):
            print(f'{ind}) {name}')
        choice = input('')
        return cpu_legal_pod_names[int(choice)]
    else:
        raise Exception(f'No such metric: {metric}')


def method_menu():
    print(f'Please choose a method:')
    for ind, method in enumerate(methods.keys()):
        print(f'{ind}) {method}')
    choice = input('')
    return list(methods.keys())[int(choice)]


def gran_menu():
    print(f'Please choose a granularity:')
    for ind, gran in enumerate(granularities):
        print(f'{ind}) {gran}')
    choice = input('')
    return granularities[int(choice)]


def cm_menu():
    print(f'Please choose a compression method:')
    for ind, cm in enumerate(compress_methods):
        print(f'{ind}) {cm}')
    choice = input('')
    return compress_methods[int(choice)]


def error_metric_menu():
    print(f'Please choose an error metric:')
    error_metrics_keys = list(error_metrics.keys())
    for ind, em in enumerate(error_metrics_keys):
        print(f'{ind}) {em}')
    choice = input('')
    return error_metrics_keys[int(choice)]


def plot_one_menu():
    options = ['Yes', 'No']
    print(f'Plot one?')
    for ind, op in enumerate(options):
        print(f'{ind}) {op}')
    choice = input('')
    return options[int(choice)]










