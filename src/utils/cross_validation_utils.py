
def print_metrics(metrics):
    """ Print metric names and its mean values obatined by a cross_validation procedure
    
    :param metrics: dictionary of metrics
    """
    print('\n')
    for metric, values in metrics.items():
        mean = values.mean()
        print(f'{metric} mean value: {mean}')


