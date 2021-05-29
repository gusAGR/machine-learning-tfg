import numpy as np


class CrossValidationMetricsResultPrinter:
    """ Prints mean values of metrics obtained by a cross validation process.

    """

    def print_metrics_report(self, metrics):
        """ Print metric names and its mean values obatined by a cross_validation procedure.

        :param metrics: dictionary of metrics values.
        """
        print(f'\nValores medios:')
        for key, values in metrics.items():
            if self._is_time_metric(key):
                metric_mean_percentage = round(np.asarray(values).mean(), 4)
            else:
                metric_mean_percentage = round(np.asarray(values).mean() * 100, 2)
            print(f'{key}: {metric_mean_percentage}.')

        self.print_metrics_values(metrics)

    def _is_time_metric(self, metric_description):
        return metric_description.find('time') != -1

    def print_metrics_values(self, metrics):
        """ Print metric mean values obatined by a cross_validation procedure.

        :param metrics: dictionary with the values for each metric.
        """

        print('\n')
        for metric, values in metrics.items():
            if self._is_time_metric(metric):
                metric_mean_percentage = round(np.asarray(values).mean(), 4)
            else:
                metric_mean_percentage = round(np.asarray(values).mean() * 100, 2)
            print(metric_mean_percentage)
