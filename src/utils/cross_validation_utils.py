import numpy as np


class CrossValidationMetricsResultPrinter:
    """ Prints mean values of metrics obtained by a cross validation process.

    """

    def __init__(self, descriptions=None):
        """

        :param descriptions: Dictionary with the desired descriptions that will be showed for each metric.
        """
        if descriptions is None:
            self.descriptions = {
                'fit_time': 'Fit time: {0}s.',
                'score_time': 'Test time: {0}s',
                'test_accuracy': 'Accuracy: {0}%.',
                'test_precision': 'Precision: {0}%.',
                'test_recall': 'Recall: {0}%.',
                'test_specifity': 'Specificity: {0}%.',
                'test_f2_score': 'F2 score: {0}%.'
            }
        else:
            self.descriptions = descriptions

    def print_metrics_report(self, metrics):
        """ Print metric names and its mean values obatined by a cross_validation procedure

        :param metrics: dictionary of metrics values
        """
        print(f'\nValores medios:')
        for metric, values in metrics.items():
            metric_description = self.descriptions.get(metric, 'Métrica sin definir: {0}.')
            mean = None
            if self._is_time_metric(metric_description):
                mean = round(np.asarray(values).mean(), 4)
            else:
                mean = round(np.asarray(values).mean() * 100, 2)
            print('\t' + metric_description.format(mean))

        self.print_metrics_values(metrics)

    def _is_time_metric(self, metric_description):
        return metric_description.find('time') != -1

    def print_metrics_values(self, metrics):
        """ Print metric mean values obatined by a cross_validation procedure

        :param metrics: dictionary of metrics values
        """
        print('\n')
        for metric, values in metrics.items():
            if self._is_time_metric(metric):
                mean = round(np.asarray(values).mean(), 4)
            else:
                mean = round(np.asarray(values).mean() * 100, 2)
            print(mean)
