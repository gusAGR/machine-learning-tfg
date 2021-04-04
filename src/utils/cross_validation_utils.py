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
                'fit_time': 'Fit time',
                'score_time': 'Test time',
                'test_accuracy': 'Accuracy',
                'test_precision': 'Precision',
                'test_recall': 'Recall',
                'test_specifity': 'Specifity',
                'test_f2_score': 'F2 score'
            }
        else:
            self.descriptions = descriptions

    def print_metrics(self, metrics):
        """ Print metric names and its mean values obatined by a cross_validation procedure

        :param metrics: dictionary of metrics values

        """
        print(f'\nValores medios:')
        for metric, values in metrics.items():
            metric_description = self.descriptions.get(metric, 'Métrica sin definir')
            mean = np.asarray(values).mean()
            print(f'\t{metric_description}  : {mean}.')


