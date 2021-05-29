import unittest
from unittest import TestCase
from src.utils.cross_validation_utils import CrossValidationMetricsResultPrinter
from unittest.mock import patch


class TestModelMetricsGenerator(TestCase):

    @patch('builtins.print')
    def test_should_print_metric_values(self, mock_print):
        results = {'test_accuracy': [1, 2, 3], 'fit_time': [0.00212, 0.34455, 2.3234]}

        printer = CrossValidationMetricsResultPrinter()
        printer.print_metrics_values(results)

        self.assertTrue(mock_print.called)

    @patch('builtins.print')
    def test_should_print_metrics_report(self, mock_print):
        results = {'test_accuracy': [1, 2, 3], 'fit_time': [0.00212, 0.34455, 2.3234]}

        printer = CrossValidationMetricsResultPrinter()
        printer.print_metrics_report(results)

        self.assertTrue(mock_print.called)

    if __name__ == '__main__':
        unittest.main()



