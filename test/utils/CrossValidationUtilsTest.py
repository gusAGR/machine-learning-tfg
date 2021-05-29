import unittest
from unittest import TestCase
from src.utils.cross_validation_utils import CrossValidationMetricsResultPrinter
from unittest.mock import Mock


class TestModelMetricsGenerator(TestCase):

    def test_print_metrics_values(self, mock):
        results = {'test_accuracy': [1, 2, 3], 'fit_time': [0.00212, 0.34455, 2.3234]}
        printer = CrossValidationMetricsResultPrinter()
        printer.print_metrics_values(results)

        self.assertTrue(mock.get_value())

    def test_print_metrics_report(self):
        results = {'test_accuracy': [1, 2, 3], 'fit_time': [0.00212, 0.34455, 2.3234]}
        printer = CrossValidationMetricsResultPrinter()
        printer.print_metrics_report(results)

    if __name__ == '__main__':
        unittest.main()



