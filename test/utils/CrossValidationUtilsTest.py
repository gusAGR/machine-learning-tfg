import unittest
from unittest import TestCase
import pandas as pd
from src.utils.cross_validation_utils import CrossValidationMetricsResultPrinter


class TestModelMetricsGenerator(TestCase):


    def test_print_metrics_values(self):
        results = {'test_accuracy': [1, 2, 3], 'fit_time': [0.00212, 0.34455, 2.3234]}
        printer = CrossValidationMetricsResultPrinter()
        printer.print_metrics_values(results)

    def test_print_metrics_report(self):
        results = {'test_accuracy': [1, 2, 3], 'fit_time': [0.00212, 0.34455, 2.3234]}
        printer = CrossValidationMetricsResultPrinter()
        printer.print_metrics_report(results)

    if __name__ == '__main__':
        unittest.main()



