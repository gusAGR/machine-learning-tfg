import unittest
from unittest import TestCase
import pandas as pd
from src.utils.cross_validation_utils import CrossValidationMetricsResultPrinter


class TestModelMetricsGenerator(TestCase):


    def test_print_metrics(self):
        results = {'test_accuracy': [1, 2, 3]}
        printer = CrossValidationMetricsResultPrinter()
        printer.print_metrics(results)

    if __name__ == '__main__':
        unittest.main()



