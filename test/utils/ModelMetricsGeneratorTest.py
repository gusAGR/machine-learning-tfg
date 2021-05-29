import unittest
from unittest import TestCase
import pandas as pd
from sklearn import svm
from src.utils.model_metrics_generator import ModelMetricsGenerator


class TestModelMetricsGenerator(TestCase):

    def test_should_generate_metrics(self):
        train_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        train_truth = pd.DataFrame({'truth': [0, 1, 0]})
        test_data = pd.DataFrame({'feature1': [1, 4, 3], 'feature2': [4, 9, 6]})
        test_truth = pd.DataFrame({'truth': [0, 1, 0]})

        model = svm.LinearSVC()
        metric_generator = ModelMetricsGenerator(model, test_truth)
        metric_generator.fit_and_predict_model(train_data, train_truth, test_data)

        self.assertTrue(len(metric_generator.predicted) > 0)
        self.assertTrue(metric_generator.get_accuracy_percentage() > 0)

    if __name__ == '__main__':
        unittest.main()
