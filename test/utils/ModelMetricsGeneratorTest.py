import unittest
from unittest import TestCase
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from src.utils.model_metrics_generator import ModelMetricsGenerator


class TestModelMetricsGenerator(TestCase):

    def setUp(self):
        pass

    def test_output(self):
        train_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        train_truth = pd.DataFrame({'truth': [0, 1, 0]})
        test_data = pd.DataFrame({'feature1': [1, 4, 3], 'feature2': [4, 9, 6]})
        test_truth = pd.DataFrame({'truth': [0, 1, 0]})

        model = svm.LinearSVC()
        metric_generator = ModelMetricsGenerator(test_data, test_truth)
        metric_generator.generate_metrics(model, train_data, train_truth)

        metric_generator.print_results()

    def test_output_with_real_dataset(self):
        input_data = pd.read_excel('./../../data/prepared/prepared_ICU_Prediction.xlsx')
        ground_truth = input_data['ICU']
        sample_data = input_data.drop('ICU', axis=1)
        train_data, test_data, train_truth, test_truth = train_test_split(sample_data, ground_truth, test_size=0.2, shuffle=True)

        model = svm.LinearSVC()
        metric_generator = ModelMetricsGenerator(test_data, test_truth)
        metric_generator.generate_metrics(model, train_data, train_truth)

        metric_generator.print_results()

    if __name__ == '__main__':
        unittest.main()
