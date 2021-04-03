import unittest
from unittest import TestCase
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from src.utils.my_metrics import accuracy_precision_recall_specifity_f2_score

from src.utils.cross_validation_utils import print_metrics


class TestModelMetricsGenerator(TestCase):


    def test_output_with_real_dataset(self):
        input_data = pd.read_excel('./../../data/prepared/prepared_ICU_Prediction.xlsx')

        ground_truth = input_data['ICU']
        sample_data = input_data.drop('ICU', axis=1)

        gnb_model = GaussianNB()
        applied_metrics = accuracy_precision_recall_specifity_f2_score()

        result = cross_validate(gnb_model, sample_data, ground_truth, cv=4, scoring=applied_metrics)
        print_metrics(result)

    if __name__ == '__main__':
        unittest.main()



