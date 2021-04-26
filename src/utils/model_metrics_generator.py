import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, fbeta_score
import matplotlib.pyplot as plt
import time


class ModelMetricsGenerator:

    def __init__(self, test_data, test_truth):
        self._test_data = test_data
        self._test_truth = np.asarray(test_truth)
        self._predicted = 0
        self._fit_time = 0
        self._predict_time = 0

    def generate_metrics(self, model, train_data, train_truth):
        fit_start_time = time.time()
        model.fit(train_data, train_truth)
        self._fit_time = time.time() - fit_start_time


        predict_start_time = time.time()
        self._predicted = np.asarray(model.predict(self._test_data))
        self._predict_time = time.time() - predict_start_time

        return model

    def print_results(self):
        self.print_metrics()
        # self.print_confusion_matrix()

    def print_metrics(self):
        print('\n Indicadores rendimiento:')
        print(f'Fit time: {self.get_fit_time()}')
        print(f'Predict time: {self.get_predict_time()}')
        print(f'Accuracy: {self.get_accuracy_percentage()}')
        print(f'Precision: {self.get_precision_percentage()}')
        print(f'Recall: {self.get_recall_percentage()}')
        print(f'Specificity: {self.get_specifity_percentage()}')
        print(f'F2-score: {self.get_f2_score_percentage()}')
        print('\n')
        print(self.get_fit_time())
        print(self.get_predict_time())
        print(self.get_accuracy_percentage())
        print(self.get_precision_percentage())
        print(self.get_recall_percentage())
        print(self.get_specifity_percentage())
        print(self.get_f2_score_percentage())

    def get_fit_time(self):
        return round(self._fit_time, 4)

    def get_predict_time(self):
        return round(self._predict_time, 4)

    def get_accuracy_percentage(self):
        return round(accuracy_score(self._test_truth, self._predicted, normalize=True) * 100, 2)

    def get_precision_percentage(self):
        return round(precision_score(self._test_truth, self._predicted) * 100, 2)

    def get_recall_percentage(self):
        return round(recall_score(self._test_truth, self._predicted) * 100, 2)

    def get_specifity_percentage(self):
        return round(recall_score(self._test_truth, self._predicted, pos_label=0) * 100, 2)

    def get_f2_score_percentage(self):
        return round(fbeta_score(self._test_truth, self._predicted, beta=2) * 100, 2)
