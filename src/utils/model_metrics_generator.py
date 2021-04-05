import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, fbeta_score
import matplotlib.pyplot as plt
import time


class ModelMetricsGenerator:

    def __init__(self, test_truth):
        self._test_truth = np.asarray(test_truth)
        self._predicted = 0
        self._fit_time = 0
        self._predict_time = 0

    def generate_metrics(self, model, train_data, train_truth, test_data):
        fit_start_time = time.time()
        model.fit(train_data, train_truth)
        self._fit_time = time.time() - fit_start_time

        predict_start_time = time.time()
        self._predicted = np.asarray(model.predict(test_data))
        self._predict_time = time.time() - predict_start_time

        return model

    def print_results(self):
        self.print_metrics()
        # self.print_confusion_matrix()

    def print_metrics(self):
        self.print_fit_time()
        self.print_predict_time()
        self.print_accuracy()
        self.print_precision()
        self.print_recall()
        self.print_specifity()
        self.print_f2_score()

    def print_fit_time(self):
        fit_time_in_seconds = round(self._fit_time, 4)
        print(f'\nFit time: {fit_time_in_seconds}s.')

    def print_predict_time(self):
        predict_time_in_seconds = round(self._predict_time, 4)
        print(f'Predict time: {predict_time_in_seconds}s.')

    def print_accuracy(self):
        accuracy = round(accuracy_score(self._test_truth, self._predicted, normalize=True) * 100, 2)
        print(f'Accuracy: {accuracy}%.')

    def print_precision(self):
        precision = round(precision_score(self._test_truth, self._predicted) * 100, 2)
        print(f'Precision: {precision}%.')

    def print_recall(self):
        recall = round(recall_score(self._test_truth, self._predicted) * 100, 2)
        print(f'Recall: {recall}%.')

    def print_specifity(self):
        specifity = round(recall_score(self._test_truth, self._predicted, pos_label=0) * 100, 2)
        print(f'Specificity: {specifity}%.')

    def print_f2_score(self):
        f2_score = round(fbeta_score(self._test_truth, self._predicted, beta=2) * 100, 2)
        print(f'F2-score: {f2_score}%.')

    def print_confusion_matrix(self):
        print(f'Confusion matrix:\n {confusion_matrix(self._test_truth, self._predicted)}')
