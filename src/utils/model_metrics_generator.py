import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score
import time


class ModelMetricsGenerator:
    """ Encapsulate model training and evaluation and allows to obtain the values of the defined metrics for assessing model performance.

    """

    def __init__(self, model, test_truth=None):
        """ Constructor

        :param model:  model that it will generate metrics measures.
        :param test_truth: ground truth of test dataset.
        """

        self._test_truth = test_truth
        self._model = model
        self.predicted = 0
        self._fit_time = 0
        self._predict_time = 0

    def fit_and_predict_model(self, train_data, train_truth, test_data):
        self.fit_model(train_data, train_truth)
        self.predict_model(test_data)

        return self._model

    def fit_model(self, train_data, train_truth):
        fit_start_time = time.time()
        self._model.fit(train_data, train_truth)
        self._fit_time = time.time() - fit_start_time

        return self._model

    def predict_model(self, test_data):
        predict_start_time = time.time()
        self.predicted = np.asarray(self._model.predict(test_data))
        self._predict_time = time.time() - predict_start_time

        return self._model

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
        return round(accuracy_score(self._test_truth, self.predicted, normalize=True) * 100, 2)

    def get_precision_percentage(self):
        return round(precision_score(self._test_truth, self.predicted) * 100, 2)

    def get_recall_percentage(self):
        return round(recall_score(self._test_truth, self.predicted) * 100, 2)

    def get_specifity_percentage(self):
        return round(recall_score(self._test_truth, self.predicted, pos_label=0) * 100, 2)

    def get_f2_score_percentage(self):
        return round(fbeta_score(self._test_truth, self.predicted, beta=2) * 100, 2)
