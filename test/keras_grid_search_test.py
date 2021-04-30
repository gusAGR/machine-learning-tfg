import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.model_metrics_generator import ModelMetricsGenerator
from src.utils.neural_networks import keras_grid_search as kgs


class MyTestCase(unittest.TestCase):

    def test_grid_search(self):
        input_data = pd.read_excel('./../data/prepared/prepared_ICU_Prediction.xlsx')
        ground_truth = input_data['ICU']
        sample_data = input_data.drop('ICU', axis=1)
        train_data, test_data, train_truth, test_truth = train_test_split(sample_data, ground_truth, test_size=0.1, shuffle=True, random_state=42)
        num_features = train_data.shape[1]
        neurons = [16, 32, 64, 128]
        layers_number = [1, 2, 3, 4, 5]
        batch_size = [64, 128, 256, 512]
        epochs = [64, 256, 512, 768]

        param_grid = dict(neurons=neurons, layers_number=layers_number, batch_size=batch_size, epochs=epochs, num_features=[num_features])

        grid_result = kgs.perform_grid_search(kgs.build_fn, param_grid, train_data, train_truth)

        # results
        print(f'El mejor resultado :{grid_result.best_score_} se consigue con {grid_result.best_params_}')

        # evaluate model against test data
        model_metrics_generator = ModelMetricsGenerator(grid_result, test_truth=test_truth)
        model_metrics_generator.predict_model(test_data)
        model_metrics_generator.print_results()


if __name__ == '__main__':
    unittest.main()
