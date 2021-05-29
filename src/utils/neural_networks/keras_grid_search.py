import pandas as pd
from sklearn.model_selection import  train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from src.utils.my_metrics import accuracy_precision_recall_specifity_f2_score
from utils.model_metrics_generator import ModelMetricsGenerator


def main():
    input_data = pd.read_excel('./../data/prepared/prepared_ICU_Prediction.xlsx')
    ground_truth = input_data['ICU']
    sample_data = input_data.drop('ICU', axis=1)
    train_data, test_data, train_truth, test_truth = train_test_split(sample_data, ground_truth, test_size=0.1,
                                                                      shuffle=True, random_state=42)
    num_features = train_data.shape[1]
    neurons = [16, 32, 64, 128]
    layers_number = [1, 2, 3, 4, 5]
    batch_size = [64, 128, 256, 512]
    epochs = [64, 256, 512, 768]
    param_grid = dict(neurons=neurons, layers_number=layers_number, batch_size=batch_size, epochs=epochs,
                      num_features=[num_features])

    grid_result = perform_grid_search(build_fn, param_grid, train_data, train_truth)

    # results
    print(f'El mejor resultado :{grid_result.best_score_} se consigue con {grid_result.best_params_}')

    # evaluate model against test data
    model_metrics_generator = ModelMetricsGenerator(grid_result, test_truth=test_truth)
    model_metrics_generator.predict_model(test_data)
    model_metrics_generator.print_results()


def perform_grid_search(build_function, param_grid, train_data, train_truth):
    """ Perform grid search

    :param param_grid: dictionary with hyperparameter description and the values that will be tested
        for getting the best performance.

    :return: a neural network model trained with the set of hyperparameter values that generates the best performance.
    """

    model = KerasClassifier(build_fn=build_function, verbose=0)
    sskfold = StratifiedShuffleSplit(random_state=1)
    scoring = accuracy_precision_recall_specifity_f2_score()
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=sskfold, scoring=scoring, refit='recall', n_jobs=-1)
    grid_result = grid.fit(train_data, train_truth)
    return grid_result

def build_fn(num_features, neurons=1, layers_number=1):
    """ Create a keras neural network model for hyperparameters grid search. It uses adam optimizer, binary
        crossentropy as loss function, and accuracy and recall as metrics.

        :param num_features
        :param neurons: indicates how many neurons will have each hidden layer.
        :param layers_number: number of hidden layers that will have the neural network.

        :return a compiled neural network model.
    """

    model = keras.Sequential([layers.Dense(units=neurons, activation='relu', input_dim=num_features)])
    for i in range(layers_number):
        model.add(layers.Dense(units=neurons, activation='relu', input_dim=num_features))
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'Recall']
    )

    return model


if __name__ == '__main__':
        main()
