from tensorflow import keras
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from src.utils.my_metrics import accuracy_precision_recall_specifity_f2_score


def build_fn(num_features, neurons=1, layers_number=1):
    """ Create a keras neural network model for hyperparameters grid search

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


def perform_grid_search(build_function, param_grid, train_data, train_truth):
    model = KerasClassifier(build_fn=build_function, verbose=0)
    sskfold = StratifiedShuffleSplit(random_state=1)
    scoring = accuracy_precision_recall_specifity_f2_score()
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=sskfold, scoring=scoring, refit='f2_score', n_jobs=4)
    grid_result = grid.fit(train_data, train_truth)
    return grid_result
