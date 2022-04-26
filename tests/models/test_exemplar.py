"""Module for testing exemplar.py.

    Test whether exemplar model learns the correct predictions.
"""

import pytest
import numpy as np

import semisupervised.models.exemplar as exemplar_model

# TODO test in d > 1, k > 2, soft labels

@pytest.fixture
def params_1():
    params = {
        'k': [2, 2],
        'd': [1, 1],
        'c': [1, 1],
        'w_ul': [1, 0.1],
        'pseudo_labels': ['hard', 'hard'],
        'rnd_if_eq': ['fixed', 'fixed']
    }
    return params


@pytest.fixture
def data_1():
    xs = np.array([-2, 2, -1, 0.25])
    ys = np.array([0, 1, 0, -1])

    data = {
        'xs': xs,
        'ys': ys,
        'test': [-2, -1, 0, 1, 2],
        'truth': [[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]]
    }
    return data


def test_exemplar_model(params_1, data_1):

    for i in range(len(params_1['c'])):
        # Initialize model.
        model = exemplar_model.StandardExemplar(k=params_1['k'][i], d=params_1['d'][i], c=params_1['c'][i], w_ul=params_1['w_ul'][i],
                                                pseudo_labels=params_1['pseudo_labels'][i], rnd_if_eq=params_1['rnd_if_eq'][i])

        # Train models.
        model.train(np.array([data_1['xs']]), data_1['ys'])

        predictions = model.prediction_multiple(np.array([data_1['test']]))
        print(predictions)

        np.testing.assert_array_equal(data_1['truth'][i], np.round(predictions[1, :]))

