"""Module for testing prototype.py.

    Test whether prototype model learns the correct prototypes.
"""

import pytest
import numpy as np

import semisupervised.models.prototype as mp

# TODO: test soft labels

@pytest.fixture
def params():
    d = 2
    params = {
        'k': 3,
        'd': d,
        'r': 1,
        'c': 1,
        'w': np.ones(d) / np.sum(np.ones(d)),
        'w_ul': 1,
        'pseudo_labels': "hard"
    }
    return params


@pytest.fixture
def xs_sl():
    xs_sl = {
        'xs_1': np.array([[1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1]]),
        'xs_2': np.array([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]])
    }
    return xs_sl


@pytest.fixture
def xs_ssl():
    xs_ssl = {
        'xs_1': np.array([[1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0]]),
        'xs_2': np.array([[1, 1, 0, 1, 0, 1, 1, 0], [1, 1, 0, 1, 0, 1, 1, 0]])
    }
    return xs_ssl


@pytest.fixture
def ys_sl():
    ys_sl = {
        'ys_1': np.array([1, 0, 1, 0, 1, 0, 2], dtype=int),
        'ys_2': np.array([0, 0, 0, 1, 1, 1, 2], dtype=int)
    }
    return ys_sl

@pytest.fixture
def ys_ssl():
    ys_ssl = {
        'ys_1': np.array([1, 1, 1, 0, 0, -1, -1, 2], dtype=int),
        'ys_2': np.array([1, 0, 0, 0, 0, -1, -1, 2], dtype=int)
    }
    return ys_ssl


@pytest.fixture
def truth_sl():
    truth_sl = {
        'xs_1_ys_1': np.array([[0, 1, 1], [0, 1, 1]]),
        'xs_2_ys_1': np.array([[1/3, 2/3, 0], [1/3, 2/3, 0]]),
        'xs_1_ys_2': np.array([[2/3, 1/3, 1], [2/3, 1/3, 1]]),
        'xs_2_ys_2': np.array([[1, 0, 0], [1, 0, 0]])
    }
    return truth_sl


@pytest.fixture
# in case of hard labels only
def truth_ssl():
    truth_ssl = {
        'xs_1_ys_1': np.array([[3/4, 1/3, 0], [3/4, 1/3, 0]]),
        'xs_2_ys_1': np.array([[1/2, 4/5, 0], [1/2, 4/5, 0]]),
        'xs_1_ys_2': np.array([[1/4, 1, 0], [1/4, 1, 0]]),
        'xs_2_ys_2': np.array([[0.5, 1, 0], [0.5, 1, 0]])
    }
    return truth_ssl


def test_prototype_sl(params, xs_sl, ys_sl, truth_sl):

    # Initialize models.
    model_1_1 = mp.StandardPrototype(k=params['k'], d=params['d'], r=params['r'], c=params['c'], w=params['w'],
                                     w_ul=params['w_ul'], pseudo_labels=params['pseudo_labels'])
    model_1_2 = mp.StandardPrototype(k=params['k'], d=params['d'], r=params['r'], c=params['c'], w=params['w'],
                                     w_ul=params['w_ul'], pseudo_labels=params['pseudo_labels'])
    model_2_1 = mp.StandardPrototype(k=params['k'], d=params['d'], r=params['r'], c=params['c'], w=params['w'],
                                     w_ul=params['w_ul'], pseudo_labels=params['pseudo_labels'])
    model_2_2 = mp.StandardPrototype(k=params['k'], d=params['d'], r=params['r'], c=params['c'], w=params['w'],
                                     w_ul=params['w_ul'], pseudo_labels=params['pseudo_labels'])

    # Train models.
    model_1_1.train(xs_sl['xs_1'], ys_sl['ys_1'])
    model_1_2.train(xs_sl['xs_1'], ys_sl['ys_2'])
    model_2_1.train(xs_sl['xs_2'], ys_sl['ys_1'])
    model_2_2.train(xs_sl['xs_2'], ys_sl['ys_2'])

    # Assert prototypes in model equal expectations.
    np.testing.assert_array_almost_equal(model_1_1.prototypes, truth_sl['xs_1_ys_1'], 5)
    np.testing.assert_array_almost_equal(model_1_2.prototypes, truth_sl['xs_1_ys_2'], 5)
    np.testing.assert_array_almost_equal(model_2_1.prototypes, truth_sl['xs_2_ys_1'], 5)
    np.testing.assert_array_almost_equal(model_2_2.prototypes, truth_sl['xs_2_ys_2'], 5)

def test_prototype_ssl(params, xs_ssl, ys_ssl, truth_ssl):

    # Initialize models.
    model_1_1 = mp.StandardPrototype(k=params['k'], d=params['d'], r=params['r'], c=params['c'], w=params['w'],
                                     w_ul=params['w_ul'], pseudo_labels=params['pseudo_labels'])
    model_1_2 = mp.StandardPrototype(k=params['k'], d=params['d'], r=params['r'], c=params['c'], w=params['w'],
                                     w_ul=params['w_ul'], pseudo_labels=params['pseudo_labels'])
    model_2_1 = mp.StandardPrototype(k=params['k'], d=params['d'], r=params['r'], c=params['c'], w=params['w'],
                                     w_ul=params['w_ul'], pseudo_labels=params['pseudo_labels'])
    model_2_2 = mp.StandardPrototype(k=params['k'], d=params['d'], r=params['r'], c=params['c'], w=params['w'],
                                     w_ul=params['w_ul'], pseudo_labels=params['pseudo_labels'])

    # Train models.
    model_1_1.train(xs_ssl['xs_1'], ys_ssl['ys_1'])
    model_1_2.train(xs_ssl['xs_1'], ys_ssl['ys_2'])
    model_2_1.train(xs_ssl['xs_2'], ys_ssl['ys_1'])
    model_2_2.train(xs_ssl['xs_2'], ys_ssl['ys_2'])

    # Assert prototypes in model equal expectations.
    np.testing.assert_array_almost_equal(model_1_1.prototypes, truth_ssl['xs_1_ys_1'], 5)
    np.testing.assert_array_almost_equal(model_1_2.prototypes, truth_ssl['xs_1_ys_2'], 5)
    np.testing.assert_array_almost_equal(model_2_1.prototypes, truth_ssl['xs_2_ys_1'], 5)
    np.testing.assert_array_almost_equal(model_2_2.prototypes, truth_ssl['xs_2_ys_2'], 5)
