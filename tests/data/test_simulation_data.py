"""Module for testing simulation_data.py.

    Test functions:
        test_get_test_grid
        test_simulation_datasets
"""

import pytest
import numpy as np

import semisupervised.data.simulation_data as sd


@pytest.fixture
def params():
    params = {
        'minimum': [-2, 0, 2],
        'maximum': [2, 2, 2],
        'n': [101, 101, 0],
    }

    return params


def test_get_test_grid(params):
    for i in params['minimum']:
        # Evaluate function.
        grid = sd.get_test_grid(params['minimum'][i], params['maximum'][i], params['n'][i])

        # Test length of vector.
        assert len(grid) == params['n'][i]


@pytest.fixture
def gaussian_samples():
    n_samples = 10000
    samples = {
        '0': sd.sample_1D_MoG([0], [1], [1], n_samples, seed=100),
        '1': sd.sample_1D_MoG([0, 0], [1, 1], [0.1, 0.9], n_samples, seed=100),
        '2': sd.sample_1D_MoG([0, 2], [1, 1], [0.5, 0.5], n_samples, seed=100),
        '3': sd.sample_1D_MoG([0, 4], [2, 2], [0.25, 0.75], n_samples, seed=100),
        '4': sd.sample_1D_MoG([-1, 0, 1], [2, 1, 2], [0.2, 0.6, 0.2], n_samples, seed=100),
    }

    return samples

@pytest.fixture
def true_mean():
    means = {
        '0': 0,
        '1': 0,
        '2': 1,
        '3': 3,
        '4': 0,
    }

    return means


@pytest.mark.slow
def test_sample_1D_MoG(gaussian_samples, true_mean):

    for k in gaussian_samples.keys():
        print(k)
        np.testing.assert_almost_equal(np.mean(gaussian_samples[k][:, 1]), true_mean[k], 2)


@pytest.fixture
def sequences():
    sequences = {
        'left-to-right': sd.zhu2010order('left-to-right'),
        'right-to-left': sd.zhu2010order('right-to-left'),
        'left-shift': sd.zhu2007shift('left', seed=100),
        'right-shift': sd.zhu2007shift('right', seed=100),
        'easy-to-hard': sd.curriculum('easy-to-hard'),
        'hard-to-easy': sd.curriculum('hard-to-easy'),
        'label-early': sd.labelOrder('early', seed=100),
        'label-late': sd.labelOrder('late', seed=100),
        'label-easy': sd.labelDifficulty('easy', seed=100),
        'label-hard': sd.labelDifficulty('hard', seed=100),
    }

    return sequences


def test_simulation_datasets(sequences):

    # Assert sequences from zhu2010order() are specified correctly.
    assert len(sequences['left-to-right']['xs']) == (10+81)
    assert np.sum(sequences['left-to-right']['ys'] == -1) == 81
    np.testing.assert_array_equal(sequences['left-to-right']['ys'], sequences['right-to-left']['ys'])
    np.testing.assert_array_equal(sequences['left-to-right']['xs'][10:], np.flip(sequences['right-to-left']['xs'][10:]))

    # Assert sequences from zhu2007shift() are specified correctly.
    assert len(sequences['left-shift']['xs']) == (20 + 21 + 3*230 + 3*21 + 21)
    assert len(sequences['right-shift']['xs']) == (20 + 21 + 3*230 + 3*21 + 21)
    assert np.sum(sequences['left-shift']['ys'] == -1) == (2 * 21 + 3*230 + 3*21)
    assert np.sum(sequences['right-shift']['ys'] == -1) == (2 * 21 + 3*230 + 3*21)
    np.testing.assert_almost_equal(np.mean(sequences['left-shift']['xs']),
                                   -0.4266666*(2 * 21 + 3*230)/(20 + 21 + 3*230 + 3*21 + 21), 1)
    np.testing.assert_almost_equal(np.mean(sequences['right-shift']['xs']),
                                   0.4266666*(2 * 21 + 3*230)/(20 + 21 + 3*230 + 3*21 + 21), 1)

    # Assert sequences from curriculum() are specified correctly.
    assert len(sequences['easy-to-hard']['xs']) == (10+82)
    assert len(sequences['hard-to-easy']['xs']) == (10+82)
    assert np.sum(sequences['easy-to-hard']['ys'] == -1) == 82
    assert np.sum(sequences['hard-to-easy']['ys'] == -1) == 82
    assert np.round(np.mean(sequences['easy-to-hard']['xs']), 4) == 0
    assert np.round(np.mean(sequences['hard-to-easy']['xs']), 4) == 0
    np.testing.assert_array_equal(np.sort(sequences['easy-to-hard']['xs']), np.sort(sequences['hard-to-easy']['xs']))

    # Assert sequences from labelOrder() are specified correctly.
    assert len(sequences['label-early']['xs']) == (10 + 3*230 + 3*21)
    assert len(sequences['label-late']['xs']) == (10 + 3*230 + 3*21)
    assert np.sum(sequences['label-early']['ys'] == -1) == np.floor((3*230 + 3*21)/2)
    assert np.sum(sequences['label-late']['ys'] == -1) == np.floor((3*230 + 3*21)/2)
    np.testing.assert_almost_equal(np.mean(sequences['label-early']['xs']), 0, 1)
    np.testing.assert_almost_equal(np.mean(sequences['label-late']['xs']), 0, 1)

    # Assert sequences from labelDifficulty() are specified correctly.
    assert len(sequences['label-easy']['xs']) == (10 + 3 * 230 + 3 * 21)
    assert len(sequences['label-hard']['xs']) == (10 + 3 * 230 + 3 * 21)
    assert np.sum(sequences['label-easy']['ys'] == -1) == np.floor((3 * 230 + 3 * 21) * 3 / 4)
    assert np.sum(sequences['label-hard']['ys'] == -1) == np.floor((3 * 230 + 3 * 21) * 3 / 4)
    np.testing.assert_almost_equal(np.mean(sequences['label-easy']['xs']), 0, 1)
    np.testing.assert_almost_equal(np.mean(sequences['label-hard']['xs']), 0, 1)