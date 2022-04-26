"""Module for testing mvnormpdf.py.

    Test whether mvnorm function returns the same probabilities as scipy.stats.multivariate_normal.pdf.

"""

import pytest
import numpy as np
from scipy.stats import multivariate_normal
import semisupervised.utils.mvnormpdf as mvnorm


@pytest.fixture
def xs():
    xs = {
        '1': np.array([1]),
        '2': np.array([0.1, 10]),
        '3': np.array([0, 0.0001, 100])
    }
    return xs


@pytest.fixture
def mus():
    mus = {
        '1': np.array([0.5]),
        '2': np.array([10, 0.1]),
        '3': np.array([10, 10, 0.1])
    }
    return mus


@pytest.fixture
def sigmas():
    sigmas = {
        '1': np.array([[0.5]]),
        '2': np.array([[2, -0.1], [-0.1, 2]]),
        '3': np.eye(3)
    }
    return sigmas


def test_mvnorm(xs, mus, sigmas):
    for d in range(3):
        np.testing.assert_array_almost_equal(mvnorm.mvnormpdf(xs[str(d+1)], mus[str(d+1)], sigmas[str(d+1)]),
                                             multivariate_normal.pdf(xs[str(d+1)], mus[str(d+1)], sigmas[str(d+1)]))

