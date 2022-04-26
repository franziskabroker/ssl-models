"""Calculate the probability of an observation under a multivariate Gaussian.

Fast implementation of scipy.stats.multivariate_normal.pdf that may be lower in precision and does
not implement tests like singularity of covariance matrix. Adapted from: https://github.com/scipy/scipy/issues/9921

"""

import numpy as np
from numpy.linalg import det, inv


def mvnormpdf(x, mu, sigma):
    """
    :param x: The observation.
    :param mu: The mean of the Gaussian.
    :param sigma: The covariance of the Gaussian.
    :return: The probability of the observation.
    """
    k = mu.shape[0]
    den = np.sqrt((2 * np.pi) ** k * det(sigma))
    diff = x - mu
    return np.exp(-diff @ inv(sigma) @ diff / 2) / den
