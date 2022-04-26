"""Machine Learning Semi-supervised prototype model.

The model is implemented based on the description of
Zhu, X., Gibson, B. R., Jun, K. S., Rogers, T. T., Harrison, J., & Kalish, C. (2010).
Cognitive models of test-item effects in human category learning.
In Proceedings of the 27th International Conference on Machine Learning (ICML-10) (pp. 1247-1254).

The model has one free parameters in the prior:  The number of pseudo-observations centered at mean zero for
all prototypes.

The model is extended to allow for self-training with hard labels in addition to soft labels, training with different
weights on unsupervised trials and extends to stimulus dimensions d > 1 and number of categories k > 2.

[This model has not been tested extensively for the cases of d > 1 and k > 2.]

Classes:
    Prototype: class for the Machine Learning prototype model.

"""

import numpy as np
import warnings
from scipy.stats import multivariate_normal

from semisupervised.models.ssl_model import SSLModel
import semisupervised.utils.mvnormpdf as mvnorm


class MachineLearningPrototype(SSLModel):
    """Class for the ML prototype model.
    The model represents categories by means of a mixture of Gaussians.  On every trial it updates the
    sufficient statistics of the data (E-step) and the mean, variance and probability of the respective category
    (M-step) after observing the label.  In the case that no category label is provided, the model learns from its own
    predictions or responses (self-training with soft or hard labels).  Unlike full EM, this model does not iterate
    until convergence on every trial but only updates once. The model assumes that the data contains as many clusters
    as there are categories and that there is a direct correspondence between cluster and label.

    Methods:
        initialize_prototypes: Set initial values of prototypes and sufficient statistics.
        reset: Reset prototypes to initial values.
        learn: Perform supervised or unsupervised update to prototypes depending on availability of the label.
        e_step_sl: Update the sufficient statistics after observing the current stimulus and its category label.
        e_step_ul: Update the sufficient statistics after observing the current stimulus in the absence of a category
            label.
        m_step: Update Gaussian mixture parameters.
        prediction: Make category prediction for current stimulus.
        prediction_multiple: Make category prediction for multiple stimuli.
        train: Train the model on a sequence on stimuli and category labels.
        optimize: Optimize model parameters to fit behavioural data.
        evaluate: Evaluate model fit on data under the set of free parameter values.
        evaluate_single_run: Evaluate model fit for one subject under the set of free parameter values.

    Attributes:
        k: The number of categories.
        d: The number of stimulus dimensions.
        w_ul: The weight of unsupervised trials.
        n0: The number of pseudo-counts at initialization.
        pseudo_labels: A string indicating whether the unsupervised update uses soft or hard labels.

        param_optim: A list of strings of all parameters that the model is built to optimize.

        mu: The mixture means.
        sigma: The mixture covariances.
        alpha: The mixture probabilities.

        s: The component of the sufficient statistics tracking the sum of observations.
        ss: The component of the sufficient statistics tracking the sum of squared observations.
        n: The component of the sufficient statistics tracking the number of observations per component.

        speed_pdf: Specifies whether fast and reliable or quick and less reliable function is used to calculate the
            probability of an input under a multivariate Gaussian distribution.
        rnd_if_eq: Specifies whether a random category assignment should be selected as a hard pseudo-label in case
            input is equally likely under multiple categories. If not, then first category is taken as default.

        """

    def __init__(self, k=2, d=1, n0=0.1, w_ul=1, pseudo_labels='hard', rnd_if_eq='fixed', speed_pdf='slow'):
        """Initialize. """

        # Call initializer of parent class.
        super().__init__(k=k, d=d, w_ul=w_ul, pseudo_labels=pseudo_labels, rnd_if_eq=rnd_if_eq)

        # Free model parameters.
        self.n0 = n0

        # Initialize prototypes and sufficient statistics.
        self.mu, self.sigma, self.alpha, self.s, self.ss, self.n = self.initialize_prototypes()

        assert n0 > 0,  'Initial pseudo-counts need to be greater than zero.'

        # Parameters that the model can optimize.
        self.param_optim = ['n0']

        # Arguments controlling model implementation.
        self.default_small_probability = 0.00001    # If Matrix is singular set prediction to this probability.
        self.speed_pdf = speed_pdf

    def initialize_prototypes(self):
        """Initialize model to initial state.

        Returns:
            mu, sigma, alpha, s, ss, n: The initialised prototype parameters and sufficient statistics.
        """
        # Initialize sufficient statistics with pseudo-counts (n=n0, mu=0, sigma=n0) as in Zhu 2010.
        s = np.zeros((self.d, self.k))
        ss = self.n0 * np.ones((self.d, self.d, self.k))
        n = self.n0 * np.ones(self.k)

        # Initialize prototypes.
        mu = np.zeros((self.d, self.k))
        sigma = np.ones((self.d, self.d, self.k))
        alpha = self.n0 * np.ones(self.k)
        alpha = alpha / np.sum(alpha)

        return mu, sigma, alpha, s, ss, n

    def reset(self):
        """Reset model to initial state."""

        # Reset prototypes.
        self.mu, self.sigma, self.alpha, self.s, self.ss, self.n = self.initialize_prototypes()

    def learn(self, x, y):
        """Perform supervised or unsupervised update to prototypes depending on availability of the label.

        Arguments:
            x: The current stimulus.
            y: The current label coded as [0, 1, 2, ...). The absence of a label is coded as -1.

        """

        if y == -1:
            self.e_step_ul(x)

        else:
            self.e_step_sl(x, y)

        self.m_step()

    def e_step_sl(self, x, y):
        """Update the sufficient statistics after observing the current stimulus and its category label.

        Arguments:
            x: The current stimulus.
            y: The current label coded as [0, 1, 2, ...)

         """

        self.n[y] += 1
        self.s[:, y] += x
        self.ss[:, :, y] += x*x.T

    def e_step_ul(self, x):
        """Update the sufficient statistics after observing the current stimulus in the absence of a category label.

        Arguments:
            x: The current stimulus.

        """

        # If hard labels, convert probabilities into most likely category label. In case multiple categories are
        # equally likely chose one at random or the first by numerical order depending on the model specification.
        prediction = super().get_prediction_ul(x)

        # Update all sufficient statistics.
        # In case of hard labels and an unsupervised weight of one, this update is equivalent to a supervised update.
        # In case of soft labels, all sufficient statistics are updated weighted by their probability.
        for i in range(self.k):

            self.n[i] += prediction[i]*self.w_ul
            self.s[:, i] += x*prediction[i]*self.w_ul
            self.ss[:, :, i] += x * x.T * prediction[i] * self.w_ul

    def m_step(self):
        """Update Gaussian mixture parameters.

        """
        self.alpha = self.n/np.sum(self.n)
        for i in range(self.k):
            if self.n[i] > 0:
                self.mu[:, i] = self.s[:, i]/self.n[i]
                self.sigma[:, :, i] = self.ss[:, :, i]/self.n[i] - self.s[:, i] * self.s[:, i].T / self.n[i]**2

    def prediction(self, x):
        """Make category prediction for current stimulus.

        Arguments:
            x: The current stimulus.

        Returns:
            prediction: Vector of probabilities of each category.
        """

        prediction_label = np.zeros(self.k)
        for i in range(self.k):
            try:
                # If matrix is not singular.
                if self.speed_pdf == 'fast':
                    prediction_label[i] = self.alpha[i]*mvnorm.mvnormpdf(x, self.mu[:, i], self.sigma[:, :, i])
                else:
                    prediction_label[i] = self.alpha[i] * multivariate_normal.pdf(x, mean=self.mu[:, i],
                                                                            cov=self.sigma[:, :, i])
            except:
                # If Matrix is singular set prediction to near zero.
                warnings.warn('Sigma is singular. Sigma: ' + str(self.sigma) + '; n: ' + str(self.n))
                prediction_label[i] = self.default_small_probability

        return prediction_label/np.sum(prediction_label)

    def prediction_multiple(self, xs):
        """Make category prediction for multiple stimuli.

        Arguments:
            xs: The stimuli with shape (self.d, number of stimuli).

        Returns:
            predictions: The predicted probabilities of each category for each stimulus with
                shape (self.k, number of stimuli)
        """

        return super().prediction_multiple(xs)

    def train(self, xs, ys):
        """Train the model on a sequence on stimuli and category labels.

        Arguments:
            xs: The stimuli with shape (self.d, number of stimuli).
            ys: The vector of stimulus labels with shape (number of stimuli,).

        Returns:
            predictions: The category predictions on each learning trial with shape (self.k, number of stimuli)

        """

        predictions = super().train(xs, ys)

        return predictions

    def optimize(self, data_train, data_behavior, free_param_ini, free_param_names, max_iter, full_output=True,
                 retall=True):
        pass

    def evaluate(self, free_params, data_train, data_behavior, free_param_names, param_format='direct'):
        pass

    def evaluate_single_run(self, free_params, xs, ys, responses, free_param_names, param_format='direct'):
        pass
