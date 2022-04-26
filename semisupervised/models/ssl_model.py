""" Parent class of all SSL models implemented in the models module.

Classes:
    SSLModel: superclass for the all SSL models.
"""

from abc import ABC
from abc import abstractmethod

import numpy as np
import scipy.optimize as opt


class SSLModel(ABC):
    """Parent class of all SSL models.

        Methods:
            reset: Reset model to initial values.
            learn: Perform supervised or unsupervised learning update.
            get_prediction_ul: Gets the model prediction in case trial is unlabelled.
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
            pseudo_labels: A string indicating whether the unsupervised update uses soft or hard labels.
    """

    def __init__(self, k=2, d=1, w_ul=1, pseudo_labels='hard', rnd_if_eq='fixed'):

        # Fixed model parameters.
        self.k = k
        self.d = d
        self.pseudo_labels = pseudo_labels

        # Free model parameters.
        self.w_ul = w_ul

        # Arguments controlling model implementation.
        self.rnd_if_eq = rnd_if_eq

    @abstractmethod
    def reset(self):
        """ Reset model to initial values. """
        pass

    @abstractmethod
    def learn(self, x, y):
        """ Perform supervised or unsupervised learning update. """
        pass

    def get_prediction_ul(self, x):
        """ Gets the model prediction in case trial is unlabelled. If hard labels, convert probabilities into most
        likely category label. In case multiple categories are equally likely chose one at random or the first by
        numerical order depending on the model specification.

        Arguments:
            x: A stimulus.

        Return:
            predictions: The predicted category labels [self.k,].

        """

        # Predict the label of the current stimulus.
        prediction = self.prediction(x)

        # If self-training with hard labels, replace predicted probabilities with most likely category label.
        if self.pseudo_labels == 'hard':
            _prediction = np.zeros(self.k)

            p_max = np.argwhere(prediction == prediction.max()).flatten()

            if len(p_max) > 1 and self.rnd_if_eq == 'random':
                # If multiple category labels are predicted with equal probability, chose one of them at random.
                _prediction[np.random.choice(p_max)] = 1

            else:
                # Treat most likely category label (or the first one in pmax if equally likely) like an observed label.
                _prediction[p_max[0]] = 1

            prediction = _prediction

        return prediction

    @abstractmethod
    def prediction(self, x):
        """ Make category prediction for current stimulus. """
        pass

    def prediction_multiple(self, xs):
        """Get category prediction for multiple stimuli.

        Arguments:
            xs: The stimuli with shape (self.d, number of stimuli).

        Returns:
            predictions: The predicted probabilities of each category for each stimulus with
                shape (model.k, number of stimuli)
        """

        _, n = xs.shape
        predictions = np.zeros((self.k, n))
        for i in range(n):
            predictions[:, i] = self.prediction(xs[:, i])

        return predictions

    def train(self, xs, ys):
        """Train the model on a sequence on stimuli and category labels.

        Arguments:
            xs: The stimuli with shape (self.d, number of stimuli).
            ys: The vector of stimulus labels with shape (1, number of stimuli).

        Returns:
            predictions: The category predictions on each learning trial with shape (self.k, number of stimuli).

        """

        n = len(ys)

        # Record label predictions.
        predictions = np.zeros((self.k, n))

        for i in range(n):
            # Store predictions.
            predictions[:, i] = self.prediction(xs[:, i])

            # Update model on next trial.
            self.learn(xs[:, i], ys[i])

        return predictions

    def optimize(self, data_train, data_behavior, free_param_ini, free_param_names, max_iter, full_output, retall):
        """Optimize the free parameters of the model to fit behavioral data.

        Arguments:
            data_train: Dictionary containing the the stimulus sequence (xs) and the label sequence (ys).
            data_behavior: Dictionary containing the responses (res) of one subject / simulation.
            free_param_ini: A list of initial values for the parameters that are being optimized.
            free_param_names: A list of names of parameters to indicate which are being optimized.
            max_iter: The maximum number of iterations.
            full_output: Boolean specifying output of optimization function.
            retall: Boolean specifying output of optimization function.

        Returns:
            result: The output of the optimization procedure.
        """

        assert len(free_param_ini) == len(free_param_names), 'Number of free parameter values and names is not the same.'

        # All parameters are optimised in n unconstrained regime and must thus be transformed in the model.
        param_format = 'transform'

        # Optimize.
        result = opt.fmin(self.evaluate, free_param_ini, args=(data_train, data_behavior, free_param_names, param_format),
                          maxiter=max_iter, full_output=full_output, retall=retall)

        return result

    @abstractmethod
    def evaluate(self, free_params, data_train, data_behavior, free_param_names, param_format):
        """ Evaluate model fit on data under the set of free parameter values. """
        pass

    @abstractmethod
    def evaluate_single_run(self, free_params, xs, ys, responses, free_param_names, param_format):
        """ Evaluate model fit for one subject under the set of free parameter values. """
        pass
