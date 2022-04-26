"""Machine Learning Semi-supervised exemplar model.

The model is implemented based on the description of
Zhu, X., Gibson, B. R., Jun, K. S., Rogers, T. T., Harrison, J., & Kalish, C. (2010).
Cognitive models of test-item effects in human category learning.
In Proceedings of the 27th International Conference on Machine Learning (ICML-10) (pp. 1247-1254).

The model has one free parameter:  The width of the similarity kernel.

The model is extended to allow for self-training with hard labels in addition to soft labels. It also extends to
stimulus dimensions d > 1 and number of categories k > 2.

[This model has not been tested extensively for the cases of d > 1 and k > 2.]

Classes:
    Exemplar: class for the Machine Learning exemplar model.

"""

import numpy as np

from semisupervised.models.ssl_model import SSLModel


class MachineLearningExemplar(SSLModel):
    """Class for the exemplar model.
    The model represents categories by means of all previous observations.  On every trial it computes its predictions
    based on how similar the observation is to all previous items within each category and adds the new observation and
    label to memory after observing both.  In the case that no category label is provided, the model memorises its own
    predictions or responses (self-training with soft or hard labels).

    Methods:
        initialize_exemplars: Initialize memory.
        reset: Reset memory to initial values.
        learn: Perform supervised or unsupervised update to memory depending on availability of the label.
        similarity: Compute similarity between two stimuli using a Gaussian kernel.
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
        h: The kernel bandwidth.
        pseudo_labels: A string indicating whether the unsupervised update uses soft or hard labels.

        param_optim: A list of strings of all parameters that the model is built to optimize.

        memory_xs: The past observations.
        memory_ys: The past (pseudo-)labels.
        memory_labelled: Indicator of past labelled or unlabelled trials.

        rnd_if_eq: Specifies whether a random category assignment should be selected as a hard pseudo-label in case
            input is equally likely under multiple categories. If not, then first category is taken as default.

        """

    def __init__(self, k=2, d=1, h=1, w_ul=1, pseudo_labels='hard', rnd_if_eq='fixed'):
        """Initialize. """

        # Call initializer of parent class.
        super().__init__(k=k, d=d, w_ul=w_ul, pseudo_labels=pseudo_labels, rnd_if_eq=rnd_if_eq)

        # Free model parameters.
        self.h = h

        # Parameters that the model can optimize.
        self.param_optim = []

        # Initialize memory.
        self.memory_xs, self.memory_ys, self.memory_labelled = self.initialize_exemplars()

        assert self.h > 0,  'Kernel bandwidth needs to be greater than zero.'

    def initialize_exemplars(self):
        """Initialize memory.

        Returns:
            memory_xs, memory_ys, memory_labelled: The initialized exemplar memories.
        """

        # Initialize memory.
        memory_xs = np.empty((self.d, 0))

        # Store labels for each observation and category in a matrix [self.k, n].
        memory_ys = np.empty((self.k, 0))

        # Store information whether trial was labelled for each observation.
        memory_labelled = np.empty(0)

        return memory_xs, memory_ys, memory_labelled

    def reset(self):
        """Reset model to initial state."""

        # Reset memory.
        self.memory_xs, self.memory_ys, self.memory_labelled = self.initialize_exemplars()

    def learn(self, x, y):
        """Perform supervised or unsupervised update to prototypes depending on availability of the label.

        Arguments:
            x: The current stimulus.
            y: The current label coded as [0, 1, 2, ...). The absence of a label is coded as -1.

        """

        # Add new label to memory.
        if y == -1:
            # If hard labels, convert probabilities into most likely category label. In case multiple categories are
            # equally likely chose one at random or the first by numerical order depending on the model specification.
            prediction = super().get_prediction_ul(x)

            self.memory_ys = np.concatenate((self.memory_ys, prediction.reshape((self.k, 1))), axis=1)
            self.memory_labelled = np.append(self.memory_labelled, 0)

        else:
            label_array = np.zeros(self.k)
            label_array[y] = 1
            self.memory_ys = np.concatenate((self.memory_ys, label_array.reshape((self.k, 1))), axis=1)
            self.memory_labelled = np.append(self.memory_labelled, 1)

        # Add new observation to memory.
        self.memory_xs = np.concatenate((self.memory_xs, x.reshape((self.d, 1))), axis=1)

    def similarity(self, x_1, x_vec):
        """Calculate similarity between one stimulus and a list of other stimuli using Gaussian kernel.

        Arguments:
            x_1: A stimulus of dimension self.d .
            x_vec: A stimulus vector of shape [self.d,n].

        Returns:
            similarity: The similarity of both stimuli [n, ].
        """
        x_1.reshape((self.d, 1))
        n = x_vec.shape[1]
        if n > 0:
            assert x_vec.shape == (self.d, n)

        similarity = 1/np.sqrt(2*np.pi)*np.exp(-0.5*np.sum(((x_1-x_vec)/self.h)**2, axis=0))

        return similarity

    def prediction(self, x):
        """Make category prediction for current stimulus.

        Arguments:
            x: The current stimulus.

        Returns:
            prediction: Vector of probabilities of each category.
        """

        # Set weight of observations.
        weights_sl_ul = np.repeat(1.0, self.memory_ys.shape[1])
        weights_sl_ul[self.memory_labelled == 0] = self.w_ul
        prediction_label = np.sum(np.tile(self.similarity(x, self.memory_xs), (self.k, 1)) * weights_sl_ul * self.memory_ys,
                            axis=1)

        assert prediction_label.shape == (self.k,)
        assert np.sum(np.isnan(prediction_label)) == 0

        if self.memory_ys.shape[1] > 0 and np.sum(prediction_label) > 0:
            return prediction_label/np.sum(prediction_label)

        else:
            # Chance prediction on initial data points with no memories to compare to, or few but very dissimilar
            # stimuli.
            return np.ones(self.k)/self.k

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