"""Semi-supervised exemplar model.

The supervised component of the model is implemented according to the description of Robert M. Nosofsky, The generalized
context model: an exemplar model of classification. In Formal Approaches in Categorisation, Eds. Pothos & Wills, 2011.

The model has several free parameters, most importantly the scaling parameter (c) of the similarity function. Four
parameters were omitted by setting them to 1 in order to closer match the prototype model implementation:
- The shape parameter relating distance to similarity (p)
- The memory strength of individual exemplars (V)
- The response-scaling parameter (gamma)
- The response-bias for each category (b)

The supervised model is extended with an unsupervised component which implements self-training using either either
soft or hard labels in the absence of category labels.

[This model has not been tested extensively.]

Classes:
    Exemplar: class for the exemplar model.

"""

import numpy as np

from semisupervised.models.ssl_model import SSLModel


class StandardExemplar(SSLModel):
    """Class for the exemplar model.
    The model represents categories by means of all previous observations.  On every trial it computes its predictions
    based on how similar the observation is to all previous items of each category and adds the new observation and
    label to memory after observing both.  In the case that no category label is provided, the model memorises its own
    predictions or responses (self-training with soft or hard labels).

    Methods:
        initialise_exemplars: Initialise memory.
        reset: Reset memory to initial values.
        learn: Perform supervised or unsupervised update to memory depending on availability of the label.
        similarity: Convert a vector of distances into a vector of similarity by applying an exponential decay.
        distance: Compute distances between a stimulus and all exemplars of each category weighting stimulus dimensions
            by attention weights.
        prediction: Make category prediction for current stimulus.
        prediction_multiple: Make category prediction for multiple stimuli.
        train: Train model on a sequence of stimuli and category labels.
        optimize: Optimize model parameters to fit behavioural data.
        evaluate: Evaluate model fit on data under the set of free parameter values.
        evaluate_single_run: Evaluate model fit for one subject under the set of free parameter values.

    Attributes:
        k: The number of categories.
        d: The number of stimulus dimensions.
        w: The attention weights on stimulus dimensions.
        r: The order of distance (typically 1 (city block) or 2 (Euclidean)).
        c: The scaling parameter of the similarity function.
        w_ul: The weight of unsupervised trials.
        pseudo_labels: A string indicating whether the unsupervised update uses soft or hard labels.

        param_optim: A list of strings of all parameters that the model is built to optimize.

        memory_xs: The past observations.
        memory_ys: The past (pseudo-)labels.
        memory_labelled: Indicator of past labelled or unlabelled trials.

        rnd_if_eq: Specifies whether a random category assignment should be selected as a hard pseudo-label in case
            input is equally likely under multiple categories. If not, then first category is taken as default.

        """

    def __init__(self, k=2, d=1, w=None, r=1, c=1, w_ul=1, pseudo_labels='hard', rnd_if_eq='fixed'):
        """Initialize. """

        # Call initializer of parent class.
        super().__init__(k=k, d=d, w_ul=w_ul, pseudo_labels=pseudo_labels, rnd_if_eq=rnd_if_eq)

        # Fixed model parameters.
        self.r = r

        # Free model parameters.
        self.c = c

        if w is None:
            w = np.ones(d) / d
        self.w = w

        # Test whether attention weights are correctly specified.
        assert (np.max(self.w) <= 1 and np.min(self.w) >= 0), 'Attention weights are out of bounds.'
        assert np.round(np.sum(self.w), 5) == 1, 'Attention weights do not sum to one.'
        assert self.w.shape[0] == self.d, 'Attention weights have wrong shape.'

        # Test whether shape parameter is correctly specified.
        assert self.c > 0, 'Shape parameter of similarity function needs to be greater than zero.'

        # Parameters that the model can optimize.
        self.param_optim = []

        # Initialize memory.
        self.memory_xs, self.memory_ys, self.memory_labelled = self.initialise_exemplars()

    def initialise_exemplars(self):
        """Initialize memory.

        Returns:
            memory_xs, memory_ys, memory_labelled: The initialised exemplar memories.
        """

        # Initialize memory of stimulus values.
        memory_xs = np.empty((self.d, 0))

        # Initialize memory of (pseudo-)labels for each observation in a matrix [self.k, n]. In case of supervised
        # trials or hard pseudo-labels the memory for that trial will be a one-hot-vector. In case of soft pseuo-labels,
        # it contains the probability of each category label.
        memory_ys = np.empty((self.k, 0))

        # Initialize memory of whether a trial was supervised or unsupervised for each observation.
        memory_labelled = np.empty(0)

        return memory_xs, memory_ys, memory_labelled

    def reset(self):
        """Reset model to initial state."""

        # Reset memory.
        self.memory_xs, self.memory_ys, self.memory_labelled = self.initialise_exemplars()

    def learn(self, x, y):
        """Perform supervised or unsupervised update to memory depending on availability of the label.

        Arguments:
            x: The current stimulus.
            y: The current label coded as [0, 1, 2, ...). The absence of a label is coded as -1.

        """

        # Add new label to memory.
        if y == -1:
            # If hard labels, convert probabilities into most likely category label. In case multiple categories are
            # equally likely chose one at random or the first by numerical order depending on the model specification.
            prediction = super().get_prediction_ul(x)

            # Store pseudo-label.
            self.memory_ys = np.concatenate((self.memory_ys, prediction.reshape((self.k, 1))), axis=1)
            # Store information on availability of label (unsupervised -> 0).
            self.memory_labelled = np.append(self.memory_labelled, 0)

        else:
            label_array = np.zeros(self.k)
            label_array[y] = 1
            # Store label.
            self.memory_ys = np.concatenate((self.memory_ys, label_array.reshape((self.k, 1))), axis=1)
            # Store information on availability of label (supervised -> 1).
            self.memory_labelled = np.append(self.memory_labelled, 1)

        # Add new observation to memory.
        self.memory_xs = np.concatenate((self.memory_xs, x.reshape((self.d, 1))), axis=1)

    def distance(self, x, x_vec):
        """Computes distances between a stimulus and all exemplars of each category weighting stimulus dimensions by
            attention weights.

        Arguments:
            x: The current stimulus.
            x_vec:  A stimulus vector of shape [self.d,n].

        Returns:
            distances: A vector of distances.
        """
        x.reshape((self.d, 1))
        n = x_vec.shape[1]
        if n > 0:
            assert x_vec.shape == (self.d, n)

            distances = np.dot(self.w, np.abs(x-x_vec)**self.r)**(1/self.r)

        else:
            # Return zero distances if no exemplars are stored yet.
            distances = np.dot(self.w, np.zeros_like(x))

        return distances

    def similarity(self, d):
        """Convert a vector of distances into a vector of similarity by applying an exponential decay.

        Arguments:
            d: A vector of distances.

        Returns:
            A vector of similarities.
        """

        return np.exp(-self.c*d)

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
        prediction_label = np.sum(np.tile(self.similarity(self.distance(x, self.memory_xs)), (self.k, 1)) * weights_sl_ul
                            * self.memory_ys, axis=1)

        assert prediction_label.shape == (self.k,)
        assert np.sum(np.isnan(prediction_label)) == 0

        if self.memory_ys.shape[1] > 0 and np.sum(prediction_label) > 0:
            # Normalize category predictions to sum to one.
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
