"""Semi-supervised prototype model.

The supervised component of the model is implemented according to the description of Minda & Smith, Prototype models of
categorization; basic formulation, predictions, and limitations. In Formal Approaches in Categorisation,
Eds. Pothos & Wills, 2011

The model has several free parameters, most importantly the scaling parameter of the similarity function (c).

The supervised model is extended with an unsupervised component which implements self-training using either either
soft or hard labels in the absence of category labels.

[This model has not been tested extensively for soft labels.]

Classes:
    Prototype: class for the prototype model.

"""

import numpy as np

from semisupervised.models.ssl_model import SSLModel


class StandardPrototype(SSLModel):
    """Class for the prototype model.
    The model represents categories as the mean of observed category members.  It does so by iteratively updating the
    mean of the respective category after observing the label of any given observation.  In the case that no category
    label is provided, the model learns from its own predictions or responses (self-training with soft or hard labels).

    Methods:
        initialise_prototypes: Set initial values of prototypes.
        reset: Reset prototype to initial values.
        learn: Perform supervised or unsupervised update to prototypes depending on availability of the label.
        update_sl: Update the respective prototype after observing the current stimulus and its category label.
        update_ul: Update the respective prototype after observing the current stimulus in the absence of a category
            label.
        prediction: Make category prediction for current stimulus.
        prediction_multiple: Make category prediction for multiple stimuli.
        similarity: Computes the similarity between a stimulus and prototype as a weighted exponential decay of
            their distance.
        distance: Computes the distances between a stimulus and all prototypes weighting stimulus dimensions by
            attention weights.
        train: Train the model on a sequence on stimuli and category labels.
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

        # Initialize prototypes.
        self.prototypes, self.n = self.initialise_prototypes()

    def initialise_prototypes(self):
        """Initialise model to initial state.

        Returns:
            prototypes, n: The initialised prototype parameters and sufficient statistics.
        """

        # Initialize all prototypes at zero.
        prototypes = np.zeros((self.d, self.k))

        # Set initial number of observations per prototype to zero.
        n = np.zeros(self.k)

        return prototypes, n

    def reset(self):
        """Reset model to initial state."""

        # Reset prototypes.
        self.prototypes, self.n = self.initialise_prototypes()

    def learn(self, x, y):
        """Perform supervised or unsupervised update to prototypes depending on availability of the label.

        Arguments:
            x: The current stimulus.
            y: The current label coded as [0, 1, 2, ...). The absence of a label is coded as -1.

        """

        if y == -1:
            self.update_ul(x)
        else:
            self.update_sl(x, y)

    def update_sl(self, x, y):
        """Update the respective prototype after observing the current stimulus and its category label.

        Arguments:
            x: The current stimulus.
            y: The current label coded as [0, 1, 2, ...)

         """

        # Increment the number of observations for the respective category.
        self.n[y] += 1

        # Update the prototype using the equation for the incremental mean (rounding to avoid numerical issues).
        self.prototypes[:, y] += np.round((x - self.prototypes[:, y])/self.n[y], 5)

    def update_ul(self, x):
        """Update the respective prototype after observing the current stimulus in the absence of a category label.

        Arguments:
            x: The current stimulus.

        """

        # If hard labels, convert probabilities into most likely category label. In case multiple categories are
        # equally likely chose one at random or the first by numerical order depending on the model specification.
        prediction = super().get_prediction_ul(x)

        # Update all prototypes.
        # In the case of hard labels and an unsupervised weight of 1, this update is equivalent to a supervised update.
        # In the case of soft labels, all prototypes are updated weighted by their probability.
        for i in range(self.k):
            # Increment the number of observations for the respective category.
            self.n[i] += prediction[i]*self.w_ul

            if self.n[i] > 0:
                # Update the prototype using the equation for the incremental mean (rounding to avoid numerical issues).
                self.prototypes[:, i] += np.round((x - self.prototypes[:, i])*prediction[i]*self.w_ul/self.n[i], 5)

    def prediction(self, x):
        """Make category prediction for current stimulus.

        Arguments:
            x: The current stimulus.

        Returns:
            prediction: Vector of probabilities of each category.
        """

        # Compute similarity between current stimulus and all prototypes.
        similarities = self.similarity(self.distance(x))

        # Turn similarities into probability over categories by normalizing.
        prediction_label = similarities / np.sum(similarities)

        return prediction_label

    # label prediction for multiple inputs
    def prediction_multiple(self, xs):
        """Make category prediction for multiple stimuli.

        Arguments:
            xs: The stimuli with shape (self.d, number of stimuli).

        Returns:
            predictions: The predicted probabilities of each category for each stimulus with
                shape (self.k, number of stimuli)
        """

        return super().prediction_multiple(xs)

    def similarity(self, d):
        """Computes the similarities from a vector of distances.

        Arguments:
            d: A vector of distances.

        Returns:
            A vector of similarities.
        """

        return np.exp(-self.c*d)

    def distance(self, x):
        """Computes the distances between a stimulus and all prototypes weighting stimulus dimensions by
            attention weights.

        Arguments:
            x: The current stimulus.

        Returns:
            distances: A vector of distances of length self.k.
        """

        distances = np.zeros(self.k)
        for i in range(self.k):
            distances[i] = np.dot(self.w, np.abs(x-self.prototypes[:, i])**self.r)**(1/self.r)

        return distances

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
