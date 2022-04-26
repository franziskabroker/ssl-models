"""Module for testing prototype_ml.py.

    Test whether prototype model learns the correct prototypes.

    Test runs with seed=1 on distribution task generation to replicate Zhu 2010. Passes test for other seeds too.
"""

import pytest
import numpy as np

import semisupervised.data.simulation_data as sd
import semisupervised.models.prototype_ml as mpb

# TODO test in d > 1, k > 2


@pytest.fixture
def params():
    params = {
        'k': [2, 2, 2, 2, 2],
        'd': [1, 1, 1, 1, 1],
        'n0': [2, 2, 4, 8, 10],
        'w_ul': [1, 1, 1, 1, 1],
        'pseudo_labels': ['hard', 'soft', 'hard', 'soft', 'hard'],
        'speed_pdf': ['slow', 'fast', 'fast', 'slow', 'fast'],
        'rnd_if_eq': ['fixed', 'random', 'random', 'fixed', 'random']
    }
    return params


@pytest.fixture
def data():
    samples_1 = sd.sample_1D_MoG([-1, 1], [1, 1], [0.5, 0.5], 5000, seed=100)
    samples_2 = sd.sample_1D_MoG([-1, 1], [1, 1], [0.5, 0.5], 5000, seed=100)
    ys = np.concatenate((samples_1[:, 0], np.repeat(-1, 5000)))

    data = {
        'xs': np.concatenate((samples_1[:, 1], samples_2[:, 1])),
        'ys': ys.astype(int),
        'mean': [-1, 1],
        'std': [1, 1]
    }
    return data

@pytest.fixture
def params_paper():
    # The parameters in Zhu 2007 and 2010.

    params = {
        'k': [2, 2, 2],
        'd': [1, 1, 1],
        'n0': [1, 12, 20],
        'w_ul': [1, 1, 1],
        'pseudo_labels': ['soft', 'soft', 'soft'],
        'rnd_if_eq': ['fixed', 'fixed', 'fixed'],
        'speed_pdf': ['fast', 'fast', 'fast']
    }
    return params


@pytest.fixture
def model_predictions_paper():
    # The predictions of the models in Zhu 2010.

    # Predictions model for the three parameter values (during learning, not test).
    predictions_order = {
        'x': np.array([-1.95, -1, 0, 1, 1.95]),
        'pred-r-2-l': [np.array([1, 1, 0.9, 0.4, 0.15]), np.array([0.8, 0.75, 0.6, 0.35, 0.2]), np.array([0.75, 0.65, 0.55, 0.4, 0.25])],
        'pred-l-2-r': [np.array([0, 0, 0.1, 0.7, 0.95]), np.array([0.2, 0.25, 0.4, 0.65, 0.8]), np.array([0.25, 0.3, 0.45, 0.6, 0.75])]
    }

    # Predictions model for the three parameter values (during post-test).
    predictions_shift = {
        'x': np.array([-1, -0.5, 0, 0.5, 1]),
        'pred-shiftL': [np.array([0, 0.2, 0.95, 1, 1]), np.array([0.1, 0.3, 0.55, 0.8, 0.9]), np.array([0.3, 0.4, 0.5, 0.6, 0.7])],
        'pred-shiftR': [np.array([0, 0, 0.05, 0.82, 1]), np.array([0.1, 0.2, 0.45, 0.7, 0.95]), np.array([0.3, 0.4, 0.5, 0.6, 0.7])]
    }

    return predictions_order, predictions_shift

@pytest.mark.slow
def test_prototype_functions(params, data):

    for i in range(len(params['d'])):
        # Initialize model.
        model = mpb.MachineLearningPrototype(k=params['k'][i], d=params['d'][i], n0=params['n0'][i], w_ul=params['w_ul'][i],
                                             pseudo_labels=params['pseudo_labels'][i], speed_pdf=params['speed_pdf'][i],
                                             rnd_if_eq=params['rnd_if_eq'][i])

        # Train models.
        model.train(np.array([data['xs']]), data['ys'])

        np.testing.assert_array_almost_equal(data['mean'], model.mu.flatten(), 1)
        if params['pseudo_labels'][i] == 'soft':
            np.testing.assert_array_almost_equal(data['std'], model.sigma.flatten(), 1)
        else:
            assert np.sum(data['std'] < model.sigma.flatten()) == 0


def test_prototype_model_paper_order(params_paper, model_predictions_paper):
    # Functional test replicating the figures in Zhu et al. 2010 for the sequences ordered left-to-right and
    # right-to-left.

    paper_predictions_order, _ = model_predictions_paper

    for i in range(len(params_paper['d'])):
        for d in ['left-to-right', 'right-to-left']:
            # Initialize model.
            model = mpb.MachineLearningPrototype(k=params_paper['k'][i], d=params_paper['d'][i], n0=params_paper['n0'][i], w_ul=params_paper['w_ul'][i],
                                                 pseudo_labels=params_paper['pseudo_labels'][i], speed_pdf=params_paper['speed_pdf'][i],
                                                 rnd_if_eq=params_paper['rnd_if_eq'][i])

            # The data.
            data = sd.zhu2010order(d)

            # Train models.
            model_predictions = model.train(np.array([data['xs']]), data['ys'])

            # Compare predictions during training to paper figures.
            if d == 'left-to-right':
                paper_predictions = paper_predictions_order['pred-l-2-r'][i]
            else:
                paper_predictions = paper_predictions_order['pred-r-2-l'][i]

            selected_model_predictions = model_predictions[1, np.isin(np.round(data['xs'], 3), paper_predictions_order['x'])]
            assert(len(selected_model_predictions) == len(paper_predictions_order['x']))
            np.testing.assert_array_almost_equal(selected_model_predictions, paper_predictions, 1)


def test_prototype_model_paper_shift(params_paper, model_predictions_paper):
    # Functional test replicating the figures in Zhu et al. 2007 for the sequences ordered left-to-right and
    # right-to-left.

    _, paper_predictions_shift = model_predictions_paper

    for i in range(len(params_paper['d'])):
        for d in ['left', 'right']:
            # Initialize model.
            model = mpb.MachineLearningPrototype(k=params_paper['k'][i], d=params_paper['d'][i], n0=params_paper['n0'][i], w_ul=params_paper['w_ul'][i],
                                                 pseudo_labels=params_paper['pseudo_labels'][i], speed_pdf=params_paper['speed_pdf'][i],
                                                 rnd_if_eq=params_paper['rnd_if_eq'][i])

            # The data.
            data = sd.zhu2007shift(d, seed=1)

            # Train models.
            _ = model.train(np.array([data['xs']]), data['ys'])

            model_predictions = model.prediction_multiple(np.array([paper_predictions_shift['x']]))
            print(model_predictions)

            # Compare predictions during test to paper figures.
            if d == 'left':
                paper_predictions = paper_predictions_shift['pred-shiftL'][i]
            else:
                paper_predictions = paper_predictions_shift['pred-shiftR'][i]

            assert(len(model_predictions[1, :]) == len(paper_predictions_shift['x']))
            np.testing.assert_array_almost_equal(model_predictions[1, :], paper_predictions, 1)