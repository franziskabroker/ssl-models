"""Module for testing exemplar_ml.py.

    Test whether exemplar model learns the correct predictions.

    Test runs with seed=1 on distribution task generation to replicate Zhu 2010: fails for other seeds due to high
    variance in predictions of exemplar model. Ballpark predictions are always correct though.
"""

import pytest
import numpy as np

import semisupervised.models.exemplar_ml as meb
import semisupervised.data.simulation_data as sd

# TODO test in d > 1, k > 2

@pytest.fixture
def params_1():
    params = {
        'k': [2, 2],
        'd': [1, 1],
        'h': [1, 1],
        'w_ul': [1, 0.5],
        'pseudo_labels': ['hard', 'hard'],
        'rnd_if_eq': ['fixed', 'fixed']
    }
    return params


@pytest.fixture
def data_1():
    xs = np.array([-2, 2, -1, 0.25])
    ys = np.array([0, 1, 0, -1])

    data = {
        'xs': xs,
        'ys': ys,
        'test': [-2, -1, 0, 1, 2],
        'truth': [[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]]
    }
    return data


@pytest.fixture
def params_paper():
    # The parameters in Zhu 2007 and 2010.

    params = {
        'k': [2, 2, 2],
        'd': [1, 1, 1],
        'h': [0.1, 0.6, 1],
        'w_ul': [1, 1, 1],
        'pseudo_labels': ['soft', 'soft', 'soft'],
        'rnd_if_eq': ['fixed', 'fixed', 'fixed']
    }
    return params


@pytest.fixture
def model_predictions_paper():
    # The predictions of the models in Zhu 2010.

    # Predictions model for the three parameter values (during learning, not test).
    predictions_order = {
        'x': np.array([-1.95, -1, 0, 1, 1.95]),
        'pred-r-2-l': [np.array([1, 1, 1, 1, 0.1]), np.array([1, 1, 1, 0.88, 0.52]), np.array([1, 1, 0.95, 0.8, 0.52])],
        'pred-l-2-r': [np.array([0, 0, 0, 0, 0.9]), np.array([0, 0, 0, 0.12, 0.45]), np.array([0, 0, 0.05, 0.2, 0.38])]
    }

    # Predictions model for the three parameter values (during post-test).
    predictions_shift = {
        'x': np.array([-1, -0.5, 0, 0.5, 1]),
        'pred-shiftL': [np.array([0, 0.04, 0.81, 0.95, 1]), np.array([0.18, 0.35, 0.6, 0.7, 0.75]), np.array([0.38, 0.4, 0.45, 0.5, 0.55])],
        'pred-shiftR': [np.array([0, 0.08, 0.62, 0.95, 1]), np.array([0.22, 0.3, 0.4, 0.62, 0.85]), np.array([0.45, 0.5, 0.55, 0.6, 0.65])]
    }

    return predictions_order, predictions_shift


def test_exemplar_model(params_1, data_1):

    for i in range(len(params_1['d'])):
        # Initialize model.
        model = meb.MachineLearningExemplar(k=params_1['k'][i], d=params_1['d'][i], h=params_1['h'][i], w_ul=params_1['w_ul'][i],
                                            pseudo_labels=params_1['pseudo_labels'][i], rnd_if_eq=params_1['rnd_if_eq'][i])

        # Train models.
        model.train(np.array([data_1['xs']]), data_1['ys'])

        predictions = model.prediction_multiple(np.array([data_1['test']]))

        np.testing.assert_array_equal(data_1['truth'][i], np.round(predictions[1, :]))

        # Test similarity function.

        # Test shape in 1D.
        # Initialize model.
        model = meb.MachineLearningExemplar(k=2, d=1, h=1, w_ul=1, pseudo_labels='hard')

        similarity = model.similarity(np.array([data_1['xs'][0]]), data_1['xs'].reshape((1, data_1['xs'].shape[0])))
        assert similarity.shape == (data_1['xs'].shape[0],)

        # Test symmetry in 1D.
        sim_1 = model.similarity(np.array([[1]]), np.array([[1, 0]]))
        sim_2 = model.similarity(np.array([[0]]), np.array([[1, 0]]))

        assert np.sum(sim_1) == np.sum(sim_2)

        # Test shape in 2D.
        model = meb.MachineLearningExemplar(k=2, d=2, h=1, w_ul=1, pseudo_labels='hard')

        similarity = model.similarity(np.array([[1], [2]]), np.array([[1, 1, 1], [2, 2, 2]]))
        assert similarity.shape == (3,)

        # Test symmetry in 2D.
        sim_1 = model.similarity(np.array([[1], [2]]), np.array([[1, 0], [1, 0]]))
        sim_2 = model.similarity(np.array([[0], [1]]), np.array([[0, -1], [0, -1]]))

        assert np.sum(sim_1) == np.sum(sim_2)


def test_exemplar_model_paper_order(params_paper, model_predictions_paper):
    # Functional test replicating the figures in Zhu et al. 2010 for the sequences ordered left-to-right and
    # right-to-left.

    paper_predictions_order, _ = model_predictions_paper

    for i in range(len(params_paper['d'])):
        for d in ['left-to-right', 'right-to-left']:
            # Initialize model.
            model = meb.MachineLearningExemplar(k=params_paper['k'][i], d=params_paper['d'][i], h=params_paper['h'][i], w_ul=params_paper['w_ul'][i],
                                                pseudo_labels=params_paper['pseudo_labels'][i], rnd_if_eq=params_paper['rnd_if_eq'][i])

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


def test_exemplar_model_paper_shift(params_paper, model_predictions_paper):
    # Functional test replicating the figures in Zhu et al. 2007 for the sequences ordered left-to-right and
    # right-to-left.

    _, paper_predictions_shift = model_predictions_paper

    for i in range(len(params_paper['d'])):
        for d in ['left', 'right']:
            # Initialize model.
            model = meb.MachineLearningExemplar(k=params_paper['k'][i], d=params_paper['d'][i], h=params_paper['h'][i], w_ul=params_paper['w_ul'][i],
                                                pseudo_labels=params_paper['pseudo_labels'][i], rnd_if_eq=params_paper['rnd_if_eq'][i])

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