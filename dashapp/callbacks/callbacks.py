"""Callbacks for DashApp.

Implements the interactive functionality of the app.

Functions:
    get_model_predictions: Creates a new figure from the predictions of the models after user input.
    register_callbacks: Registers user input and triggers further functions.
    initialise_model: Initialises specified model.
"""

from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

import semisupervised.models.prototype as mp
import semisupervised.models.prototype_ml as mpb
import semisupervised.models.exemplar as me
import semisupervised.models.exemplar_ml as meb

import semisupervised.data.simulation_data as sd
import dashapp.config as cfg

# Global variables.
seed = 50
w_ul = 1
c_proto = 1
c_exemplar = 1
n0 = 1
h = 1
datasets = np.array(['zhu2010l2r'])
models = np.array(['proto'])
self_label_type = 'soft'


def initialise_model(selected_model):
    """Initialises specified model.

        Arguments:
            selected_model: String specifying selected model.

        Returns:
            The model object.
    """
    global self_label_type
    global w_ul

    # Standard prototype model.
    if selected_model == 'proto':
        k = 2
        d = 1
        r = 1
        w = np.ones(d) / np.sum(np.ones(d))

        return mp.StandardPrototype(k=k, d=d, r=r, c=c_proto, w=w, w_ul=w_ul, pseudo_labels=self_label_type,
                                    rnd_if_eq=cfg.rnd_if_eq)

    # ML prototype model.
    elif selected_model == 'protoML':
        k = 2
        d = 1

        return mpb.MachineLearningPrototype(k=k, d=d, n0=n0, w_ul=w_ul, pseudo_labels=self_label_type,
                                            rnd_if_eq=cfg.rnd_if_eq, speed_pdf=cfg.speed_pdf)

    # Standard exemplar model.
    elif selected_model == 'exemplar':
        k = 2
        d = 1
        r = 1
        w = np.ones(d) / np.sum(np.ones(d))

        return me.StandardExemplar(k=k, d=d, r=r, c=c_exemplar, w=w, w_ul=w_ul, pseudo_labels=self_label_type,
                                   rnd_if_eq=cfg.rnd_if_eq)

    # ML exemplar model.
    elif selected_model == 'exemplarML':
        k = 2
        d = 1

        return meb.MachineLearningExemplar(k=k, d=d, h=h, w_ul=w_ul, pseudo_labels=self_label_type,
                                           rnd_if_eq=cfg.rnd_if_eq)


def get_model_predictions():
    """Creates a new figure from the predictions of the models after user input.

    Returns:
        The new plotly figure object.
    """

    # Global parameters.
    global seed
    global c_proto
    global c_exemplar
    global datasets
    global models
    global self_label_type

    trace = []

    # Load default dataset.
    selected_data = sd.zhu2010order('left-to-right')

    # Get predictions on test trials.
    for dataset in datasets:

        for model in models:

            # Initialize model.
            model_obj = initialise_model(model)

            selected_data = sd.get_data(dataset, seed)
            last_test_trial = 0
            for test_trial in selected_data['test_trials']:

                # Train model.
                xs = np.array([selected_data['xs'][last_test_trial:test_trial]])
                ys = selected_data['ys'][last_test_trial:test_trial]
                _ = model_obj.train(xs, ys)

                # Evaluate model.
                test = np.array([selected_data['test_xs']])
                prediction_test = model_obj.prediction_multiple(test)

                # Update previous test trial.
                last_test_trial = test_trial

                # Append model prediction.
                color = cfg.colors[model]
                offset = len(selected_data['ys'])*0.5
                transparency = (test_trial+offset)/(len(selected_data['ys'])+offset)
                trace.append(go.Scatter(x=test[0],
                                        y=prediction_test[1, :],
                                        mode='lines',
                                        line=dict(color='rgba('+color+','+str(transparency)+')'),
                                        opacity=0.7,
                                        name=dataset + ' test trial ' + str(test_trial),
                                        textposition='bottom center'))

    traces = [trace]
    data = [val for sublist in traces for val in sublist]

    # Define figure.
    xs_min = selected_data['xs'].min()
    xs_max = selected_data['xs'].max()
    y_min = 0
    y_max = 1
    figure = {'data': data,
              'layout': go.Layout(
                  template='plotly',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Model Predictions', 'x': 0.5},
                  xaxis={'range': [xs_min, xs_max]},
                  yaxis={'range': [y_min, y_max]},
              ),

              }

    return figure


def get_model_test_scores():
    """Creates a new figure displaying the models scores on the test set after user input.

    Returns:
        The new plotly figure object.
    """

    # Global parameters.
    global seed
    global c_proto
    global c_exemplar
    global datasets
    global models
    global self_label_type

    df = pd.DataFrame(columns=['score', 'model', 'dataset'])

    # Load default dataset.
    selected_data = sd.zhu2010order('left-to-right')

    if datasets == [] or models == []:
        return {}

    else:
        # Get predictions on test trials.
        for dataset in datasets:

            for model in models:

                # Initialize model.
                model_obj = initialise_model(model)

                selected_data = sd.get_data(dataset, seed)

                # Test score on last test trial.
                test_trial = selected_data['test_trials'][-1]

                # Train model.
                xs = np.array([selected_data['xs'][0:test_trial]])
                ys = selected_data['ys'][0:test_trial]
                _ = model_obj.train(xs, ys)

                # Evaluate model.
                test = np.array([selected_data['test_xs']])
                prediction_test = model_obj.prediction_multiple(test)

                # Test score as average log likelihood
                prediction_score = np.sum(np.log(prediction_test[selected_data['test_ys'], np.arange(prediction_test.shape[1])]))

                # Append trace.
                df = df.append({'score': prediction_score, 'model': model, 'dataset': dataset}, ignore_index=True)

        # Define figure.
        figure = px.bar(df, x="model", y="score",
                        color="dataset", barmode="group",
                        title='Model performance (bigger is better)')
        figure.update(layout=dict(title=dict(x=0.5)))

    return figure


def register_callbacks(app):
    """Registers user input and updates graphs.

    Registers
        The value on the random seed slider.
        The value on the unsupervised weight slider.
        The value on the prototype parameter slider.
        The value on the ML prototype slider.
        The type of self-labelling on the radio item.
        The dataset selected or deselected in the dropdown.
        The models selected or deselected in the dropdown.

    """
    @app.callback(Output('slider-random-seed-output-container', 'children'),
                  Output('slider-unsupervised-weight-output-container', 'children'),
                  Output('slider-prototype-c-output-container', 'children'),
                  Output('slider-prototype-n0-output-container', 'children'),
                  Output('slider-exemplar-c-output-container', 'children'),
                  Output('slider-exemplar-h-output-container', 'children'),
                  Output('predictions', 'figure'),
                  Output('test-scores', 'figure'),
                  Input('slider-random-seed', 'value'),
                  Input('slider-unsupervised-weight', 'value'),
                  Input('slider-prototype-c', 'value'),
                  Input('slider-prototype-n0', 'value'),
                  Input('slider-exemplar-c', 'value'),
                  Input('slider-exemplar-h', 'value'),
                  Input('radio-item-self-label-type', 'value'),
                  Input('datasetselector', 'value'),
                  Input('modelselector', 'value'))
    def update_graphs(value_seed, value_w_ul, value_c_proto, value_n0, value_c_exemplar, value_h,
                      selected_self_label_type, selected_datasets, selected_models):

        global seed
        global w_ul
        global c_proto
        global c_exemplar
        global n0
        global h
        global self_label_type
        global datasets
        global models

        seed = np.int(value_seed)
        w_ul = value_w_ul
        c_proto = value_c_proto
        c_exemplar = value_c_exemplar
        n0 = value_n0
        h = value_h
        self_label_type = selected_self_label_type
        datasets = selected_datasets
        models = selected_models

        # Update figure.
        fig_predictions = get_model_predictions()
        fig_test_scores = get_model_test_scores()
        return 'seed: {}'.format(np.int(value_seed)), \
               'w: {}'.format(w_ul), \
               'c: {}'.format(value_c_proto), \
               'n0: {}'.format(value_n0), \
               'c: {}'.format(value_c_exemplar), \
               'h: {}'.format(value_h), \
               fig_predictions,\
               fig_test_scores
