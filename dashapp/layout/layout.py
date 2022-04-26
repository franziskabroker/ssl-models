"""Layout for DashApp.

Sets up the html divs.

Functions:
    create_layout(): Creates the html divs.

"""
from dash import html
from dash import dcc
import numpy as np

import semisupervised.data.simulation_data as sd


def create_layout():
    """Creates the html divs for the DashApp.

        Arguments:

        Returns:
            layout: The html layout.

    """
    layout = html.Div(
        children=[
            html.Div(className='row',
                     children=[
                         html.Div(className='four columns div-user-controls',
                                  children=[
                                      html.H2('Semi-supervised Categorization Models'),
                                      html.H1('''Visualising model fits.'''),
                                      html.P('''Pick a random seed.'''),
                                      html.Div(className='div-for-slider',
                                               children=[
                                                   dcc.Slider(
                                                       id='slider-random-seed',
                                                       min=0,
                                                       max=100,
                                                       step=0.1,
                                                       value=50,
                                                       updatemode='mouseup',
                                                       marks=None,
                                                   ),
                                                   html.Div(id='slider-random-seed-output-container')
                                               ]),
                                      html.P('''Pick the weight on unsupervised trials.'''),
                                      html.Div(className='div-for-slider',
                                               children=[
                                                   dcc.Slider(
                                                       id='slider-unsupervised-weight',
                                                       min=0,
                                                       max=1,
                                                       step=0.01,
                                                       value=1,
                                                       updatemode='mouseup',
                                                       marks=None,
                                                   ),
                                                   html.Div(id='slider-unsupervised-weight-output-container')
                                               ]),
                                      html.P('''Standard prototype model: vary the scaling parameter.'''),
                                      html.Div(className='div-for-slider',
                                               children=[
                                                   dcc.Slider(
                                                       id='slider-prototype-c',
                                                       min=1,
                                                       max=10,
                                                       step=0.01,
                                                       value=1,
                                                       updatemode='mouseup',
                                                       marks=None,
                                                   ),
                                                   html.Div(id='slider-prototype-c-output-container')
                                               ]),
                                      html.P('''ML prototype model: vary the pseudo-count parameter.'''),
                                      html.Div(className='div-for-slider',
                                               children=[
                                                   dcc.Slider(
                                                       id='slider-prototype-n0',
                                                       min=0.01,
                                                       max=50,
                                                       step=0.01,
                                                       value=1,
                                                       updatemode='mouseup',
                                                       marks=None,
                                                   ),
                                                   html.Div(id='slider-prototype-n0-output-container')
                                               ]),
                                      html.P('''Standard exemplar model: vary the scaling parameter.'''),
                                      html.Div(className='div-for-slider',
                                               children=[
                                                   dcc.Slider(
                                                       id='slider-exemplar-c',
                                                       min=1,
                                                       max=10,
                                                       step=0.01,
                                                       value=1,
                                                       updatemode='mouseup',
                                                       marks=None,
                                                   ),
                                                   html.Div(id='slider-exemplar-c-output-container')
                                               ]),
                                      html.P('''ML exemplar model: vary the kernel width.'''),
                                      html.Div(className='div-for-slider',
                                               children=[
                                                   dcc.Slider(
                                                       id='slider-exemplar-h',
                                                       min=0.001,
                                                       max=2,
                                                       step=0.01,
                                                       value=1,
                                                       updatemode='mouseup',
                                                       marks=None,
                                                   ),
                                                   html.Div(id='slider-exemplar-h-output-container')
                                               ]),
                                      html.P('''Select self-label type.'''),
                                      html.Div(className='div-for-radioitem',
                                               children=[
                                                   dcc.RadioItems(
                                                       id='radio-item-self-label-type',
                                                       options=[
                                                           {'label': 'hard label', 'value': 'hard'},
                                                           {'label': 'soft label', 'value': 'soft'}
                                                       ],
                                                       value='soft',
                                                       labelStyle={'display': 'inline-block'}
                                                   )
                                               ]),
                                      html.P('''Pick the models that will be trained.'''),
                                      html.Div(className='div-for-dropdown',
                                               children=[
                                                   dcc.Dropdown(id='modelselector',
                                                                options=[
                                                                    {'label': 'proto', 'value': 'proto'},
                                                                    {'label': 'protoML', 'value': 'protoML'},
                                                                    {'label': 'exemplar', 'value': 'exemplar'},
                                                                    {'label': 'exemplarML', 'value': 'exemplarML'}
                                                                ],
                                                                multi=True,
                                                                value=np.array(['proto']),
                                                                className='modelselector')
                                               ]),
                                      html.P('''Pick the datasets that the model will be trained on.'''),
                                      html.Div(className='div-for-dropdown',
                                               children=[
                                                   dcc.Dropdown(id='datasetselector',
                                                                options=sd.get_data_options(),
                                                                multi=True,
                                                                value=np.array(['zhu2010l2r']),
                                                                className='datasetselector')
                                               ])
                                  ]),
                         html.Div(className='eight columns div-for-charts',
                                  children=[
                                      dcc.Graph(id='predictions', config={'displayModeBar': False}),
                                      dcc.Graph(id='test-scores', config={'displayModeBar': False})
                                  ])
                     ])
        ])

    return layout
