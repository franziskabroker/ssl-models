# Semi-supervised Categorization Models

This code accompanies the PhD thesis: _Bröker, F. (2022). Semi-supervised categorisation: the role of feedback in human learning._

## Models

Two versions of semi-supervised prototype and exemplar models are provided.

### Machine learning inspired semi-supervised prototype and exemplar model
The prototype and exemplar model in `prototype_ml.py` and `exemplar_ml.py` are implementations of the models described in: 
_Zhu, X., Gibson, B. R., Jun, K. S., Rogers, T. T., Harrison, J., & Kalish, C. (2010).
Cognitive models of test-item effects in human category learning._

The model are extended to allow for self-training with hard labels in addition to soft labels, training with different
weights on unsupervised trials and extend to stimulus dimensions d > 1 and number of categories k > 2.

### Supervised categorisation inspired semi-supervised prototype and exemplar model

The supervised component of the prototype model `prototype.py` is implemented according to the description of _Minda & Smith, Prototype models of
categorization; basic formulation, predictions, and limitations. In Formal Approaches in Categorisation,
Eds. Pothos & Wills, 2011_. 

The supervised component of the exemplar model `exemplar.py` is implemented according to the description of _Robert M. Nosofsky, The generalized
context model: an exemplar model of classification. In Formal Approaches in Categorisation, Eds. Pothos & Wills, 2011._

The supervised model versions are both extended with an unsupervised component which implements self-training using either either
soft or hard labels in the absence of category labels based on the same principles as described in _Zhu et al. (2010)_.


## Datasets
Five different training datasets  are provided on which the models can be evaluated. 
Two of them capture the training data in the experiments reported in _Zhu et al. (2007, 2010)_. 

## Visualizing model predictions
Predictions of all models can be visualized on the available datasets using the dash app. To run the app, run `dashapp/app.py` and open app in browser.

Model parameters can be changed interactively as well as the type of labels or label weight in the model. 
It is also possible to compare predictions of different models or predictions on different training data.

The app provides two visualizations:

a) The category prediction of the model across the whole input range evaluated at different test points.

b) The score of the model defined as the summed log-probabilities for a 100 evenly spaced test data to be correctly labelled by the model 
after training.

