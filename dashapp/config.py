"""Config file for app.

"""

# Fast or slow implementation of multivariate Gaussian pdf evaluation.
speed_pdf = 'fast'

# If two categories are equally likely pick at random of default to one in unsupervised learning.
rnd_if_eq = 'fixed'

# Colors.
colors = {
    'proto': '228,26,28',
    'protoML': '152,78,163',
    'exemplar': '55,126,184',
    'exemplarML': '77,175,74',
    'clusterML': '231,41,138'}
