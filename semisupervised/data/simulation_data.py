"""Datasets used in simulations.

Datasets (functions):
    zhu2007shift: The two conditions (unsupervised distributions shifted to the left or right) from
        Zhu, X., Rogers, T., Qian, R., & Kalish, C. (2007, January).
        Humans perform semi-supervised classification too. In AAAI (Vol. 2007, pp. 864-870).
    zhu2010order: The two conditions (stimuli ordered left-to-right and right-to-left) from
        Zhu, X., Gibson, B. R., Jun, K. S., Rogers, T. T., Harrison, J., & Kalish, C. (2010).
        Cognitive models of test-item effects in human category learning.
        In Proceedings of the 27th International Conference on Machine Learning (ICML-10) (pp. 1247-1254).
    curriculum: Two conditions in which stimuli are either ordered from easy items far of the boundary to hard items
        near the boundary or the other way round. Item range and trial number are taken from Zhu et al. 2010.
    labelOrder: Two conditions in which either the first half or the second half of trial is labelled. Stimuli are
        presented in random order drawn from a bimodal Gaussian centered at the category boundary (stimulus space
        adapted from Zhu et al. 2007).
    labelDifficulty: Two conditions in which either the easy or the hard trials are labelled. Stimuli are
        presented in random order drawn from a bimodal Gaussian centered at the category boundary (stimulus space
        adapted from Zhu et al. 2007).

Functions:
    get_data_options: Returns the available data options.
    get_data: Takes a string specifying the selected datasets and returns the corresponding dataset.
    get_test_grid: Creates an equally spaced grid to evaluate the predictions of the models.
    sample_1D_MoG: Samples from a 1D mixture of Gaussians.
    
"""

import numpy as np

# Available datasets to train models.
data_options = ['zhu2007shiftL',
                'zhu2007shiftR',
                'zhu2010l2r',
                'zhu2010r2l',
                'curriculumE2H',
                'curriculumH2E',
                'labelEarly',
                'labelLate',
                'labelEasy',
                'labelHard']


def get_data_options():
    """Returns the available data options.

        Returns:
            dict_list: A Dictionary containing all strings in data_options as label-value-pairs.
    """
    dict_list = []
    for i in data_options:
        dict_list.append({'label': i, 'value': i})

    return dict_list


def get_data(dataset, seed):
    """Takes a string from data_options and returns the corresponding dataset.
    Arguments:
        dataset: A string specifying the dataset.
        seed: A random seed.

    Returns:
        The data dictionary of the corresponding dataset.
    """
    if dataset == 'zhu2007shiftL':
        return zhu2007shift('left', seed)

    elif dataset == 'zhu2007shiftR':
        return zhu2007shift('right', seed)

    elif dataset == 'zhu2010l2r':
        return zhu2010order('left-to-right')

    elif dataset == 'zhu2010r2l':
        return zhu2010order('right-to-left')

    elif dataset == 'curriculumE2H':
        return curriculum('easy-to-hard')

    elif dataset == 'curriculumH2E':
        return curriculum('hard-to-easy')

    elif dataset == 'labelEarly':
        return labelOrder('early', seed)

    elif dataset == 'labelLate':
        return labelOrder('late', seed)

    elif dataset == 'labelEasy':
        return labelDifficulty('easy', seed)

    elif dataset == 'labelHard':
        return labelDifficulty('hard', seed)

    else:
        return {}


def get_test_grid(minimum, maximum, n):
    """Creates an equally spaced 1D grid to evaluate the predictions of the models.

        Arguments:
            minimum: The lowest value of the grid.
            maximum: The highest value of the grid.
            n: The number of equally spaced test items.

        Returns:
            grid: A vector with n equally spaced entries in the interval [minimum, maximum].

    """
    spacing = (maximum-minimum)/(n-1)
    if spacing > 0:
        grid = np.arange(minimum, maximum + spacing, spacing)
    else:
        grid = np.array([])

    # If one additional element due to numerical error, delete last item.
    if len(grid) == n+1:
        grid = grid[0:-1]

    assert len(grid) == n, 'Grid does not have correct length.'

    return grid


def sample_1D_MoG(mean, std, prob, n, seed):
    """Samples from a 1D mixture of Gaussians.

    Inputs:
        mean: An array containing the means of the Gaussian components.
        std: An array containing the  standard deviations of the Gaussian components.
        prob: The prior probability of each component.
        n: The number of samples to be drawn.
        seed: The random seed.
    """
    np.random.seed(seed)
    k = len(mean)
    samples = np.zeros((n, 2))
    for i in range(n):
        # Randomly choose Gaussian component to sample from.
        m = np.random.choice(k, 1, p=prob)[0]

        # Sample from selected Gaussian.
        samples[i, :] = [m, np.random.normal(mean[m], std[m], 1)[0]]
    return samples


def zhu2010order(order, stimulus_range=[-2, 2], num_model_test_trials=100):
    """Stimuli shown to subjects ordered right-to-left or left-to-right in Zhu et al. (2010).

    Argument:
        order: A string specifying the order of the stimuli.

    Returns:
        data: A dictionary containing the stimulus sequence (xs), the label sequence (ys), the test stimuli for the
            model (test) and a list of trial numbers after which the test is administered to the model (test_trials).

    """
    # Number of test trials for models should be even.
    assert np.mod(num_model_test_trials, 2) == 0

    # Stimulus range.
    x_range = np.array(stimulus_range)

    # Initial 10 supervised trials.
    xs_sl = np.tile(stimulus_range, 5)
    ys_sl = np.tile([0, 1], 5)

    # 81 unsupervised trials ordered right-to-left or left-to-right.
    if order == 'left-to-right':
        xs_ul = get_test_grid(x_range[0], x_range[1], 81)
    else:
        xs_ul = get_test_grid(x_range[0], x_range[1], 81)[::-1]

    ys_ul = np.repeat(-1, len(xs_ul))

    # Test grid the model is evaluated on.
    test_model_xs = get_test_grid(x_range[0], x_range[1], num_model_test_trials)
    test_model_ys = np.concatenate((np.repeat(0, int(num_model_test_trials / 2)),
                                    np.repeat(1, int(num_model_test_trials / 2)))).astype(int)
    # Trials after which to evaluate model on test.
    test_trials = np.array([10, len(xs_sl) + len(xs_ul)])

    data = {
        'xs': np.concatenate((xs_sl, xs_ul)),
        'ys': np.concatenate((ys_sl, ys_ul)),
        'test_xs': test_model_xs,
        'test_ys': test_model_ys,
        'test_trials': test_trials
    }

    return data


def zhu2007shift(shift, seed, stimulus_range=[-2.5, 2.5], num_model_test_trials=100):
    """Stimuli shown to subjects in the left or right shift condition in Zhu et al. (2007).
    Other than in Zhu et al. (2010), the stimulus range was constrained to [-2.5, 2.5]

    Argument:
        shift: A string specifying the direction of the distributional shift.
        seed: A random seed.

    Returns:
        data: A dictionary containing the stimulus sequence (xs), the label sequence (ys), the test stimuli for the
            model (test) and a list of trial numbers after which the test is administered to the model (test_trials).

    """
    # Number of test trials for models should be even.
    assert np.mod(num_model_test_trials, 2) == 0

    # Set random seed.
    np.random.seed(seed)

    # Stimulus range.
    x_range = np.array(stimulus_range)

    # Initial 20 supervised trials.
    xs_sl = np.tile([-1, 1], 10)
    ys_sl = np.tile([0, 1], 10)

    # 21 grid items subjects are tested on.
    test_subject_xs = get_test_grid(-1, 1, 21)
    np.random.shuffle(test_subject_xs)
    test_subject_ys = np.repeat(-1, len(test_subject_xs))

    # 3 blocks of 230 unsupervised trials with mean shifted and 21 grid examples randomly intermixed.
    if shift == 'left':
        mean = np.array([-1 - 1.28 * 1/3, 1 - 1.28 * 1/3])
    else:
        mean = np.array([-1 + 1.28 * 1/3, 1 + 1.28 * 1/3])
    std = np.array([1/3, 1/3])
    prob = np.array([0.5, 0.5])
    mog_samples = sample_1D_MoG(mean, std, prob, 230, seed)[:, 1]

    grid_xs = get_test_grid(stimulus_range[0], stimulus_range[1], 21)

    xs_ul = np.concatenate((mog_samples, grid_xs))

    # The three blocks.
    xs_ul_block_1 = xs_ul.copy()
    np.random.shuffle(xs_ul_block_1)
    xs_ul_block_2 = xs_ul.copy()
    np.random.shuffle(xs_ul_block_2)
    xs_ul_block_3 = xs_ul.copy()
    np.random.shuffle(xs_ul_block_3)

    # The three blocks concatenated.
    xs_ul_block_123 = np.concatenate((xs_ul_block_1, xs_ul_block_2, xs_ul_block_3))
    ys_ul_block_123 = np.repeat(-1, len(xs_ul_block_123))

    # The concatenated stimulus and label sequence.
    xs = np.concatenate((xs_sl, test_subject_xs, xs_ul_block_123, test_subject_xs))
    ys = np.concatenate((ys_sl, test_subject_ys, ys_ul_block_123, test_subject_ys))

    # Test grid the model is evaluated on.
    test_model_xs = get_test_grid(x_range[0], x_range[1], num_model_test_trials)
    test_model_ys = np.concatenate((np.repeat(0, int(num_model_test_trials/2)),
                                    np.repeat(1, int(num_model_test_trials/2)))).astype(int)

    # Trials after which to evaluate model on test.
    test_trials = np.array([20+21, len(xs)])
    data = {
        'xs': xs,
        'ys': ys,
        'test_xs': test_model_xs,
        'test_ys': test_model_ys,
        'test_trials': test_trials
    }

    return data


def curriculum(order, stimulus_range=[-2, 2], num_model_test_trials=100):
    """Stimuli ordered from easy items far of the boundary to hard items near the boundary or the other way round.
    Item range and trials number are taken from Zhu et al. 2010.

    Argument:
        order: A string specifying the order of the stimuli.

    Returns:
        data: A dictionary containing the stimulus sequence (xs), the label sequence (ys), the test stimuli for the
            model (test) and a list of trial numbers after which the test is administered to the model (test_trials).

    """
    # Number of test trials for models should be even.
    assert np.mod(num_model_test_trials, 2) == 0

    # Stimulus range.
    x_range = np.array(stimulus_range)

    # Initial 10 supervised trials.
    xs_sl = np.tile(stimulus_range, 5)
    ys_sl = np.tile([0, 1], 5)

    # 82 unsupervised trials ordered easy-to-hard (converging zig-zag pattern in x-space) or hard-to-easy.
    category_0 = get_test_grid(x_range[0], 0, 41)
    category_1 = get_test_grid(0, x_range[1], 41)

    xs_ul = np.array(list(zip(category_0, np.flip(category_1)))).flatten()
    if order == 'hard-to-easy':
        xs_ul = np.flip(xs_ul)

    ys_ul = np.repeat(-1, len(xs_ul))

    # Test grid the model is evaluated on.
    test_model_xs = get_test_grid(x_range[0], x_range[1], num_model_test_trials)
    test_model_ys = np.concatenate((np.repeat(0, int(num_model_test_trials / 2)),
                                    np.repeat(1, int(num_model_test_trials / 2)))).astype(int)
    # Trials after which to evaluate model on test.
    test_trials = np.array([10, len(xs_sl) + len(xs_ul)])

    data = {
        'xs': np.concatenate((xs_sl, xs_ul)),
        'ys': np.concatenate((ys_sl, ys_ul)),
        'test_xs': test_model_xs,
        'test_ys': test_model_ys,
        'test_trials': test_trials
    }

    return data


def labelOrder(order, seed, stimulus_range=[-2.5, 2.5], num_model_test_trials=100):
    """Stimuli labelled either in the first half or the second half of trials. Stimuli are presented in random order
    drawn from a bi-modal Gaussian centered at the category boundary (stimulus space adapted from Zhu et al. 2007), but
    only half the initial supervised trials as in Zhu et al. 2010 and no test grids administered to learner.

    Stimuli with value 0 are labelled as category 1.

    Argument:
        order: A string specifying which half of trials is labelled.
        seed: A random seed.

    Returns:
        data: A dictionary containing the stimulus sequence (xs), the label sequence (ys), the test stimuli for the
            model (test) and a list of trial numbers after which the test is administered to the model (test_trials).

    """
    # Number of test trials for models should be even.
    assert np.mod(num_model_test_trials, 2) == 0

    # Set random seed.
    np.random.seed(seed)

    # Stimulus range.
    x_range = np.array(stimulus_range)

    # Initial 10 supervised trials.
    xs_sl = np.tile([-1, 1], 5)
    ys_sl = np.tile([0, 1], 5)

    # 3 blocks of 230 unsupervised trials with mean at -1/3 and 1/3 and 21 grid examples randomly intermixed.
    mean = np.array([-1*1/3, 1*1/3])

    std = np.array([1/3, 1/3])
    prob = np.array([0.5, 0.5])
    mog_samples = sample_1D_MoG(mean, std, prob, 230, seed)[:, 1]

    grid_xs = get_test_grid(stimulus_range[0], stimulus_range[1], 21)

    xs_ul = np.concatenate((mog_samples, grid_xs))

    # The three blocks.
    xs_ul_block_1 = xs_ul.copy()
    np.random.shuffle(xs_ul_block_1)
    xs_ul_block_2 = xs_ul.copy()
    np.random.shuffle(xs_ul_block_2)
    xs_ul_block_3 = xs_ul.copy()
    np.random.shuffle(xs_ul_block_3)

    # The three blocks concatenated.
    xs_ul_block_123 = np.concatenate((xs_ul_block_1, xs_ul_block_2, xs_ul_block_3))
    ys_ul_block_123 = np.repeat(-1, len(xs_ul_block_123))

    if order == 'early':
        trial_half = np.arange(len(xs_ul_block_123) / 2).astype(int)
        category_0 = xs_ul_block_123[trial_half] < 0
        category_1 = xs_ul_block_123[trial_half] >= 0
    else:
        trial_half = np.arange(len(xs_ul_block_123) / 2, len(xs_ul_block_123)).astype(int)
        category_0 = xs_ul_block_123[trial_half] < 0
        category_1 = xs_ul_block_123[trial_half] >= 0

    ys_ul_block_123[trial_half[category_0]] = np.repeat(0, sum(category_0))
    ys_ul_block_123[trial_half[category_1]] = np.repeat(1, sum(category_1))

    # The concatenated stimulus and label sequence.
    xs = np.concatenate((xs_sl, xs_ul_block_123))
    ys = np.concatenate((ys_sl, ys_ul_block_123))

    # Test grid the model is evaluated on.
    test_model_xs = get_test_grid(x_range[0], x_range[1], num_model_test_trials)
    test_model_ys = np.concatenate((np.repeat(0, int(num_model_test_trials / 2)),
                                    np.repeat(1, int(num_model_test_trials / 2)))).astype(int)
    # Trials after which to evaluate model on test.
    test_trials = np.array([10, len(xs)])
    data = {
        'xs': xs,
        'ys': ys,
        'test_xs': test_model_xs,
        'test_ys': test_model_ys,
        'test_trials': test_trials
    }

    return data


def labelDifficulty(difficulty, seed, stimulus_range=[-2.5, 2.5], num_model_test_trials=100):
    """Either the 25% most easy or hard trials are labelled. Stimuli are presented in random order drawn from a bi-modal
    Gaussian centered at the category boundary (stimulus space adapted from Zhu et al. 2007), but only half the initial
    supervised trials as in Zhu et al. 2010 and no test grids administered to learner.

    Argument:
        order: A string specifying which difficulty level of trials is labelled.
        seed: A random seed.

    Returns:
        data: A dictionary containing the stimulus sequence (xs), the label sequence (ys), the test stimuli for the
            model (test) and a list of trial numbers after which the test is administered to the model (test_trials).

    """
    # Set random seed.
    np.random.seed(seed)

    # Stimulus range.
    x_range = np.array(stimulus_range)

    # Initial 10 supervised trials.
    xs_sl = np.tile([-1, 1], 5)
    ys_sl = np.tile([0, 1], 5)

    # 3 blocks of 230 unsupervised trials with mean at -1/3 and 1/3 and 21 grid examples randomly intermixed.
    mean = np.array([-1 * 1/3, 1 * 1/3])

    std = np.array([1/3, 1/3])
    prob = np.array([0.5, 0.5])
    mog_samples = sample_1D_MoG(mean, std, prob, 230, seed)[:, 1]

    grid_xs = get_test_grid(stimulus_range[0], stimulus_range[1], 21)

    xs_ul = np.concatenate((mog_samples, grid_xs))

    # The three blocks.
    xs_ul_block_1 = xs_ul.copy()
    np.random.shuffle(xs_ul_block_1)
    xs_ul_block_2 = xs_ul.copy()
    np.random.shuffle(xs_ul_block_2)
    xs_ul_block_3 = xs_ul.copy()
    np.random.shuffle(xs_ul_block_3)

    # The three blocks concatenated.
    xs_ul_block_123 = np.concatenate((xs_ul_block_1, xs_ul_block_2, xs_ul_block_3))
    ys_ul_block_123 = np.repeat(-1, len(xs_ul_block_123))

    xs_ul_block_123_sorted = sorted(np.abs(xs_ul_block_123))
    if difficulty == 'hard':
        boundary = xs_ul_block_123_sorted[int(len(xs_ul_block_123) / 4)]
        category_0 = (xs_ul_block_123 <= 0) & (xs_ul_block_123 >= -boundary)
        category_1 = (xs_ul_block_123 > 0) & (xs_ul_block_123 <= boundary)

    else:
        boundary = xs_ul_block_123_sorted[int(len(xs_ul_block_123) * 3 / 4)]
        category_0 = xs_ul_block_123 <= -boundary
        category_1 = xs_ul_block_123 >= boundary

    ys_ul_block_123[category_0] = np.repeat(0, sum(category_0))
    ys_ul_block_123[category_1] = np.repeat(1, sum(category_1))

    # The concatenated stimulus and label sequence.
    xs = np.concatenate((xs_sl, xs_ul_block_123))
    ys = np.concatenate((ys_sl, ys_ul_block_123))

    # Test grid the model is evaluated on.
    test_model_xs = get_test_grid(x_range[0], x_range[1], num_model_test_trials)
    test_model_ys = np.concatenate((np.repeat(0, int(num_model_test_trials / 2)),
                                    np.repeat(1, int(num_model_test_trials / 2)))).astype(int)

    # Trials after which to evaluate model on test.
    test_trials = np.array([10, len(xs)])
    data = {
        'xs': xs,
        'ys': ys,
        'test_xs': test_model_xs,
        'test_ys': test_model_ys,
        'test_trials': test_trials
    }

    return data
