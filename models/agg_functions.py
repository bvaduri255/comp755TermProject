import numpy as np
from scipy.stats import laplace

""" Methods that only *requires* one argument """

def simple_mean(parameters, support=None):
    """Aggregates a list of parameters"""
    return np.mean(parameters, axis=0)

def proximal_operator_weighted_average(parameters, support=None):
    avg_params = np.mean(parameters, axis=0)
    # Apply proximal_operator to avg_params
    weights = proximal_operator(avg_params, 0.1)

    # Normalize weights
    weights = weights / np.sum(weights)
    return np.mean([w * x for w, x in zip(weights, parameters)], axis=0)

def random_weighted_average(parameters):
    """Applies validation accuracy to a list of parameters"""
    # Make random weights that sum to 1
    random_weights = np.random.dirichlet(np.ones(len(parameters)), size=1)[0]
    return sum(w * sum(x) for w, x in zip(random_weights, parameters)) / sum(random_weights) if sum(random_weights) != 0 else 0

def random_parameter_select(parameters, support=None):
    # Select random element from parameters
    return parameters[np.random.randint(0, len(parameters))]

""" Methods that require a constant hyperparameter """

def clipped_average(parameters, clip_range):
    new_params = np.clip([parameters], clip_range[0], clip_range[1])
    return np.mean(new_params, axis=0)

def differential_privacy_average(parameters, epsilon):
    """Applies differential privacy to a list of parameters"""
    return [laplace.rvs(loc=param, scale=1/epsilon) for param in parameters]

def proximal_operator(parameters, lam):
    """Applies proximal operator to a list of parameters"""
    return np.sign(parameters) * np.maximum(np.abs(parameters) - lam, 0)

""" Methods that require input information from each split data"""

def auc_roc_weighted_average(parameters, accuracies):
    auc_weights = accuracies / sum(accuracies)
    return sum(w * sum(x) for w, x in zip(auc_weights, parameters)) / sum(auc_weights) if sum(auc_weights) != 0 else 0
