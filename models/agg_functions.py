import numpy as np
from scipy.stats import laplace

def simple_mean(parameters):
    """Aggregates a list of parameters"""
    return np.mean(parameters, axis=0)

def clipped_average(parameters, clip_range):
    new_params = np.clip([parameters], clip_range[0], clip_range[1])
    return np.mean(new_params, axis=0)

def auc_roc_weighted_average(parameters, accuracies):
    auc_weights = accuracies / sum(accuracies)
    return sum(w * sum(x) for w, x in zip(auc_weights, parameters)) / sum(auc_weights) if sum(auc_weights) != 0 else 0


def differential_privacy_average(parameters, epsilon):
    """Applies differential privacy to a list of parameters"""
    return [laplace.rvs(loc=param, scale=1/epsilon) for param in parameters]

def validation_weighted_average(parameters, validation_acc):
    """Applies validation accuracy to a list of parameters"""
    return sum(w * sum(x) for w, x in zip(validation_acc, parameters)) / sum(validation_acc) if sum(validation_acc) != 0 else 0

def random_weighted_average(parameters):
    """Applies validation accuracy to a list of parameters"""
    # Make random weights that sum to 1
    random_weights = np.random.dirichlet(np.ones(len(parameters)), size=1)[0]
    return sum(w * sum(x) for w, x in zip(random_weights, parameters)) / sum(random_weights) if sum(random_weights) != 0 else 0

def random_parameter_select(parameters):
    # Select random element from parameters
    return parameters[np.random.randint(0, len(parameters))]


def proximal_operator(lam, parameters):
    """Applies proximal operator to a list of parameters"""
    return np.sign(parameters) * np.maximum(np.abs(parameters) - lam, 0)

def proximal_operator_weighted_average(parameters):
    avg_params = np.mean(parameters, axis=0)
    # Apply proximal_operator to avg_params
    weights = proximal_operator(0.1, avg_params)

    # Normalize weights
    weights = weights / np.sum(weights)
    return np.mean([w * x for w, x in zip(weights, parameters)], axis=0)
