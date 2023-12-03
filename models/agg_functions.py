import numpy as np

def simple_mean(parameters):
    """Aggregates a list of parameters"""
    return np.mean(parameters, axis=0)