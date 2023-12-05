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
    # Make random weights
    random_weights = np.random.rand(len(parameters))
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
    return np.mean([w * x for w, x in zip(weights, parameters)], axis=0)


accuracies= np.abs(predicitions - np.vstack(y_train_attack, y_test_attack))

percent = (1- np.mean(accuracies))*100

prediciton_squared = (np.multiply(predictions, predictions))

0,2,5,6

import matplotlib.pyplot as plt

def plot_accuracy_privacy_graph(accuracy, privacy, labels):
    """
    Plots a graph with accuracy on the x-axis and privacy on the y-axis.

    :param accuracy: List of accuracy values.
    :param privacy: List of privacy values.
    :param labels: List of labels for each point.
    """
    fig, ax = plt.subplots()

    # Plot each point with its label
    for i in range(len(accuracy)):
        ax.scatter(accuracy[i], privacy[i])
        ax.text(accuracy[i], privacy[i], labels[i], fontsize=9, ha='right')

    # Adding 50% dotted lines to form quadrants
    ax.axhline(y=50, color='gray', linestyle='--')
    ax.axvline(x=50, color='gray', linestyle='--')

    # Setting the range of axes from 0% to 100%
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Labels for axes
    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Privacy (%)')

    # Title of the graph
    ax.set_title('Accuracy vs Privacy Graph')

    plt.show()

# Example data
accuracy = [20, 40, 70, 90]
privacy = [80, 60, 30, 10]
labels = ['A', 'B', 'C', 'D']

# Plotting the graph
plot_accuracy_privacy_graph(accuracy, privacy, labels)
