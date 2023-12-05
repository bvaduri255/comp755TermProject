import matplotlib.pyplot as plt
import numpy as np


accuracies= np.abs(predicitions - np.vstack(y_train_attack, y_test_attack))

percent = (1- np.mean(accuracies))*100

prediciton_squared = (np.multiply(predictions, predictions))




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
labels = ['Average Aggregation', 'Accuracy Weighted Aggregation', 'Proximal Operator Weighted Average ', 'Differential Privacy', 'Random Weights', 'Base Model (No Aggregation)', 'Random Selection' ]

# Plotting the graph
plot_accuracy_privacy_graph(accuracy, privacy, labels)
