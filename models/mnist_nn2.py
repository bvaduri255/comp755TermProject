import numpy as np
import tensorflow as tf
tf.random.set_seed(123)

from keras.models import Sequential
from keras.optimizers.legacy import Adam
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import metrics

from art.utils import load_mnist
from art.estimators.classification import KerasClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

from tqdm import tqdm
from agg_functions import * 

def split_data(data, num_splits):
    N = len(data)
    split_size = N // num_splits
    last_split_size = split_size + N % num_splits
    splits = []

    for i in range(num_splits-1):
        start_idx, end_idx = i, i+split_size
        current_split = data[start_idx:end_idx, :]
        splits.append(current_split)
    splits.append(data[-last_split_size:, :])
    return splits

def create_model():
    """Returns a Keras model."""
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[metrics.categorical_accuracy])
    return model

def compute_gradients(model, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train)
        loss = categorical_crossentropy(y_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients

# Function to aggregate gradients (averaging)
def aggregate_gradients(gradients):
    avg_gradients = [np.mean([client_gradients[layer] for client_gradients in gradients], axis=0) for layer in range(len(gradients[0]))]
    return avg_gradients

def simple_mean(arr):
    return np.mean(arr, axis=0)

def aggregate_gradients2(gradients, agg_func, support=None):
    num_layers = len(gradients[0])
    agg_grads = []
    for layer_i in range(num_layers):
        layer_grads = [split[layer_i] for split in gradients]
        agg_layer_grad = agg_func(layer_grads, support)
        agg_grads.append(agg_layer_grad)
    return agg_grads

# Function to apply aggregated gradients to the model
def apply_gradients(model, gradients, learning_rate):
    optimizer = Adam(learning_rate=learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Predict on new data, return metrics (accuracy, confusion matrix, AUC)
def eval_model(model, x_test, y_test):
    predictions = model.predict(x_test, verbose=0)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    return accuracy


if __name__ == "__main__":

    """Load MNIST dataset"""
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    # Split MNIST
    num_splits = 3
    x_train_splits = split_data(x_train, num_splits)
    y_train_splits = split_data(y_train, num_splits)
    x_test_splits = split_data(x_test, num_splits)
    y_test_splits = split_data(y_test, num_splits)

    # Create four nodes
    split_models = [create_model() for _ in range(num_splits)]

    # Create global model
    global_model = create_model()

    # Run a round
    num_rounds = 5
    for round in tqdm(range(num_rounds)):
        split_gradients = []
        for idx, model in enumerate(split_models):
            x_train = x_train_splits[idx]
            y_train = y_train_splits[idx]

            # Set the weights for client model to global model
            model.set_weights(global_model.get_weights())

            # Get gradient for split model
            gradient = compute_gradients(model, x_train, y_train)

            # Apply gradient on split model
            apply_gradients(model, gradient, learning_rate=0.01)

            # Save gradient into list
            split_gradients.append(gradient)

        # Aggregate gradients 
        global_gradient = aggregate_gradients(split_gradients, agg_func=simple_mean, support=None)

        # Apply to global model
        apply_gradients(global_model, global_gradient, learning_rate=0.01)

    # Evaluate each split model on its own dataset
    for idx, model in enumerate(split_models):
        print(x_train.shape)
        print(y_train.shape)
        x_test = x_test_splits[idx]
        y_test = y_test_splits[idx]
        print(f"Split {idx}:")
        model.evaluate(x_test, y_test)

    # Evaluate global model in full test dataset
    print(f"Global model:")
    global_model.evaluate(x_test, y_test)

