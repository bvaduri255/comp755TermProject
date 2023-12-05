import os
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import metrics

from art.utils import load_mnist
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

# Fit new model to input features and labels, return model & model parameters
def fit_model(x_train, y_train):
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=64, epochs=1)

    l0_w = model.layers[0].get_weights()[0]
    l0_b  = model.layers[0].get_weights()[1]
    l2_w = model.layers[2].get_weights()[0]
    l2_b  = model.layers[2].get_weights()[1]
    l5_w = model.layers[5].get_weights()[0]
    l5_b  = model.layers[5].get_weights()[1]
    l6_w = model.layers[6].get_weights()[0]
    l6_b  = model.layers[6].get_weights()[1]
    return model, l0_w, l0_b, l2_w, l2_b, l5_w, l5_b, l6_w, l6_b


# Load parameter into model, return model
def load_model(l0_w, l0_b, l2_w, l2_b, l5_w, l5_b, l6_w, l6_b):
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

    # Train the model
    model.layers[0].set_weights([l0_w, l0_b])
    model.layers[2].set_weights([l2_w, l2_b])
    model.layers[5].set_weights([l5_w, l5_b])
    model.layers[6].set_weights([l6_w, l6_b])
    return model


# Predict on new data, return metrics (accuracy, confusion matrix, AUC)
def eval_model(model, x_test, y_test):
    predictions = model.predict(x_test, verbose=0)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    return accuracy, -1


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

    model, *params = fit_model(x_train, y_train)
    accuracy, auc = eval_model(model, x_test, y_test)
    print(f"Baseline model: Test Accuracy: {accuracy:.5f}, AUC: {auc:.5f}")

    # Load model faithfulness
    loaded_model = load_model(*params)
    accuracy, auc = eval_model(loaded_model, x_test, y_test)
    print(f"Loaded model: Test Accuracy: {accuracy:.5f}, AUC: {auc:.5f}")
