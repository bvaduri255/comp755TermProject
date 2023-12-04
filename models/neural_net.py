from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import metrics

import os
import numpy as np
from pathlib import Path

INPUT_DIM = 13
OUTPUT_DIM = 2

# Fit new model to input features and labels, return model & model parameters
def fit_model(data):
    features, labels = data[:, :-1], data[:, -1:]
    labels = to_categorical(labels, num_classes=len(np.unique(labels)))

    model = Sequential()
    model.add(Dense(10, input_dim=INPUT_DIM, activation='relu')) 
    model.add(Dense(10, activation='relu')) 
    model.add(Dense(10, activation='relu'))
    model.add(Dense(OUTPUT_DIM, activation='softmax')) 

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC()])

    # Train the model
    model.fit(features, labels, epochs=100, batch_size=64, validation_split=0.2, verbose=0)

    l1_w = model.layers[0].get_weights()[0]
    l1_b  = model.layers[0].get_weights()[1]
    l2_w = model.layers[1].get_weights()[0]
    l2_b  = model.layers[1].get_weights()[1]
    l3_w = model.layers[2].get_weights()[0]
    l3_b  = model.layers[2].get_weights()[1]
    l4_w = model.layers[3].get_weights()[0]
    l4_b  = model.layers[3].get_weights()[1]
    return model, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b, l4_w, l4_b


# Load parameter into model, return model
def load_model(l1_w, l1_b, l2_w, l2_b, l3_w, l3_b, l4_w, l4_b):
    model = Sequential()
    model.add(Dense(10, input_dim=INPUT_DIM, activation='relu')) 
    model.add(Dense(10, activation='relu')) 
    model.add(Dense(10, activation='relu'))
    model.add(Dense(OUTPUT_DIM, activation='softmax')) 

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC()])

    # Train the model
    model.layers[0].set_weights([l1_w, l1_b])
    model.layers[1].set_weights([l2_w, l2_b])
    model.layers[2].set_weights([l3_w, l3_b])
    model.layers[3].set_weights([l4_w, l4_b])
    return model


# Predict on new data, return metrics (accuracy, confusion matrix, AUC)
def eval_model(model, test_data):
    features, labels = test_data[:, :-1], test_data[:, -1:]
    labels = to_categorical(labels, num_classes=len(np.unique(labels)))
    loss, accuracy, auc = model.evaluate(features, labels)   
    return accuracy, auc


if __name__ == "__main__":
    from run_model import read_data, combine_split_datasets
    DATA_FOLDER = "heart_data"

    train_splits, eval_splits, test_splits = read_data(DATA_FOLDER)
    full_train, full_eval, full_test = combine_split_datasets(train_splits, eval_splits, test_splits)

    X_train, y_train = full_train[:, :-1], full_train[:, -1:]
    X_test, y_test = full_test[:, :-1], full_test[:, -1:]

    print(f"Input dim: {X_train.shape[1]}, output dim: {len(np.unique(y_train))}")

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    model, *params = fit_model(full_train)
    accuracy, auc = eval_model(model, full_test)
    print(f"Baseline model: Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    # Load model faithfulness
    loaded_model = load_model(*params)
    accuracy, auc = eval_model(loaded_model, full_test)
    print(f"Loaded model: Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")