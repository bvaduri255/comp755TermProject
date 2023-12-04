from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

import os
import pandas as pd
import numpy as np
from pathlib import Path

def read_data(folder_name):
    """Given name of dataset folder, read all data from splits into lists"""
    train_splits, eval_splits, test_splits = [], [], []

    data_path = os.path.join(os.getcwd(), Path(folder_name))
    files = os.listdir(data_path)
    num_splits = len(files) // 3
    print(f"Calculated number of splits in dataset: {num_splits}")

    for split_i in range(num_splits):
        train_file = os.path.join(data_path, f"split{split_i}_train")
        eval_file = os.path.join(data_path, f"split{split_i}_eval")
        test_file = os.path.join(data_path, f"split{split_i}_test")

        train_split = np.genfromtxt(train_file, delimiter=",")
        eval_split = np.genfromtxt(eval_file, delimiter=",")
        test_split = np.genfromtxt(test_file, delimiter=",")

        train_splits.append(train_split)
        eval_splits.append(eval_split)
        test_splits.append(test_split)
    return train_splits, eval_splits, test_splits
        

def combine_split_datasets(train_splits, eval_splits, test_splits):
    train_combined = np.vstack(train_splits)
    eval_combined = np.vstack(eval_splits)
    test_combined = np.vstack(test_splits)
    return train_combined, eval_combined, test_combined


if __name__ == "__main__":
    DATA_FOLDER = "shopping_behavior"
    train_splits, eval_splits, test_splits = read_data(DATA_FOLDER)
    full_train, full_eval, full_test = combine_split_datasets(train_splits, eval_splits, test_splits)

    X_train, y_train = full_train[:, :-1], full_train[:, -1:]
    X_test, y_test = full_test[:, :-1], full_test[:, -1:]

    # Convert labels to one-hot encoding
    y_one_hot_train = to_categorical(y_train, num_classes=7)
    y_one_hot_test = to_categorical(y_test, num_classes=7)

    # Create a neural network model
    model = Sequential()
    model.add(Dense(10, input_dim=16, activation='relu'))  # First hidden layer with 5 units and ReLU activation
    model.add(Dense(10, activation='relu'))  # Second hidden layer with 5 units and ReLU activation
    model.add(Dense(10, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # Output layer with 7 units and softmax activation

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_one_hot_train, epochs=100, batch_size=64, validation_split=0.2, verbose=2)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_one_hot_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")