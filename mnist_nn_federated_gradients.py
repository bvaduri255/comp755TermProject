
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import metrics
from art.utils import load_mnist
from tqdm import tqdm

# Load dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Function to create a new model
def create_model():
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

# Function to compute gradients on a subset of data
def compute_gradients(model, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train)
        loss = categorical_crossentropy(y_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients

# Function to simulate multiple clients
def simulate_clients(num_clients, x_train, y_train):
    size = len(x_train) // num_clients
    clients = []
    for i in range(num_clients):
        start, end = i * size, (i + 1) * size
        client_data = (x_train[start:end], y_train[start:end])
        clients.append(client_data)
    return clients

# Function to aggregate gradients (averaging)
def aggregate_gradients(gradients):
    avg_gradients = [np.mean([client_gradients[layer] for client_gradients in gradients], axis=0) for layer in range(len(gradients[0]))]
    return avg_gradients

# Function to apply aggregated gradients to the model
def apply_gradients(model, gradients, learning_rate):
    optimizer = Adam(learning_rate=learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Simulating federated learning
num_clients = 5
clients = simulate_clients(num_clients, x_train, y_train)

# Initialize global model
global_model = create_model()

# Training rounds
for round in range(900):
    client_gradients = []
    for client_data in clients:
        client_model = create_model()
        client_model.set_weights(global_model.get_weights())
        gradients = compute_gradients(client_model, *client_data)
        client_gradients.append(gradients)
    
    # Aggregate gradients and update global model
    new_gradients = aggregate_gradients(client_gradients)
    apply_gradients(global_model, new_gradients, learning_rate=0.01)

# Evaluate the global model
global_model.evaluate(x_test, y_test)

x_training = x_train[:int(0.7*len(x_train)), :]
y_training = y_train[:int(0.7*len(y_train)), :]

x_evals = x_test[:int(0.7*len(x_test)), :]
y_evals = y_test[:int(0.7*len(y_test)), :]

art_train_attack_x = x_train[:len(x_train) - int(0.7*x_train), :]
art_eval_attack_x = x_test[:len(x_test) - int(0.7*x_test), :]
art_attack_x = np.vstack([art_train_attack_x, art_eval_attack_x])

art_train_attack_y = y_train[:len(y_train) - int(0.7*y_train), :]
art_eval_attack_y = y_test[:len(y_test) - int(0.7*y_test), :]
art_attack_y = np.vstack([art_train_attack_y, art_eval_attack_y])

model = KerasClassifier(model=global_model)
mi = MembershipInferenceBlackBox(model, attack_model_type="nn", input_type="prediction", nn_model_batch_size=128)
mi.fit(x_training, y_training, x_evals, y_evals)
values = mi.infer(art_attack_x, art_attack_y, probabilities=True)
print(values)
