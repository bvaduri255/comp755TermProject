"""
The script demonstrates a simple example of using ART with Keras. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from keras.models import Sequential
from keras.optimizers.legacy import Adam
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from art.utils import load_mnist
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

NUM_CLASSES = len(np.unique(y_train))

# Step 2: Create the model
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

# Step 3: Create the ART classifier
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

# Step 4: Train the ART classifier
# classifier.fit(x_train, y_train, batch_size=64, nb_epochs=1) # TODO change back

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
# (x_train, y_train), (x_test, y_test)
x_train = x_train[:300]
y_train = y_train[:300]
x_test = x_test[:300]
y_test = y_test[:300]

x_train_fit, x_train_attack = x_train[:-100], x_train[-100:]
y_train_fit, y_train_attack = y_train[:-100], y_train[-100:]

x_test_fit, x_test_attack = x_test[:-100], x_test[-100:]
y_test_fit, y_test_attack = y_test[:-100], y_test[-100:]

x_attack = np.vstack([x_train_attack, x_test_attack])
y_attack = np.vstack([y_train_attack, y_test_attack])


mi = MembershipInferenceBlackBox(classifier, attack_model_type="nn", input_type="prediction", nn_model_batch_size=128)
mi.fit(x_train_fit, y_train_fit, x_test_fit, y_test_fit)
predictions = mi.infer(x_attack, y_attack, probabilities=True)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# print(predictions)
# print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

keras_model = classifier.model
# print(type(keras_model))
# print(keras_model.summary())

for idx, layer in enumerate(keras_model.layers): 
    print(f"Layer: {idx}")
    print(layer.get_config())