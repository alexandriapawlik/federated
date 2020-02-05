from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import json
import random

# disable CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# disable warnings
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# hyperparameters
with open('config.JSON') as f:
	options = json.load(f)
	NUM_EPOCHS = options['NUM_EPOCHS']  # for client model
	BATCH_SIZE = options['BATCH_SIZE']  # for client model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax', kernel_initializer='zeros')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
# model.evaluate(x_test,  y_test, verbose=2)