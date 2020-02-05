# disable CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# disable deprecation warnings
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json
import random

# hyperparameters
with open('config.JSON') as f:
	options = json.load(f)
	NUM_ROUNDS = options['NUM_ROUNDS']  # total num of aggregations (global rounds)
	NUM_CLIENTS = options['NUM_CLIENTS']  # per round (client batches)
	NUM_EPOCHS = options['NUM_EPOCHS']  # for client model
	BATCH_SIZE = options['BATCH_SIZE']  # for client model
	SHUFFLE_BUFFER = options['SHUFFLE_BUFFER']

# prep environment
warnings.simplefilter('ignore')
tf.compat.v1.enable_v2_behavior()
np.random.seed(0)
tff.framework.set_default_executor(tff.framework.create_local_executor())
print("Running",NUM_ROUNDS,"rounds of",NUM_CLIENTS,"clients each...")

# load MNIST dataset, variables are tff.simulation.ClientData objects
# (emnist_train, emnist_test) = tff.simulation.datasets.emnist.load_data()
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# create sample batch for Keras model wrapper
# do preprocessing here
# TODO: add .map and function parameter back in?
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
# sample_batch = tf.nest.map_structure(lambda x: x.numpy(), iter(dataset).next())
# TODO: map_structure syntax? x.numpy()?

# simple model with Keras
def create_compiled_keras_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(input_shape=(28, 28)),
		tf.keras.layers.Dense(
      10, activation=tf.nn.softmax, kernel_initializer='zeros')
		])
		# model from TF beginner tutorial, lower accuracy
		# tf.keras.layers.Dense(128, activation='relu'),
		# tf.keras.layers.Dropout(0.2),
		# tf.keras.layers.Dense(10, activation='softmax')
		# ])
	
	model.compile(
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()
		])
	return model

# let TFF wrap compiled Keras model
# TODO: why do we need a sample batch here?
def model_fn():
	keras_model = create_compiled_keras_model()
	return tff.learning.from_compiled_keras_model(keras_model, dataset)

# let TFF construct a Federated Averaging algorithm 
iterative_process = tff.learning.build_federated_averaging_process(model_fn)

# TODO: modify to take any number of datasets
dataset_list = []
dataset_list.append(dataset)
client_list = [0]

# shuffle client ids for "random sampling" of clients
random.shuffle(client_list)

# construct the server state
state = iterative_process.initialize()

# run server training rounds
# won't necessarily complete a "federated epoch"
for round_num in range(1, NUM_ROUNDS + 1):

	# pull client groups in order from shuffled client ids ("random sampling")
	start = (round_num - 1) * NUM_CLIENTS
	end = (round_num * NUM_CLIENTS) % len(client_list)
	if end > start:
		sample_clients = client_list[start:end]
	else:  # loop around
		sample_clients = client_list[start:len(client_list)]
		sample_clients.extend(client_list[0:end])

	# construct a list of datasets from the given set of users 
	# as an input to a round of training or evaluation
	def make_federated_data(client_data, client_ids):
		return [dataset_list[x] for x in client_ids]

	# make dataset for current client group
	federated_train_data = make_federated_data(dataset_list, sample_clients)

	# single round of Federated Averaging
	# passes federated_train_data: a list of tf.data.Dataset, one per client
	state, metrics = iterative_process.next(state, federated_train_data)
	print('round {:2d}, metrics={}'.format(round_num, metrics))

	# single round consists of: 
	# pushing the server state (including the model parameters) to the clients
	# on-device training on their local data
	# collecting and averaging model updates
	# producing a new updated model at the server