# disables CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json

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

# load MNIST dataset, variables are tff.simulation.ClientData objects
(emnist_train, emnist_test) = tff.simulation.datasets.emnist.load_data()

# preprocessing for individual client data
def preprocess(dataset):

	# flatten
	# renames the features from pixels and label to x and y for use with Keras
	def element_fn(element):
		return collections.OrderedDict([
			('x', tf.reshape(element['pixels'], [-1])), 
			('y', tf.reshape(element['label'], [1])),
		])
	
	# shuffle the individual examples and organize them into batches
	# repeat over the data set to run several epochs
	return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
	
# creates a new tf.data.Dataset containing the client[0] training examples
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
# create sample batch for Keras model wrapper
preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(), iter(preprocessed_example_dataset).next())
# TODO: map_structure syntax? x.numpy()?

# simple model with Keras
def create_compiled_keras_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(
      10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))
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
	return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

# let TFF construct a Federated Averaging algorithm 
iterative_process = tff.learning.build_federated_averaging_process(model_fn)

# shuffle client ids for "random sampling" of clients
# TODO

# construct the server state
state = iterative_process.initialize()

# run server training rounds
for round_num in range(1, NUM_ROUNDS):

	# pull client groups in order from shuffled client ids ("random sampling")
	start = (round_num - 1) * NUM_CLIENTS
	end = (round_num * NUM_CLIENTS) % len(emnist_train.client_ids)
	sample_clients = emnist_train.client_ids[start:end]

	# construct a list of datasets from the given set of users 
	# as an input to a round of training or evaluation
	def make_federated_data(client_data, client_ids):
		return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

	# make dataset for current client group
	federated_train_data = make_federated_data(emnist_train, sample_clients)

	# single round of Federated Averaging
	state, metrics = iterative_process.next(state, federated_train_data)
	print('round {:2d}, metrics={}'.format(round_num, metrics))

	# single round consists of: 
	# pushing the server state (including the model parameters) to the clients
	# on-device training on their local data
	# collecting and averaging model updates
	# producing a new updated model at the server