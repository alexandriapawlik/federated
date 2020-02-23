import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json
import random
import math

# disable CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# disable warnings
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Partitioner:
	# call member functions in order, partitioning data before build_model()

	def __init__(self):
		self.NUM_ROUNDS = 1
		self.COHORT_SIZE = 1
		self.MAX_FANOUT = 1
		self.NUM_EPOCHS = 1
		self.BATCH_SIZE = 1
		self.SHUFFLE_BUFFER = 0
		self.LEARNING_RATE = 0.0

		# partitioner data
		self.CLIENTS = 1
		self.SHARDS = 1

		# dataset data
		self.LABELS = 1

		self.iterative_process = None
		self.sample_batch = None

		# only variable that needs to be modified by inherited classes
		self.dataset_list = []

	# parse config file
	def prep(self):
		# hyperparameters
		# TODO: modify to take array of epoch/batch values to run series of tests
		with open('config.JSON') as f:
			options = json.load(f)
			self.NUM_ROUNDS = math.ceil(options['model']['NUM_GLOBAL_ROUNDS'])  # total num of aggregations
			self.COHORT_SIZE = math.ceil(options['model']['COHORT_SIZE'])  # per round (client batches)
			self.MAX_FANOUT = math.ceil(options['system']['MAX_THREADS'])  # controlls multi-threading
			self.NUM_EPOCHS = math.ceil(options['model']['NUM_LOCAL_EPOCHS'])  # for client model
			self.BATCH_SIZE = math.ceil(options['model']['LOCAL_BATCH_SIZE'])  # for client model
			self.SHUFFLE_BUFFER = math.ceil(options['model']['SHUFFLE_BUFFER'])
			self.LEARNING_RATE = options['model']['LEARNING_RATE']  # SGD learning rate
			self.CLIENTS = math.ceil(options['partitioner']['NUM_CLIENTS'])  # number of clients to partition to
			self.SHARDS = math.ceil(options['partitioner']['NUM_SHARDS_PER']) # number of shards per client
			self.LABELS = int(options['data']['NUM_LABELS'])  # number of labels in y set

		# prep environment
		warnings.simplefilter('ignore')
		tf.compat.v1.enable_v2_behavior()
		np.random.seed(0)
		if self.MAX_FANOUT < 1:  # standard multi-threading
			tff.framework.set_default_executor(tff.framework.create_local_executor())
		elif self.MAX_FANOUT == 1:  # single thread
			tff.framework.set_default_executor(None)
		else:
			tff.framework.set_default_executor(tff.framework.create_local_executor(self.COHORT_SIZE, self.MAX_FANOUT))

	# returns datasets ready for partitioning
	def load_data(self):
		# load MNIST dataset
		mnist = tf.keras.datasets.mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train, x_test = x_train / 255.0, x_test / 255.0
		x_train = np.float64(x_train)
		y_train = np.float64(y_train)

		# do preprocessing here
		dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

		# create sample batch for Keras model wrapper
		# note: sample batch is different data type than dataset used in iterative process
		self.sample_batch = tf.nest.map_structure(lambda x: x.numpy(), iter(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(self.SHUFFLE_BUFFER)).next())
		# TODO: what does Tensorflow do if passed a batch size larger than 

		return (x_train, y_train)

	# compile model
	def build_model(self):
		# simple model with Keras
		def create_compiled_keras_model():
			model = tf.keras.models.Sequential([
				tf.keras.layers.Flatten(input_shape=(28, 28)),
				tf.keras.layers.Dense(128, activation='relu'),
				tf.keras.layers.Dropout(0.2),
				tf.keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer='zeros')
				])
			
			model.compile(
				loss=tf.keras.losses.SparseCategoricalCrossentropy(),
				optimizer=tf.keras.optimizers.SGD(learning_rate=self.LEARNING_RATE),
				metrics=[tf.keras.metrics.SparseCategoricalAccuracy()
				])
			return model

		# let TFF wrap compiled Keras model
		def model_fn():
			keras_model = create_compiled_keras_model()
			return tff.learning.from_compiled_keras_model(keras_model, self.sample_batch)

		# let TFF construct a Federated Averaging algorithm 
		self.iterative_process = tff.learning.build_federated_averaging_process(model_fn)

	# run federated training algorithm
	def train(self):
		print("Running",self.NUM_ROUNDS,"rounds of",self.COHORT_SIZE,"clients each...")

		# shuffle client ids for "random sampling" of clients
		client_list = list(range(len(self.dataset_list)))
		random.shuffle(client_list)

		# construct the server state
		state = self.iterative_process.initialize()

		# run server training rounds
		# won't necessarily complete a "federated epoch"
		# TODO: set to run until a certain accuracy is reached
		for round_num in range(1, self.NUM_ROUNDS + 1):

			# pull client groups in order from shuffled client ids ("random sampling")
			start = ((round_num - 1) * self.COHORT_SIZE) % len(client_list)
			end = (round_num * self.COHORT_SIZE) % len(client_list)
			if end > start:
				sample_clients = client_list[start:end]
			else:  # loop around or entire list
				sample_clients = client_list[start:len(client_list)]
				sample_clients.extend(client_list[0:end])

			# construct a list of datasets from the given set of users 
			# as an input to a round of training or evaluation
			def make_federated_data(client_data, client_ids):
				return [self.dataset_list[x] for x in client_ids]

			# make dataset for current client group
			federated_train_data = make_federated_data(self.dataset_list, sample_clients)

			# single round of Federated Averaging
			# passes federated_train_data: a list of tf.data.Dataset, one per client
			state, metrics = self.iterative_process.next(state, federated_train_data)
			print('round {:2d}, metrics={}'.format(round_num, metrics))
			# TODO: mute metrics to only show last line