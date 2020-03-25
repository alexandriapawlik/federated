import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json
import random
import math
import time

# disable warnings
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Partitioner:
	# call member functions in order, partitioning data before build_model()

	def __init__(self):
		self.COHORT_SIZE = 1
		self.MAX_FANOUT = 1
		self.NUM_EPOCHS = 1
		self.BATCH_SIZE = 1
		self.SHUFFLE_BUFFER = 0
		self.LR = 0.1
		self.TARGET = 50
		self.TEST_PERIOD = 1

		# partitioner data
		self.CLIENTS = 1
		self.SHARDS = 1

		# dataset data
		self.LABELS = 1

		self.verbose = False
		self.iterative_process = None
		self.sample_batch = None

		# only variable that needs to be modified by inherited classes
		self.dataset_list = []

	# parse config file
	def prep(self):
		# hyperparameters
		with open('config.JSON') as f:
			options = json.load(f)
			self.COHORT_SIZE = math.ceil(options['model']['COHORT_SIZE'])  # per round (client batches)
			self.MAX_FANOUT = math.ceil(options['system']['MAX_THREADS'])  # controlls multi-threading
			self.NUM_EPOCHS = math.ceil(options['model']['NUM_LOCAL_EPOCHS'])  # for client model
			self.BATCH_SIZE = math.ceil(options['model']['LOCAL_BATCH_SIZE'])  # for client model
			self.SHUFFLE_BUFFER = math.ceil(options['model']['SHUFFLE_BUFFER'])
			self.LR = options['model']['LEARNING_RATE']  # SGD learning rate
			self.TARGET = options['model']['TARGET_ACCURACY'] # target accuracy for model when tested with test set
			self.TEST_PERIOD = options['model']['ROUNDS_BETWEEN_TESTS'] # number of rounds between testset evaluation
			self.CLIENTS = math.ceil(options['partitioner']['NUM_CLIENTS'])  # number of clients to partition to
			self.SHARDS = math.ceil(options['partitioner']['NUM_SHARDS_PER']) # number of shards per client
			self.LABELS = int(options['data']['NUM_LABELS'])  # number of labels in y set
			self.verbose = options['system']['VERBOSE']  

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

	# process test number command line parameter as hyperparameter values
	def test_num(self, n):
		print("Test ", n)
		n = n - 1  # make working with array indices easier

		# construct value array
		cohort_size = [10] 
		num_epochs = [20]
		batch_size = [10]
		learning_rate = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22]

		# convert test number to array indices and set constants to array values
		self.COHORT_SIZE = cohort_size[n // (len(num_epochs) * len(batch_size) * len(learning_rate))]
		n = n % (len(num_epochs) * len(batch_size) * len(learning_rate))
		self.NUM_EPOCHS = num_epochs[n // (len(batch_size) * len(learning_rate))]
		n = n % (len(batch_size) * len(learning_rate))
		self.BATCH_SIZE = batch_size[n // len(learning_rate)]
		self.LR = learning_rate[n % len(learning_rate)]

		# test specs
		print("cohort size: ",self.COHORT_SIZE)
		print("number of local epochs: ",self.NUM_EPOCHS)
		print("local batch size: ", self.BATCH_SIZE)
		print("learning rate: ", self.LR)
		print("Sampling",self.COHORT_SIZE,"clients per round until",self.TARGET,"%","accuracy...")
		print()

	# returns datasets ready for partitioning
	def load_data(self):
		# load MNIST dataset
		mnist = tf.keras.datasets.mnist
		(x_train, y_train), (x_trash, y_trash) = mnist.load_data()
		x_train = x_train / 255.0
		x_train = np.reshape(np.float64(x_train), (60000,28,28,1))
		y_train = np.float64(y_train)

		# do preprocessing here
		dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

		# create sample batch for Keras model wrapper
		# note: sample batch is different data type than dataset used in iterative process
		self.sample_batch = tf.nest.map_structure(lambda x: x.numpy(), iter(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(self.SHUFFLE_BUFFER)).next())

		return (x_train, y_train)

	# compile model
	def build_model(self):
		# let TFF wrap compiled Keras model
		def model_fn():
			keras_model = self.create_compiled_keras_model()
			return tff.learning.from_compiled_keras_model(keras_model, self.sample_batch)

		# let TFF construct a Federated Averaging algorithm 
		self.iterative_process = tff.learning.build_federated_averaging_process(model_fn)
		# TODO: build second model with different learning rate for non-IID

	# run federated training algorithm
	def train(self):
		# load test dataset
		mnist = tf.keras.datasets.mnist
		(x_trash, y_trash), (x_test, y_test) = mnist.load_data()
		x_test = x_test / 255.0
		x_test = np.reshape(np.float64(x_test), (10000,28,28,1))
		y_test = np.float64(y_test)

		# preprocess test dataset
		testset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
		processed_testset = testset.batch(self.BATCH_SIZE).shuffle(self.SHUFFLE_BUFFER)
		model = self.create_compiled_keras_model()

		# print(model.count_params())
		# print(model.summary())

		# shuffle client ids for "random sampling" of clients
		client_list = list(range(len(self.dataset_list)))
		random.shuffle(client_list)

		# construct the server state
		state = self.iterative_process.initialize()

		# construct a list of datasets from the given set of users 
		# as an input to a round of training or evaluation
		def make_federated_data(client_data, client_ids):
			return [self.dataset_list[x] for x in client_ids]

		# run server training rounds
		# won't necessarily complete a "federated epoch"
		below_target = True
		round_num = 0
		time_sum = 0
		while below_target:
			round_num = round_num + 1
			tic = time.perf_counter()

			# pull client groups in order from shuffled client ids ("random sampling")
			start = ((round_num - 1) * self.COHORT_SIZE) % len(client_list)
			end = (round_num * self.COHORT_SIZE) % len(client_list)
			if end > start:
				sample_clients = client_list[start:end]
			else:  # loop around or entire list
				sample_clients = client_list[start:len(client_list)]
				sample_clients.extend(client_list[0:end])

			# make dataset for current client group
			federated_train_data = make_federated_data(self.dataset_list, sample_clients)

			# single round of Federated Averaging
			# passes federated_train_data: a list of tf.data.Dataset, one per client
			state, metrics = self.iterative_process.next(state, federated_train_data)

			# print relevant metrics
			toc = time.perf_counter()
			time_sum = time_sum + toc - tic
			if self.verbose:
				print('round {:2d}, metrics={}'.format(round_num, metrics))
				print('{:0.4f} seconds'.format(toc - tic))
			
			# run test set every so often and stop if we've reached a target accuracy
			if round_num % self.TEST_PERIOD == 0:
				# test model, run same number of epochs as in training set
				tff.learning.assign_weights_to_keras_model(model, state.model)
				loss, accuracy = model.evaluate(processed_testset, steps=self.NUM_EPOCHS, verbose=0)
				if self.verbose:
					print('Tested. Sparse categorical accuracy: {:0.2f}'.format(accuracy * 100))

				# set continuation bool
				if accuracy >= (self.TARGET / 100):
					below_target = False
			
			if self.verbose:
				print()

		# print final test stats
		print("Target accuracy reached after ",round_num," rounds")
		print('Average time per round: {:0.2f}'.format(time_sum // round_num))
		print()


	# simple model with Keras
	# internal method
	def create_compiled_keras_model(self):
		model = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(32, (5,5), padding="same", activation='relu', input_shape=(28,28,1)),
			tf.keras.layers.MaxPool2D((2,2)),
			tf.keras.layers.Conv2D(64, (5,5), padding="same", activation='relu'),
			tf.keras.layers.MaxPool2D((2,2)),
			tf.keras.layers.Flatten(input_shape=(7,7)),
			tf.keras.layers.Dense(512, activation='relu'),
			tf.keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer='zeros')
			])
		
		model.compile(
			loss=tf.keras.losses.SparseCategoricalCrossentropy(),
			optimizer=tf.keras.optimizers.SGD(learning_rate=self.LR),
			metrics=[tf.keras.metrics.SparseCategoricalAccuracy()
			])
		return model	