import sys
import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json
import random
import math
import time
from datetime import datetime
import csv

# disable warnings
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Partitioner:
	# call member functions in order, partitioning data before build_model()

	def __init__(self):
		self.ROUND_LIMIT = 50
		self.SHUFFLE_BUFFER = 1000
		self.COHORT_SIZE = 1
		self.MAX_FANOUT = 1
		self.NUM_EPOCHS = 1
		self.BATCH_SIZE = 1
		self.SHUFFLE_SEED = 0
		self.LR = 0.1
		self.TARGET = 50
		self.TEST_PERIOD = 1
		self.verbose = False
		self.iterative_process = None
		self.sample_batch = None

		# partitioner data
		self.CLIENTS = 1
		self.SHARDS = 2
		self.NUMDATAPTS_MEAN = 600  
		self.NUMDATAPTS_STDEV = 0

		# each client partially iid
		self.PERCENT_DATA_IID = 100

		# some clients iid
		self.PERCENT_CLIENTS_IID = 0

		# dataset data
		self.LABELS = 10  # number of labels in y set

		# only variable that needs to be modified by inherited classes
		self.dataset_list = []

		# time the entire script
		self.TIC = time.perf_counter()

		# random number generators
		self.RNG1 = np.random.default_rng()
		self.RNG2 = np.random.default_rng()

	# parse config file
	def prep(self):
		# hyperparameters
		with open('config.JSON') as f:
			options = json.load(f)
			self.COHORT_SIZE = math.ceil(options['model']['COHORT_SIZE'])  # per round (client batches)
			self.MAX_FANOUT = math.ceil(options['system']['MAX_THREADS'])  # controlls multi-threading
			self.NUM_EPOCHS = math.ceil(options['model']['NUM_LOCAL_EPOCHS'])  # for client model
			self.BATCH_SIZE = math.ceil(options['model']['LOCAL_BATCH_SIZE'])  # for client model
			self.SHUFFLE_SEED = math.ceil(options['model']['SHUFFLE_SEED'])
			self.LR = options['model']['LEARNING_RATE']  # SGD learning rate
			self.TARGET = options['model']['TARGET_ACCURACY'] # target accuracy for model when tested with test set
			self.TEST_PERIOD = options['model']['ROUNDS_BETWEEN_TESTS'] # number of rounds between testset evaluation
			self.verbose = options['system']['VERBOSE']  

			# partitioner
			self.CLIENTS = math.ceil(options['partitioner']['NUM_CLIENTS'])  # number of clients to partition to
			self.SHARDS = math.ceil(options['partitioner']['NUM_CLASSES_PER']) # number of shards per client
			self.NUMDATAPTS_MEAN = options['partitioner']['MEAN_NUM_DATA_PTS_PER_CLIENT']
			self.NUMDATAPTS_STDEV = options['partitioner']['STD_DEV_NUM_DATA_PTS_PER_CLIENT'] 

			# each client partially iid
			self.PERCENT_DATA_IID = options['each_client_partially_iid']['PERCENT_DATA_IID']
			# some clients iid
			self.PERCENT_CLIENTS_IID = options['some_clients_iid']['PERCENT_CLIENTS_IID']

		# prep environment
		warnings.simplefilter('ignore')
		tf.compat.v1.enable_v2_behavior()
		if self.MAX_FANOUT < 1:  # standard multi-threading
			tff.framework.set_default_executor(tff.framework.create_local_executor())
		elif self.MAX_FANOUT == 1:  # single thread
			tff.framework.set_default_executor(None)
		else:
			tff.framework.set_default_executor(tff.framework.create_local_executor(self.COHORT_SIZE, self.MAX_FANOUT))

	# process test number command line parameter as hyperparameter values
	def test_num(self, n):
		if n > 0: # if test number is 0, use only config file values
			n = n - 1  # make working with array indices easier

			# construct value array
			# learning rate chosen/iterates first, batch size second, ...
			shuffle_seed = list(range(1,11))
			percent_data_iid = [40]  # schema 1
			percent_clients_iid = [50]  # schema 2
			cohort_size = [5, 10, 15, 20, 30] 
			num_epochs = [1] 
			batch_size = [50]
			learning_rate = [0.1]

			# convert test number to array indices and set constants to array values
			self.SHUFFLE_SEED = shuffle_seed[n // (len(percent_data_iid) * len(percent_clients_iid) * len(cohort_size) * len(num_epochs) * len(batch_size) * len(learning_rate))]
			n = n % (len(percent_data_iid) * len(percent_clients_iid) * len(cohort_size) * len(num_epochs) * len(batch_size) * len(learning_rate))
			self.PERCENT_DATA_IID = percent_data_iid[n // (len(percent_clients_iid) * len(cohort_size) * len(num_epochs) * len(batch_size) * len(learning_rate))]
			n = n % (len(percent_clients_iid) * len(cohort_size) * len(num_epochs) * len(batch_size) * len(learning_rate))
			self.PERCENT_CLIENTS_IID = percent_clients_iid[n // (len(cohort_size) * len(num_epochs) * len(batch_size) * len(learning_rate))]
			n = n % (len(cohort_size) * len(num_epochs) * len(batch_size) * len(learning_rate))
			self.COHORT_SIZE = cohort_size[n // (len(num_epochs) * len(batch_size) * len(learning_rate))]
			n = n % (len(num_epochs) * len(batch_size) * len(learning_rate))
			self.NUM_EPOCHS = num_epochs[n // (len(batch_size) * len(learning_rate))]
			n = n % (len(batch_size) * len(learning_rate))
			self.BATCH_SIZE = batch_size[n // len(learning_rate)]
			self.LR = learning_rate[n % len(learning_rate)]

			# set learning rate based on percent IID
			# 20% = 0.1 LR, 100% = 0.2 LR
			# self.LR = (float(self.PERCENT_DATA_IID) / 800) + 0.075

			# set learning rate based on percent data IID
			# if self.PERCENT_DATA_IID < 30:
			# 	self.LR = 0.1

		# set numpy shuffle seed for random generator objects
		# multiply seed to be large enough to be effective
		self.RNG1 = np.random.default_rng(self.SHUFFLE_SEED * 123456789) # partitioning data into clients (files 1-4)
		self.RNG2 = np.random.default_rng(self.SHUFFLE_SEED * 987654321) # selection of clients

	# output configuation data to csv file
	def make_config_csv(self, test, batch):
		filename = 'results/' + str(batch) + '/' + str(batch) + '.' + str(test) + '.config.csv'
		with open(filename, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(['COHORT_SIZE', 'NUM_LOCAL_EPOCHS', 'LOCAL_BATCH_SIZE', 'SHUFFLE_SEED', 
				'LEARNING_RATE', 'TARGET_ACCURACY', 'ROUNDS_BETWEEN_TESTS', 'NUM_CLIENTS', 'NUM_CLASSES_PER', 
				'MEAN_NUM_DATA_PTS_PER_CLIENT', 'STD_DEV_NUM_DATA_PTS_PER_CLIENT', 'PERCENT_DATA_IID', 
				'PERCENT_CLIENTS_IID','MAX_THREADS'])
			writer.writerow([self.COHORT_SIZE, self.NUM_EPOCHS, self.BATCH_SIZE, self.SHUFFLE_SEED,
				self.LR, self.TARGET, self.TEST_PERIOD, self.CLIENTS, self.SHARDS,
				self.NUMDATAPTS_MEAN, self.NUMDATAPTS_STDEV, self.PERCENT_DATA_IID,
				self.PERCENT_CLIENTS_IID, self.MAX_FANOUT])

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

		# create sample batch (all data) for Keras model wrapper
		# note: sample batch is different data type than dataset used in iterative process
		self.sample_batch = tf.nest.map_structure(lambda x: x.numpy(), iter(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(60000, seed = self.SHUFFLE_SEED * 123456789, reshuffle_each_iteration=True)).next())

		return (x_train, y_train)

	# compile model
	def build_model(self):
		# let TFF wrap compiled Keras model
		def model_fn():
			keras_model = self.create_compiled_keras_model()
			return tff.learning.from_compiled_keras_model(keras_model, self.sample_batch)

		# let TFF construct a Federated Averaging algorithm 
		self.iterative_process = tff.learning.build_federated_averaging_process(model_fn)

	# run federated training algorithm
	def train(self, test, batch, schema_num):
		# load test dataset
		mnist = tf.keras.datasets.mnist
		(x_trash, y_trash), (x_test, y_test) = mnist.load_data()
		x_test = x_test / 255.0
		x_test = np.reshape(np.float64(x_test), (10000,28,28,1))
		y_test = np.float64(y_test)

		# preprocess test dataset
		testset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
		processed_testset = testset.batch(self.BATCH_SIZE).shuffle(10000, seed = self.SHUFFLE_SEED * 123456789, reshuffle_each_iteration=True)
		model = self.create_compiled_keras_model()

		# print(model.count_params())
		# print(model.summary())

		# construct the server state
		state = self.iterative_process.initialize()

		# construct a list of datasets from the given set of users 
		# as an input to a round of training or evaluation
		def make_federated_data(client_data, client_ids):
			return [self.dataset_list[x] for x in client_ids]

		# output to CSV file
		filename = 'results/' + str(batch) + '/' + str(batch) + '.' + str(test) + '.s' + str(schema_num) + 'out.csv'
		with open(filename, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(['ROUND_NUM', 'ROUND_START', 'SPARSE_CATEGORICAL_ACCURACY_TRAIN', 'SPARSE_CATEGORICAL_CROSSENTROPY_LOSS_TRAIN', 
				'SPARSE_CATEGORICAL_ACCURACY_TEST', 'SPARSE_CATEGORICAL_CROSSENTROPY_LOSS_TEST', 'COMPLETION_TIME_SECONDS'])

			# run server training rounds
			# won't necessarily complete a "federated epoch"
			below_target = True
			under_limit = True
			round_num = 0
			time_sum = 0
			while below_target and under_limit:
				round_num = round_num + 1
				tic = time.perf_counter()
				start_time = datetime.now()

				# shuffle client ids for "random sampling" of clients
				self.RNG2 = np.random.default_rng(self.SHUFFLE_SEED * 987654321 + round_num) # seed changes with round
				client_list = self.RNG2.permutation(len(self.dataset_list))

				# pull clients from shuffled client ids ("random sampling")
				sample_clients = client_list[:self.COHORT_SIZE]

				# set new shuffle seed for each client dataset
				# determined by seed, round number, and client number
				for i in sample_clients:
					s = (self.SHUFFLE_SEED * 987654321) + (round_num * 12345) + i
					self.dataset_list[i].shuffle(self.SHUFFLE_BUFFER, seed = s, reshuffle_each_iteration=True)
				# 	print(i)
				# 	print(list(self.dataset_list[i].as_numpy_iterator()))
				# sys.exit()

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

					# store relevant metrics in CSV
					# 'ROUND_NUM', 'ROUND_START', 'SPARSE_CATEGORICAL_ACCURACY_TRAIN', 'SPARSE_CATEGORICAL_CROSSENTROPY_LOSS_TRAIN', 
					# 'SPARSE_CATEGORICAL_ACCURACY_TEST', 'SPARSE_CATEGORICAL_CROSSENTROPY_LOSS_TEST', 'COMPLETION_TIME_SECONDS'
					writer.writerow([round_num, start_time, metrics[0], metrics[1], accuracy, loss, toc - tic])

					# set continuation bool
					if accuracy >= (self.TARGET / 100):
						below_target = False

				else:
					# store relevant metrics in CSV
					# 'ROUND_NUM', 'ROUND_START', 'SPARSE_CATEGORICAL_ACCURACY_TRAIN', 'SPARSE_CATEGORICAL_CROSSENTROPY_LOSS_TRAIN', 
					# 'SPARSE_CATEGORICAL_ACCURACY_TEST', 'SPARSE_CATEGORICAL_CROSSENTROPY_LOSS_TEST', 'COMPLETION_TIME_SECONDS'
					writer.writerow([round_num, start_time, metrics[0], metrics[1], 'N/A', 'N/A', toc - tic])

				if self.verbose:
					print()
				
				if round_num >= self.ROUND_LIMIT:
					under_limit = False

		# print final test stats
		print("Target accuracy reached after ",round_num," rounds")
		print('Average time per round: {:0.2f}'.format(time_sum // round_num))
		print()

		# output final test stats to CSV
		filename = 'results/' + str(batch) + '/' + str(batch) + '.' + str(test) + '.s' + str(schema_num) + 'summary.csv'
		toc = time.perf_counter()
		with open(filename, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(['TOTAL_ROUNDS', 'AVERAGE_SECONDS_PER_ROUND', 'SCRIPT_TOTAL_SECONDS'])
			writer.writerow([round_num, time_sum // round_num, toc - self.TIC])

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