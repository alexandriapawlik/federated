import partitioner

import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json
import random
import math

class Label_Partitioner(partitioner.Partitioner):

	def __init__(self):
		super().__init__()

	def go(self):
		# call parent functions
		self.prep()
		(x_train, y_train) = self.load_data()
		# TODO: randomize data
		client_data_x = [[] for i in range(5)] # list for each of five clients
		client_data_y = [[] for i in range(5)]

		# partition data and store
		for data_num in range(0, x_train.shape[0]):
			# sort each of 10 categories to 5 clients
			client_data_x[int(y_train[data_num] // 2)].append(x_train[data_num])
			client_data_y[int(y_train[data_num] // 2)].append(y_train[data_num])

		for client_num in range(5):
			dataset = tf.data.Dataset.from_tensor_slices((client_data_x[client_num], client_data_y[client_num]))
			self.dataset_list.append(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(self.SHUFFLE_BUFFER))

		# train
		self.build_model()
		self.train()