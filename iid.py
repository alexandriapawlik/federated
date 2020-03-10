import partitioner

import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json
import random
import math

class IID_Partitioner(partitioner.Partitioner):

	def __init__(self):
		super().__init__()

	def go(self, num):
		# call parent functions
		self.prep()
		self.test_num(num)
		(x_train, y_train) = self.load_data()
		data_per_client = x_train.shape[0] // self.CLIENTS

		# randomize data (generate array of random permutation of indices)
		indices = np.random.permutation(x_train.shape[0])

		# partition data and store in class
		for client_num in range(self.CLIENTS):
			# calculate indices of slice
			start = client_num * data_per_client
			end = (client_num + 1) * data_per_client 
			# slice indices for single client
			x_indices = indices[start:end]
			y_indices = indices[start:end]
			# slice data for single client
			dataset = tf.data.Dataset.from_tensor_slices((x_train[x_indices], y_train[y_indices])) 
			# add to list of client datasets
			self.dataset_list.append(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(self.SHUFFLE_BUFFER))  

		# train
		self.build_model()
		self.train()