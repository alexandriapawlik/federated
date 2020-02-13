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

	def go(self):
		# call parent functions
		self.prep()
		(x_train, y_train) = self.load_data()
		data_per_client = x_train.shape[0] // self.NUM_CLIENTS
		# TODO: randomize data

		# partition data and store
		for client_num in range(1, self.NUM_CLIENTS + 1): # TODO: should this be +1?
			start = (client_num - 1) * self.NUM_CLIENTS
			end = (client_num * self.NUM_CLIENTS)
			dataset = tf.data.Dataset.from_tensor_slices((x_train[start:end], y_train[start:end]))
			self.dataset_list.append(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(self.SHUFFLE_BUFFER)

		# train
		self.build_model()
		self.train()