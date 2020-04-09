import partitioner

import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json
import random
import math
import matplotlib.pyplot as plt

# schema 4: iid data
# TODO: describe schema
# configure:
# distribution of number of data points per client
class Partitioner4(partitioner.Partitioner):

	def __init__(self):
		super().__init__()

	def go(self, num):
		# call parent functions
		self.prep()
		self.test_num(num)
		(x_train, y_train) = self.load_data()
		
		multi = np.zeros(x_train.shape[0])
		num_per_client = np.zeros(self.CLIENTS)

		# partition data and store in class
		for client_num in range(self.CLIENTS):
			# randomize data (generate array of random permutation of indices)
			indices = np.random.permutation(x_train.shape[0])
			# number of data points for this client
			data_per_client = int(np.random.normal(self.NUMDATAPTS_MEAN, self.NUMDATAPTS_STDEV))
			# indices of slice
			start = 0
			end = data_per_client 
			# slice indices for single client
			x_indices = indices[start:end]
			y_indices = indices[start:end]
			# count data point multiplicities and store number of data points
			for i in range(data_per_client):
				multi[int(indices[i])] = multi[int(indices[i])] + 1
			num_per_client[client_num] = data_per_client
			# slice data for single client
			dataset = tf.data.Dataset.from_tensor_slices((x_train[x_indices], y_train[y_indices])) 
			# add to list of client datasets
			self.dataset_list.append(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(self.SHUFFLE_BUFFER))  

		# train
		self.build_model()

		# print("Data point multiplicity count average: ", np.average(multi))
		# print("Data point multiplicity count std dev: ", np.std(multi))
		# plt.hist(multi)
		# plt.title("Data point multiplicities")
		# plt.show()
		# print("Number of data points per client:")
		# print(num_per_client.astype(int))
		# print()

		print()
		print("Schema 4: IID")
		print("--------------------------------------------------")
		print("data points per client (mean, std dev): (", self.NUMDATAPTS_MEAN, ", ", self.NUMDATAPTS_STDEV, ")")
		print()
		print("cohort size: ",self.COHORT_SIZE)
		print("number of local epochs: ",self.NUM_EPOCHS)
		print("local batch size: ", self.BATCH_SIZE)
		print("learning rate: ", self.LR)
		print("target accuracy: ",self.TARGET,"%")
		print("--------------------------------------------------")
		print()

		self.train()