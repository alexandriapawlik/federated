import partitioner

import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json
import random
import math

# schema 2: some clients iid
# TODO: describe schema
# configure:
# fraction of clients with all IID data
# number of classes of data per non-IID client
# distribution of number of data points per client
class Partitioner2(partitioner.Partitioner):

	def __init__(self):
		super().__init__()

	def go(self, num, batch):
		# call parent functions
		self.prep()
		self.test_num(num)
		self.make_config_csv(num, batch)
		(x_train, y_train) = self.load_data()
		total_shards = self.CLIENTS * self.SHARDS
		shard_size = int(x_train.shape[0] // total_shards)  # trumps number of shards per client
		
		# list for each of 10 labels
		sorted_data_x = [[] for i in range(self.LABELS)]
		sorted_data_y = [[] for i in range(self.LABELS)]

		# sort data by label and store
		for data_num in range(y_train.shape[0]):
			# sort data into each of 10 categories based on label 0-9 (6,000/label)
			sorted_data_x[int(y_train[data_num])].append(x_train[data_num])
			sorted_data_y[int(y_train[data_num])].append(y_train[data_num])

		# split into shards
		# (divide into 200 shards of size 300)
		shards_x = np.empty([total_shards, shard_size, 28, 28, 1])
		shards_y = np.empty([total_shards, shard_size])
		shards_idx = 0

		# for each label, make 20 shards
		for label_num in range(self.LABELS): 
			# make ndarrays from label lists
			sorted_x = np.array(sorted_data_x[label_num])
			sorted_y = np.array(sorted_data_y[label_num])

			# make sure we have enough data for desired shard size
			if (len(sorted_data_x[label_num]) // shard_size) == 0:
				print("Error: Shard size larger than number of datapoints (",len(sorted_data_x[label_num]),") per label for label ",label_num,".") 
				print("Increase number of clients, number of shards per client, or datapoints for this label.")

			# randomize data for this label before making shards (generate array of random permutation of indices)
			indices = np.random.permutation(len(sorted_data_x[label_num]))

			# for each shard chunk in one label
			for shard_num in range(len(indices) // shard_size): 
				# calculate indices of slice
				start = shard_num * shard_size
				end = (shard_num + 1) * shard_size

				# slice indices for single shard
				x_indices = indices[start:end]
				y_indices = indices[start:end]

				# slice data for single shard and add to shard lists
				shards_x[shards_idx] = sorted_x[x_indices]
				shards_y[shards_idx] = sorted_y[y_indices]
				shards_idx = shards_idx + 1

		# randomize order of shards before assigning to clients
		shard_indices = np.random.permutation(len(shards_x))

		# assign each client shards
		current_shard = 0
		for client_num in range(self.CLIENTS):
			# add shards to current client using randomized index list
			# wrap around if we run out
			client_sample_x = np.empty([shard_size * self.SHARDS, 28, 28, 1])
			client_sample_y = np.empty([shard_size * self.SHARDS])

			# for as many shards as config file says we should have per client
			for shard_count in range(self.SHARDS):
				# get shard based on randomized indices
				start = shard_count * shard_size
				end = (shard_count + 1) * shard_size
				client_sample_x[start:end] = shards_x[shard_indices[current_shard]]
				client_sample_y[start:end] = shards_y[shard_indices[current_shard]]
				# increment pointer
				current_shard = (current_shard + 1) % len(shard_indices)

			# assign slices to single client
			dataset = tf.data.Dataset.from_tensor_slices((client_sample_x, client_sample_y))
			# add to list of client datasets
			self.dataset_list.append(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(60000, seed = self.SHUFFLE_SEED, reshuffle_each_iteration=True))

		# train
		self.build_model()

		print()
		print("Schema 2: Some clients IID")
		print("--------------------------------------------------")
		print("percent clients IID: ", self.PERCENT_CLIENTS_IID)
		print("number of classes for non-IID clients: ", self.SHARDS)
		print("data points per client (mean, std dev): (", self.NUMDATAPTS_MEAN, ", ", self.NUMDATAPTS_STDEV, ")")
		# print()
		# print("number of clients: ", self.CLIENTS)
		# print("cohort size: ", self.COHORT_SIZE)
		# print("number of local epochs: ", self.NUM_EPOCHS)
		# print("local batch size: ", self.BATCH_SIZE)
		# print("learning rate: ", self.LR)
		# print("target accuracy: ", self.TARGET,"%")
		print("--------------------------------------------------")
		print()

		self.train(num, batch, 2)