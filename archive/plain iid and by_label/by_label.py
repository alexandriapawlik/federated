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

	def go(self, num):
		# call parent functions
		self.prep()
		self.test_num(num)
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

			# # count label types for each client
			# label_count = np.zeros((10,), dtype=int)
			# for i in range(len(client_sample_y)):
			# 	label_count[int(client_sample_y[i])] = label_count[int(client_sample_y[i])] + 1
			# print(label_count)

			# assign slices to single client
			dataset = tf.data.Dataset.from_tensor_slices((client_sample_x, client_sample_y))
			# add to list of client datasets
			self.dataset_list.append(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(self.SHUFFLE_BUFFER))

		# train
		self.build_model()

		print()
		print("Non-IID partitioning:")
		print("60,000 samples divided by label, split into 200 shards, and randomly distributed to 100 clients")
		print()

		self.train()