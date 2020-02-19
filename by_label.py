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
		total_shards = self.CLIENTS * self.SHARDS
		shard_size = int(x_train.shape[0] // total_shards)  # trumps number of shards per client
		
		# list for each of 10 labels
		sorted_data_x = [[] for i in range(self.LABELS)]
		sorted_data_y = [[] for i in range(self.LABELS)]

		# partition data and store
		for data_num in range(y_train.shape[0]):
			# sort data into each of 10 categories based on label 0-9 (6,000/label)
			sorted_data_x[int(y_train[data_num])].append(x_train[data_num])
			sorted_data_y[int(y_train[data_num])].append(y_train[data_num])

		sorted_x = np.array(sorted_data_x)
		sorted_y = np.array(sorted_data_y)

		# split into shards
		# (divide into 200 shards of size 300)
		shards_x = [[] for i in range(total_shards)]
		shards_y = [[] for i in range(total_shards)]

		# for each label, make 20 shards
		for label_num in range(self.LABELS): 

			# make sure we have enough data for desired shard size
			if (len(sorted_x[label_num]) // shard_size) == 0:
				print("Error: Shard size larger than number of datapoints (",len(sorted_x[label_num]),") per label for label ",label_num,".") 
				print("Increase number of clients, number of shards per client, or datapoints for this label.")

			# randomize data for this label (generate array of random permutation of indices)
			indices = np.random.permutation(len(sorted_x[label_num]))

			# for each shard chunk in one label
			for shard_num in range(int(len(sorted_x[label_num]) // shard_size)): 
				# calculate indices of slice
				start = shard_num * shard_size
				end = (shard_num + 1) * shard_size
				# slice indices for single shard
				x_indices = indices[start:end]
				y_indices = indices[start:end]
				# slice data for single shard and add to shard list
				shards_x.append(sorted_x[label_num][x_indices.astype(int)])
				shards_y[shard_num].extend((sorted_y[label_num])[y_indices.astype(int)])
				# check
				if len(shards_x[shard_num]) != 300:
					print("Shard size is",len(shards_x[shard_num]))
			
			# check
			if len(shards_x) != 200:
				print("Number of shards is",len(shards_x))

		# randomize order of shards
		shard_indices = np.random.permutation(len(shards_x))

		# assign each client shards
		current_shard = 0
		for client_num in range(self.CLIENTS):
			# add shards to current client using randomized index list
			# wraparound if we run out
			client_sample_x = []
			client_sample_y = []
			for shard_count in range(self.SHARDS):
				client_sample_x.extend(shards_x[shard_indices[current_shard]])
				client_sample_y.extend(shards_y[shard_indices[current_shard]])
				current_shard = (current_shard + 1) % len(shard_indices)

			# assign slices to single client
			dataset = tf.data.Dataset.from_tensor_slices((client_sample_x, client_sample_y))
			# add to list of client datasets
			self.dataset_list.append(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(self.SHUFFLE_BUFFER))
			# TODO: does shuffle buffer guarantee that the data gets shuffled at each client?

		# train
		self.build_model()
		self.train()