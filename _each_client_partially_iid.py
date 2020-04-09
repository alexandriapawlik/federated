import partitioner

import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json
import random
import math

# schema 1: each client partially iid
# TODO: describe schema
# uses sharding for non-IID
# configure:
# fraction of data that's distributed to clients IID
# number of classes of non_IID data per client
# distribution of number of data points per client
# NOTE: significant data overlap will only occur within IID data if at all
# slight overlap occurs in non-IID data in redistribution of a few shards
class Partitioner1(partitioner.Partitioner):

	def __init__(self):
		super().__init__()

	def go(self, num):
		# call parent functions
		self.prep()
		self.test_num(num)
		(x_train, y_train) = self.load_data()
		
		# list for each of 10 labels ( arr[label][data point index] )
		sorted_data_x = [[] for i in range(self.LABELS)]
		sorted_data_y = [[] for i in range(self.LABELS)]

		# sort data by label and store
		for data_num in range(y_train.shape[0]):
			# sort data into each of 10 categories based on label 0-9 (6,000/label)
			sorted_data_x[int(y_train[data_num])].append(x_train[data_num])
			sorted_data_y[int(y_train[data_num])].append(y_train[data_num])

		# pull IID portion of data from each label
		iid_data_x_temp = []
		iid_data_y_temp = []
		for i in range(self.LABELS):
			num_iid_pts = int(self.PERCENT_DATA_IID / 100 * len(sorted_data_x[i]))
			this_label_x = np.array(sorted_data_x[i])
			this_label_y = np.array(sorted_data_y[i])

			# shuffle data within label
			indices = np.random.permutation(len(this_label_x))

			# take IID portion
			iid_indices = indices[:num_iid_pts] # indices slice
			iid_data_x_temp.append(this_label_x[iid_indices])
			iid_data_y_temp.append(this_label_y[iid_indices])

			# return non-IID portion
			non_iid_indices = indices[num_iid_pts:]
			sorted_data_x[i] = this_label_x[non_iid_indices]
			sorted_data_y[i] = this_label_y[non_iid_indices]

		# flatten iid data list
		iid_data_x = np.concatenate(np.array(iid_data_x_temp))
		iid_data_y = np.concatenate(np.array(iid_data_y_temp))

		# split remaining data into shards
		total_shards = self.CLIENTS * self.SHARDS
		shard_size = int(x_train.shape[0] / total_shards * (100 - self.PERCENT_DATA_IID) / 100)
		shards_x = np.empty([total_shards, shard_size, 28, 28, 1])
		shards_y = np.empty([total_shards, shard_size])
		shards_idx = 0

		# for each label, make shards
		for label_num in range(self.LABELS): 
			# make ndarrays from label lists
			this_label_x = np.array(sorted_data_x[label_num])
			this_label_y = np.array(sorted_data_y[label_num])

			# randomize data for this label before making shards
			indices = np.random.permutation(len(this_label_x))

			# for each shard chunk in this label
			for shard_num in range(len(this_label_x) // shard_size): 
				# calculate indices of slice
				start = shard_num * shard_size
				end = (shard_num + 1) * shard_size

				# slice indices for single shard
				chosen_indices = indices[start:end]

				# slice data for single shard and add to shard lists
				shards_x[shards_idx] = this_label_x[chosen_indices]
				shards_y[shards_idx] = this_label_y[chosen_indices]
				shards_idx = shards_idx + 1

		# shorten shard array to actual length
		shards_x = np.delete(shards_x, [range(shards_idx, len(shards_x))], axis=0)
		shards_y = np.delete(shards_y, [range(shards_idx, len(shards_y))], axis=0)

		# randomize order of data
		shard_indices = np.random.permutation(len(shards_x))
		iid_indices = np.random.permutation(len(iid_data_x))

		# assign each client shards and IID data
		shards_idx = 0
		iid_idx = 0
		num_data_per_client = []
		for client_num in range(self.CLIENTS):
			# number of data points for this client
			num_data = int(np.random.normal(self.NUMDATAPTS_MEAN, self.NUMDATAPTS_STDEV))
			client_sample_x_temp = [] # unflattened
			client_sample_y_temp = []

			# add shards to current client
			for i in range(self.SHARDS):
				client_sample_x_temp.append(shards_x[shard_indices[shards_idx]])
				client_sample_y_temp.append(shards_y[shard_indices[shards_idx]])
				shards_idx = (shards_idx + 1) % len(shards_x)

			# flatten client data arrays
			client_sample_x = np.concatenate(np.array(client_sample_x_temp))
			client_sample_y = np.concatenate(np.array(client_sample_y_temp))

			# add IID data to current client (if number of data points isn't enough after shards)
			if (self.SHARDS * shard_size) < num_data:
				for i in range((self.SHARDS * shard_size), num_data):
					data_x = np.reshape(iid_data_x[iid_indices[iid_idx]], (1,28,28,1))
					client_sample_x = np.append(client_sample_x, data_x, axis=0)
					client_sample_y = np.append(client_sample_y, iid_data_y[iid_indices[iid_idx]])
					iid_idx = (iid_idx + 1) % len(iid_data_x)

			# track number of data points per client
			num_data_per_client.append(len(client_sample_x))

			# TODO: count data multiplicities
			# TODO: change parameter to be percent non-IID (technically)

			# assign slices to single client
			dataset = tf.data.Dataset.from_tensor_slices((client_sample_x, client_sample_y))
			# add to list of client datasets
			self.dataset_list.append(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(self.SHUFFLE_BUFFER))

		# train
		self.build_model()

		print()
		print("Schema 1: Each client partially IID")
		print("--------------------------------------------------")
		print("percent data distributed IID: ", self.PERCENT_DATA_IID)
		print("number of classes for non-IID data: ", self.SHARDS)
		print("data points per client (mean, std dev): (", self.NUMDATAPTS_MEAN, ", ", self.NUMDATAPTS_STDEV, ")")
		print()
		print("cohort size: ",self.COHORT_SIZE)
		print("number of local epochs: ",self.NUM_EPOCHS)
		print("local batch size: ", self.BATCH_SIZE)
		print("learning rate: ", self.LR)
		print("target accuracy: ",self.TARGET,"%")
		print("--------------------------------------------------")
		# print("number of data points per client:")
		# print(num_data_per_client)
		# print("--------------------------------------------------")
		print()

		self.train()