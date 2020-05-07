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
# fraction of data from each label is set aside and mixed as IID data
# each client randomly selects IID data from pool
# each client randomly selects labels to pull non-IID data from
# and pulls an even fraction of non-IID data from that label
# configure:
# fraction of data that's distributed to clients IID
# number of classes of non_IID data per client
# distribution of number of data points per client
class Partitioner1(partitioner.Partitioner):

	def __init__(self):
		super().__init__()

	def go(self, num, batch):
		# call parent functions
		self.prep()
		self.test_num(num)
		self.make_config_csv(num, batch)
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
		if self.PERCENT_DATA_IID > 0:
			for i in range(self.LABELS):
				num_iid_pts = int(self.PERCENT_DATA_IID / 100 * len(sorted_data_x[i]))
				this_label_x = np.array(sorted_data_x[i], np.float64)
				this_label_y = np.array(sorted_data_y[i], np.float64)

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
			
		# convert sorted non-IID data to a numpy array
		sorted_x = np.asarray(sorted_data_x)
		sorted_y = np.asarray(sorted_data_y)

		# flatten iid data list
		if self.PERCENT_DATA_IID > 0:
			iid_data_x = np.concatenate(np.array(iid_data_x_temp))
			iid_data_y = np.concatenate(np.array(iid_data_y_temp))

		# duplicate counting setup
			multi_iid = np.zeros(iid_data_x.shape[0], int) # count duplicates in IID
		multi_labels = [] # count duplicates in non IID
		if self.PERCENT_DATA_IID > 0:
			for i in range(self.LABELS):
				multi_labels.append(np.zeros(sorted_x[i].shape, int))
		multi_labels = np.asarray(multi_labels)
		num_data_per_client = []

		# assign each client non-IID and IID data
		for client_num in range(self.CLIENTS):
			# number of data points for this client
			num_data = int(np.random.normal(self.NUMDATAPTS_MEAN, self.NUMDATAPTS_STDEV))
			num_iid = math.ceil(self.PERCENT_DATA_IID / 100 * num_data)
			num_non_iid = num_data - num_iid

			# add IID data to current client
			client_sample_x = []
			client_sample_y = []
			if self.PERCENT_DATA_IID > 0:
				iid_indices = np.random.permutation(iid_data_x.shape[0])
				iid_indices_slice = iid_indices[:num_iid]
				client_sample_x = iid_data_x[iid_indices_slice]
				client_sample_y = iid_data_y[iid_indices_slice]

			# count multiplicities
			if self.PERCENT_DATA_IID > 0:
				for i in range(num_iid):
					multi_iid[int(iid_indices[i])] = multi_iid[int(iid_indices[i])] + 1
					
			# select labels for non_IID
			label_indices = np.random.permutation(10)
			chosen_labels = label_indices[:self.SHARDS]
			
			# add data from each label
			data_per_label = num_non_iid // self.SHARDS
			extra = (num_non_iid % self.SHARDS) + data_per_label
			for i in range(self.SHARDS):
				# add non-IID data from this label to current client
				label = chosen_labels[i]
				label_data_x = np.array(sorted_x[label], np.float64)
				label_data_y = np.array(sorted_y[label], np.float64)
				indices = np.random.permutation(label_data_x.shape[0])
				indices_slice = []
				if i == len(chosen_labels) - 1:  # pull remainder data from last label
					indices_slice = indices[:extra]
				else:
					indices_slice = indices[:data_per_label]

				if self.PERCENT_DATA_IID > 0:
					client_sample_x = np.append(client_sample_x, label_data_x[indices_slice], axis=0)
					client_sample_y = np.append(client_sample_y, label_data_y[indices_slice], axis=0)
				else:
					client_sample_x = label_data_x[indices_slice]
					client_sample_y = label_data_y[indices_slice]

				# count multiplicities
				if self.PERCENT_DATA_IID > 0:
					for i in range(len(indices_slice)):
						multi_labels[label][int(indices_slice[i])] = multi_labels[label][int(indices_slice[i])] + 1
					
				# check data
				if np.average(client_sample_y) > 9 or np.average(client_sample_y) < 0:
					print("Error: At least one label out of range")
					print(np.average(client_sample_y), label)
					print()

				# TODO: print data multiplicities

			# track number of data points per client
			num_data_per_client.append(len(client_sample_x))

			# assign slices to single client
			dataset = tf.data.Dataset.from_tensor_slices((client_sample_x, client_sample_y))
			# add to list of client datasets
			self.dataset_list.append(dataset.repeat(self.NUM_EPOCHS).batch(self.BATCH_SIZE).shuffle(60000, seed = self.SHUFFLE_SEED, reshuffle_each_iteration=True))

		# train
		self.build_model()

		print()
		print("Schema 1: Each client partially IID")
		print("--------------------------------------------------")
		print("percent data distributed IID: ", self.PERCENT_DATA_IID)
		print("number of classes for non-IID data: ", self.SHARDS)
		print("data points per client (mean, std dev): (", self.NUMDATAPTS_MEAN, ", ", self.NUMDATAPTS_STDEV, ")")
		# print()
		# print("number of clients: ", self.CLIENTS)
		# print("cohort size: ",self.COHORT_SIZE)
		# print("number of local epochs: ",self.NUM_EPOCHS)
		# print("local batch size: ", self.BATCH_SIZE)
		# print("learning rate: ", self.LR)
		# print("target accuracy: ",self.TARGET,"%")
		print("--------------------------------------------------")
		# print("number of data points per client:")
		# print(num_data_per_client)
		# print("--------------------------------------------------")
		print()

		self.train(num, batch, 1)