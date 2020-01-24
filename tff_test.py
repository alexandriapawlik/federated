# disables CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tutorial: https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification
import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# check that environment is correctly set up
warnings.simplefilter('ignore')
tf.compat.v1.enable_v2_behavior()
np.random.seed(0)
tff.framework.set_default_executor(tff.framework.create_local_executor())

assert tff.federated_computation(lambda: 'Hello, World!')() == b'Hello, World!'

# variables are tff.simulation.ClientData objects
(emnist_train, emnist_test) = tff.simulation.datasets.emnist.load_data()

assert len(emnist_train.client_ids) == 3383
assert emnist_train.element_type_structure == OrderedDict([('label', TensorSpec(shape=(), dtype=tf.int32, name=None)), ('pixels', TensorSpec(shape=(28, 28), dtype=tf.float32, name=None))])

# creates a new tf.data.Dataset containing the client[0] training examples
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])

# test for specific datum
example_element = iter(example_dataset).next()
assert example_element['label'].numpy() == 5

# end environment test

# plot
# from matplotlib import pyplot as plt
# plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
# plt.grid('off')
# _ = plt.show()

# preprocessing can be accomplished using Dataset transformations
NUM_ROUNDS = 11  # total num of aggregations
NUM_CLIENTS = 10  # per round
NUM_EPOCHS = 10  # for client model
BATCH_SIZE = 20  # for client model
SHUFFLE_BUFFER = 500

def preprocess(dataset):

	# flatten
	# renames the features from pixels and label to x and y for use with Keras
  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], [-1])),
        ('y', tf.reshape(element['label'], [1])),
    ])

	# shuffle the individual examples and organize them into batches
	# repeat over the data set to run several epochs
  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)

# create sample batch for Keras model wrapper
# TODO: map_structure syntax
preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

# one of the ways to feed federated data to TFF in a simulation is simply as a Python list
# with each element of the list holding the data of an individual user 
# whether as a list or as a tf.data.Dataset
# we already have an interface that provides the latter

# construct a list of datasets from the given set of users 
# as an input to a round of training or evaluation
def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]

# sample the set of clients once
# and reuse the same set across rounds to speed up convergence 
# (intentionally over-fitting to these few user's data)
# leave it as an exercise for the reader to modify this tutorial to simulate random sampling 
# fairly easy to do (once you do, keep in mind that getting the model to converge may take a while)
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)
assert len(federated_train_data) == 10
# print(federated_train_data[0])

# simple model with Keras
def create_compiled_keras_model():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])
  
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model

# if you have a compiled Keras model like the one defined above
# you can have TFF wrap it for you by invoking tff.learning.from_compiled_keras_model
# and passing the model and a sample data batch as arguments
# TODO: why do we need a sample batch here?
def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

# let TFF construct a Federated Averaging algorithm 
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
# now TFF has constructed a pair of federated computations and packaged them 
# into a tff.utils.IterativeProcess in which these computations are available 
# as a pair of properties initialize and next

# returns the representation of the state of the Federated Averaging process on server
str(iterative_process.initialize.type_signature)

# construct the server state
state = iterative_process.initialize()

# next represents a single round of Federated Averaging
# which consists of pushing the server state (including the model parameters) to the clients,
# on-device training on their local data, collecting and averaging model updates, 
# and producing a new updated model at the server

# think about next() not as being a function that runs on a server, 
# but rather being a declarative functional representation of the entire 
# decentralized computation - some of the inputs are provided by the server (SERVER_STATE), 
# but each participating device contributes its own local dataset

# run a single round of training
state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))

# run a few more rounds
# typically at this point you would pick a subset of your simulation data from 
# a new randomly selected sample of users for each round in order to 
# simulate a realistic deployment in which users continuously come and go (TODO)
for round_num in range(2, NUM_ROUNDS):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))

# visualize the metrics from these federated computations using Tensorboard