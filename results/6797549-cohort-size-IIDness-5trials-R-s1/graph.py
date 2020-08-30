import numpy as np
import math
import csv
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

batch = 6797549
batch_name = '6797549-cohort-size-IIDness-5trials-R-s1'

cohort_size = [5,10,15,20,30]
seed = [1,5,10,14,20]

plt.subplot(121)
# seeds
for i in range(5):

	accuracy = []

	# cohort sizes, 40% IID
	for j in range(5):
		test = (i * 10) + j + 1
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.s1summary.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				accuracy.append(float(row[0]))

	# add rounds vs cohort size to plot
	plt.plot(cohort_size, accuracy, label="seed " + str(seed[i]))

# finish plot
plt.grid(b=True, which='both', axis='y')
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xlabel('Cohort Size for Each Global Round')
plt.ylabel('Maximum Sparse Categorical Accuracy Reached')
plt.title("5 Shuffle Seeds for 40% IID Data")
# plt.legend()

plt.subplot(122)
# seeds
for i in range(5):
	accuracy = []

	# cohort sizes, 80% IID
	for j in range(5):
		test = (i * 10) + j + 6
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.s1summary.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				accuracy.append(float(row[0]))

	# add rounds vs cohort size to plot
	plt.plot(cohort_size, accuracy, label="seed " + str(seed[i]))

# finish plot
plt.grid(b=True, which='both', axis='y')
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xlabel('Cohort Size for Each Global Round')
plt.ylabel('Maximum Sparse Categorical Accuracy Reached')
plt.title("5 Shuffle Seeds for 80% IID Data")
plt.suptitle("FL on CNN with MNIST and Partially IID Client Data") # LR 0.1
# plt.legend()
plt.savefig("results/" + str(batch) + "_accuracy_vs_cohort.png")