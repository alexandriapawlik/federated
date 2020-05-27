import numpy as np
import math
import csv
import matplotlib.pyplot as plt

batch = 2
batch_name = '2'

cohort_size = [5,10,15,20,30,5,10,15,20,30,5,10,15,20,30]
rounds = []

# seeds
for i in range(3):
	# cohort sizes
	for j in range(5):
		test = (i * 5) + j + 1
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.s1summary.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				rounds.append(int(row[0]))

# make rounds into rounds more than cohort size 5
maxR = rounds[0]
rounds[0] = 0
for x in range(1,5):
	rounds[x] -= maxR
maxR = rounds[5]
rounds[5] = 0
for x in range(6,10):
	rounds[x] -= maxR
maxR = rounds[10]
rounds[10] = 0
for x in range(11,15):
	rounds[x] -= maxR

# add rounds vs cohort size to plot
plt.scatter(cohort_size, rounds)

# finish plot
plt.xlabel('Cohort Size for Each Global Round')
plt.ylabel('Number of Communication Rounds to Reach 90% Accuracy More than that of Smallest Cohort Size')
plt.title("Difference in Number of Rounds with Increased Cohort Size")
plt.suptitle("MNIST, Partially IID Data")
plt.savefig("results/2/rounds_vs_cohort_dif.png")


plt.clf()

cohort_size = [5,10,15,20,30]
seed = [1,5,10]

# seeds
for i in range(3):
	rounds = []
	# cohort sizes
	for j in range(5):
		test = (i * 5) + j + 1
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.s1summary.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				rounds.append(int(row[0]))

	# add rounds vs cohort size to plot
	plt.plot(cohort_size, rounds, label="seed " + str(seed[i]))

# finish plot
plt.xlabel('Cohort Size for Each Global Round')
plt.yticks(np.arange(10, 15, step=1))
plt.ylabel('Number of Communication Rounds to Reach 90% Accuracy')
plt.title("Varied Shuffle Seeds")
plt.suptitle("MNIST, Partially IID Data")
plt.legend()
plt.savefig("results/2/rounds_vs_cohort.png")