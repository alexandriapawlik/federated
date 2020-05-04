import numpy as np
import math
import csv
import matplotlib.pyplot as plt

batch = 5814359
batch_name = '5814359-cohort-size-IIDness-LR-s1'

x = [5,9,13,17,21]
y = [0,1,2,3]

plt.subplot(121)
i = 1
filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.config.csv'
with open(filename,'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	header = next(data)
	for row in data:
		iidness = int(row[-3])
		lr = float(row[4])
	
cohort_size = [5,10,15,20]
rounds = []

for j in y: # cohort sizes
	test = i + j
	# get number of rounds
	filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.s1summary.csv'
	with open(filename,'r') as csvfile:
		data = csv.reader(csvfile, delimiter=',')
		header = next(data)
		for row in data:
			rounds.append(int(row[0]))

# add rounds vs cohort size to plot
plt.plot(cohort_size, rounds, label=str(iidness) + "% IID data, LR " + str(lr))

# finish plot
plt.xticks(np.arange(5, 21, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Number of Rounds to Reach 90% Accuracy')
plt.title("MNIST, Partially IID Data")
plt.legend()


plt.subplot(122)

for i in x: # IIDness groups
	# get IIDness and learning rate
	filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.config.csv'
	with open(filename,'r') as csvfile:
		data = csv.reader(csvfile, delimiter=',')
		header = next(data)
		for row in data:
			iidness = int(row[-3])
			lr = float(row[4])
	
	cohort_size = [5,10,15,20]
	rounds = []

	for j in y: # cohort sizes
		test = i + j
		# get number of rounds
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.s1summary.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				rounds.append(int(row[0]))

	# add rounds vs cohort size to plot
	plt.plot(cohort_size, rounds, label=str(iidness) + "% IID data, LR " + str(lr), alpha=0.6)

# finish plot
plt.xticks(np.arange(5, 21, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.title("MNIST, Partially IID Data")
plt.legend()
plt.savefig('results/' + batch_name + '/' + str(batch) + '.rounds_vs_cohortsize.png')