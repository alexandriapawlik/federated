import numpy as np
import math
import csv
import matplotlib.pyplot as plt

batch = 5666478
batch_name = '5666478-cohort-size-IIDness-s1'

### PLOTTING NUM ROUNDS BY COHORT SIZE, COLORED FOR % IID

x = [1,2,3,4]
y = [2,3,4,5]
# plt.figure(figsize=(16, 8))

for i in x: # iterate through cohort sizes
	# get cohort size
	filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.config.csv'
	with open(filename,'r') as csvfile:
		data = csv.reader(csvfile, delimiter=',')
		header = next(data)
		for row in data:
			cohort_size = int(row[0])
	
	iidness = []
	rounds = []

	for j in y: # iterate through %IID
		test = i + (4 * j)
		# get % IID
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.config.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				iidness.append(int(row[-3]))
		# get number of rounds
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.s1summary.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				rounds.append(int(row[0]))

	# add rounds vs iidness to plot
	plt.plot(iidness, rounds, label="Cohort size " + str(cohort_size))

# finish plot
plt.xlabel('Percent IID Data per Client')
plt.ylabel('Number of Rounds to Reach 99% Accuracy')
plt.title("MNIST, Partially IID Data")
plt.legend()
plt.savefig('results/' + batch_name + '/' + str(batch) + '.rounds_vs_IIDness.png')