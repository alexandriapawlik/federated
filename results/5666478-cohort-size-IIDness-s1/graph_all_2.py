import numpy as np
import math
import csv
import matplotlib.pyplot as plt

batch = 5666478
batch_name = '5666478-cohort-size-IIDness-s1'

### PLOTTING NUM ROUNDS BY COHORT SIZE, COLORED FOR % IID

x = [9,13,17,21]
y = [0,1,2,3]
# plt.figure(figsize=(16, 8))

for i in x: # iterate through %IID
	# get %IID
	filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.config.csv'
	with open(filename,'r') as csvfile:
		data = csv.reader(csvfile, delimiter=',')
		header = next(data)
		for row in data:
			iidness = int(row[-3])
	
	cohort_size = []
	rounds = []

	for j in y: # iterate through cohort size
		test = i + j
		# get number of rounds
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.s1summary.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				if int(row[0]) < 100:
					rounds.append(int(row[0]))
					# get cohort size
					filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.config.csv'
					with open(filename,'r') as csvfile:
						data = csv.reader(csvfile, delimiter=',')
						header = next(data)
						for row in data:
							cohort_size.append(int(row[0]))

	# add rounds vs cohort size to plot
	plt.plot(cohort_size, rounds, label=str(iidness) + "% IID data")

# finish plot
plt.xticks(np.arange(5, 21, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Number of Rounds to Reach 99% Accuracy')
plt.title("MNIST, Partially IID Data")
plt.legend()
plt.savefig('results/' + batch_name + '/' + str(batch) + '.rounds_vs_cohortsize.png')