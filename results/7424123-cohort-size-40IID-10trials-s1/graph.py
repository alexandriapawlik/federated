import numpy as np
import math
import csv
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

batch = 7424123
batch_name = '7424123-cohort-size-40IID-10trials-s1'

cohort_size = [5,10,15,20,30]
seed = list(range(1,11))

# seeds
for i in range(len(seed)):
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
plt.yticks(np.arange(5, 30, step=1))
plt.ylabel('Number of Communication Rounds to Reach 90% Accuracy')
plt.title("Varied Shuffle Seeds")
plt.suptitle("MNIST, Partially IID Data")
plt.legend()
plt.savefig("results/" + batch_name + "/rounds_vs_cohort.png")