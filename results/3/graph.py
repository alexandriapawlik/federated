import numpy as np
import math
import csv
import matplotlib.pyplot as plt

batch = 3
batch_name = '3'

cohort_size = [5,10,15,20,30]
seed = [1,5,10]

# seeds
for i in range(3):
	accuracy = []
	# cohort sizes
	for j in range(5):
		test = (i * 5) + j + 1
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
plt.xlabel('Cohort Size for Each Global Round')
plt.ylabel('Maximum Sparse Categorical Accuracy Reached')
plt.title("MNIST, Partially IID Data, Varied Shuffle Seeds")
plt.legend()
plt.savefig("results/3/accuracy_vs_cohort.png")