import numpy as np
import math
import csv
import matplotlib.pyplot as plt

batch = 1
batch_name = '1'

cohort_size = [5,20,30,40]
rounds = []

for i in range(4):
	filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i + 1) + '.s1summary.csv'
	with open(filename,'r') as csvfile:
		data = csv.reader(csvfile, delimiter=',')
		header = next(data)
		for row in data:
			rounds.append(int(row[0]))

# add rounds vs cohort size to plot
plt.plot(cohort_size, rounds, label="80% IID data, LR 0.1")

# finish plot
plt.xlabel('Cohort Size for Each Global Round')
plt.ylabel('Number of Communication Rounds to Reach 99% Accuracy')
plt.title("MNIST, Partially IID Data, Consistent Shuffle Seed")
plt.legend()
plt.savefig("results/1/seeded_rounds_vs_cohort.png")