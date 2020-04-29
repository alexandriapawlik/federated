import numpy as np
import math
import csv
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

batch = 5699753
batch_name = '5699753-cohort-size-IIDness-halfLR-s1'

### PLOTTING NUM ROUNDS BY COHORT SIZE, COLORED FOR % IID

x = [5,9,13,17,21]
iidness = []
cohort_size = [5,10,15,20]
lr = []
rounds = []

# all data
filename = 'results/' + batch_name + '/' + str(batch) + '.all.target90.csv'
with open(filename,'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	header = next(data)
	for row in data:
		iidness.append(int(row[1]))
		lr.append(float(row[3]))
		rounds.append(int(row[4]))

plt.subplot(121)
i = 1
plt.plot(cohort_size, rounds[i-1:i+3], label=str(iidness[i]) + "% IID data, LR " + str(lr[i]))
# finish plot
plt.xticks(np.arange(5, 21, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Number of Rounds to Reach 90% Accuracy')
plt.title("MNIST, Partially IID Data")
plt.legend()
plt.savefig('results/' + batch_name + '/' + str(batch) + '.rounds_vs_cohortsize.png')

plt.subplot(122)
for i in x: # iterate through %IID
	# add rounds vs cohort size to plot
	if i == 17:
		plt.plot([5,15,20], [rounds[16], rounds[18], rounds[19]], label=str(iidness[i]) + "% IID data, LR " + str(lr[i]))
	else:
		plt.plot(cohort_size, rounds[i-1:i+3], label=str(iidness[i]) + "% IID data, LR " + str(lr[i]))

# finish plot
plt.xticks(np.arange(5, 21, step=5))  # Set label locations.
plt.yticks(np.arange(0,5, step=1))
plt.xlabel('Cohort Size per Global Round')
plt.title("MNIST, Partially IID Data")
plt.legend()
plt.savefig('results/' + batch_name + '/' + str(batch) + '.rounds_vs_cohortsize.png')