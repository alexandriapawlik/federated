import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import statistics as stat

plt.figure(figsize=(12, 6))

batch = 8194978
batch_name = '8194978-cohort-size-80IID-primeseed-100trials-R-s1'

IID = "80"

cohort_size = [5,10,15,20,30]
seed = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547]


# seeds
for i in range(len(seed)):

	accuracy = []

	# cohort sizes
	for j in range(len(cohort_size)):
		test = (i * int(len(cohort_size))) + j + 1
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
plt.suptitle("MNIST, " + str(len(seed)) + " Prime Shuffle Seeds for " + IID + "% IID Data")
plt.title("Maximum Accuracy Reached for Each Seed, with Fairness of Trials")
plt.savefig("results/" + batch_name + "/accuracy_vs_cohort.png")


# averages
plt.clf()
accuracy1 = []
accuracy2 = []
accuracy3 = []
accuracy4 = []
accuracy5 = []

# seeds
for i in range(len(seed)):
	# cohort sizes
	for j in range(len(cohort_size)):
		test = (i * int(len(cohort_size))) + j + 1
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.s1summary.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				if j == 0:
					accuracy1.append(float(row[0]))
				elif j == 1:
					accuracy2.append(float(row[0]))
				elif j == 2:
					accuracy3.append(float(row[0]))
				elif j == 3:
					accuracy4.append(float(row[0]))
				else:
					accuracy5.append(float(row[0]))

# add rounds vs cohort size to plot
accuracy = [sum(accuracy1)/len(accuracy1), sum(accuracy2)/len(accuracy2), sum(accuracy3)/len(accuracy3), sum(accuracy4)/len(accuracy4), sum(accuracy5)/len(accuracy5)]
std = [stat.pstdev(accuracy1), stat.pstdev(accuracy2), stat.pstdev(accuracy3), stat.pstdev(accuracy4), stat.pstdev(accuracy5)]
# plt.plot(cohort_size, accuracy)
plt.errorbar(cohort_size, accuracy, yerr=std, capsize=4)


# finish plot
plt.grid(b=True, which='both', axis='y')
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xlabel('Cohort Size for Each Global Round')
plt.ylabel('Maximum Sparse Categorical Accuracy Reached')
plt.suptitle("MNIST, " + str(len(seed)) + " Prime Shuffle Seeds for " + IID + "% IID Data")
plt.title("Maximum Accuracy Reached Averaged Over Cohort Size, with Fairness of Trials")
plt.savefig("results/" + batch_name + "/avg_accuracy_vs_cohort.png")


# averages as deviation from cohort size 5
plt.clf()
accuracy1 = []
accuracy2 = []
accuracy3 = []
accuracy4 = []
accuracy5 = []

# seeds
for i in range(len(seed)):
	# cohort sizes
	for j in range(len(cohort_size)):
		test = (i * int(len(cohort_size))) + j + 1
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test) + '.s1summary.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				if j == 0:
					accuracy1.append(float(row[0]))
				elif j == 1:
					accuracy2.append(float(row[0]) / accuracy1[-1])
				elif j == 2:
					accuracy3.append(float(row[0]) / accuracy1[-1])
				elif j == 3:
					accuracy4.append(float(row[0]) / accuracy1[-1])
				else:
					accuracy5.append(float(row[0]) / accuracy1[-1])

# add rounds vs cohort size to plot
accuracy = [1, sum(accuracy2)/len(accuracy2), sum(accuracy3)/len(accuracy3), sum(accuracy4)/len(accuracy4), sum(accuracy5)/len(accuracy5)]
std = [0, stat.pstdev(accuracy2), stat.pstdev(accuracy3), stat.pstdev(accuracy4), stat.pstdev(accuracy5)]
plt.errorbar(cohort_size, accuracy, yerr=std, capsize=4)

# finish plot
plt.grid(b=True, which='both', axis='y')
# plt.yticks(np.arange(0, 0.2, step=0.05))
plt.xlabel('Cohort Size for Each Global Round')
plt.ylabel('Maximum Sparse Categorical Accuracy Reached, Normalized')
plt.suptitle("MNIST, " + str(len(seed)) + " Prime Shuffle Seeds for " + IID + "% IID Data")
plt.title("Maximum Accuracy Reached Normalized and Averaged Over Cohort Size, with Fairness of Trials")
plt.savefig("results/" + batch_name + "/avg_accuracy_diff_vs_cohort.png")