import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import statistics as stat

plt.figure(figsize=(12, 6))

batch = 7852999
batch_name = '7852999-cohort-size-80IID-10trials-R-s1'

IID = "80"

cohort_size = [5,10,15,20,30]
seed = list(range(1001,2001,20))

# seeds
for i in range(len(seed)):

	accuracy = []

	# cohort sizes, 40% IID
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
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xlabel('Cohort Size for Each Global Round')
plt.ylabel('Maximum Sparse Categorical Accuracy Reached')
plt.suptitle("MNIST, " + str(len(seed)) + " Shuffle Seeds for " + IID + "% IID Data")
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
	# cohort sizes, 40% IID
	for j in range(5):
		test = (i * 5) + j + 1
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
plt.suptitle("MNIST, " + str(len(seed)) + " Shuffle Seeds for " + IID + "% IID Data")
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
	# cohort sizes, 40% IID
	for j in range(5):
		test = (i * 5) + j + 1
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
plt.suptitle("MNIST, " + str(len(seed)) + " Shuffle Seeds for " + IID + "% IID Data")
plt.title("Maximum Accuracy Reached Normalized and Averaged Over Cohort Size, with Fairness of Trials")
plt.savefig("results/" + batch_name + "/avg_accuracy_diff_vs_cohort.png")