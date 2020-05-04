import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import statistics as stat

plt.figure(figsize=(12, 6))

batch = 5842441
batch_name = '5842441-cohort-size-IIDness-10trials-s1'

x1 = [1,7]
x2 = [13,19,25,31]  # round nums
y = [0,1,2,3,4,5]
cohort_size = [5,10,15,20,25,30]

plt.subplot(121)
# iterate through IIDness
for i in x1: 
	rounds = []
	rounds_std = []
	rounds_min = []
	rounds_max = []

	# get IIDness
	filename = 'results/' + batch_name + '/r0/' + str(batch) + '.' + str(i) + '.config.csv'
	with open(filename,'r') as csvfile:
		data = csv.reader(csvfile, delimiter=',')
		header = next(data)
		for row in data:
			iidness = int(row[-3])

	# iterate though cohort sizes
	for j in y:
		test = i + j

		# iterate though 10 trials
		rounds_all_trials = []
		for z in range(10): # folders 0-9
			# get number of rounds
			filename = 'results/' + batch_name + '/r' + str(z) + '/' + str(batch) + '.' + str(test + (36 * z)) + '.s1summary.csv'
			with open(filename,'r') as csvfile:
				data = csv.reader(csvfile, delimiter=',')
				header = next(data)
				for row in data:
					rounds_all_trials.append(int(row[0]))

		# get average and stdev num rounds for this cohort size in this IIDness
		rounds.append(sum(rounds_all_trials) / len(rounds_all_trials))
		rounds_std.append(stat.pstdev(rounds_all_trials))
		rounds_min.append(rounds[-1] - min(rounds_all_trials))
		rounds_max.append(max(rounds_all_trials) - rounds[-1])
		print(cohort_size[j], rounds_all_trials)

	# add average rounds vs cohort size to plot
	# plt.errorbar(cohort_size, rounds, yerr=rounds_std, label=str(iidness) + '% IID data', capsize=4)
	plt.errorbar(cohort_size, rounds, yerr=[rounds_min, rounds_max], label=str(iidness) + '% IID data', capsize=4)

# finish plot
plt.xticks(np.arange(5, 31, step=5))  # Set label locations.
# plt.yticks(np.arange(0,201, step=20))
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Average Number of Rounds to Reach 99% Accuracy')
plt.title("MNIST, Partially IID Data, LR 0.1")
plt.legend()

plt.subplot(122)
# iterate through IIDness
for i in x2: 
	rounds = []
	rounds_std = []
	rounds_min = []
	rounds_max = []

	# get IIDness
	filename = 'results/' + batch_name + '/r0/' + str(batch) + '.' + str(i) + '.config.csv'
	with open(filename,'r') as csvfile:
		data = csv.reader(csvfile, delimiter=',')
		header = next(data)
		for row in data:
			iidness = int(row[-3])

	# iterate though cohort sizes
	for j in y:
		test = i + j

		# iterate though 10 trials
		rounds_all_trials = []
		for z in range(10): # folders 0-9
			# get number of rounds
			filename = 'results/' + batch_name + '/r' + str(z) + '/' + str(batch) + '.' + str(test + (36 * z)) + '.s1summary.csv'
			with open(filename,'r') as csvfile:
				data = csv.reader(csvfile, delimiter=',')
				header = next(data)
				for row in data:
					rounds_all_trials.append(int(row[0]))

		# get average and stdev num rounds for this cohort size in this IIDness
		rounds.append(sum(rounds_all_trials) / len(rounds_all_trials))
		rounds_std.append(stat.pstdev(rounds_all_trials))
		rounds_min.append(rounds[-1] - min(rounds_all_trials))
		rounds_max.append(max(rounds_all_trials) - rounds[-1])
		print(cohort_size[j], rounds_all_trials)

	# add average rounds vs cohort size to plot
	# plt.errorbar(cohort_size, rounds, yerr=rounds_std, label=str(iidness) + '% IID data', capsize=4)
	plt.errorbar(cohort_size, rounds, yerr=[rounds_min, rounds_max], label=str(iidness) + '% IID data', capsize=4)

# finish plot
plt.xticks(np.arange(5, 31, step=5))  # Set label locations.
# plt.yticks(np.arange(0,201, step=20))
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Average Number of Rounds to Reach 99% Accuracy')
plt.title("MNIST, Partially IID Data, LR 0.2")
plt.legend()
plt.savefig('results/' + batch_name + '/' + str(batch) + '.rounds_vs_cohortsize_99bars.png')