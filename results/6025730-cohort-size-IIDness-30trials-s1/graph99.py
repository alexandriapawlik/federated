import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import statistics as stat
from sklearn.linear_model import LinearRegression

plt.figure(figsize=(10, 5))

batch = 6025730
batch_name = '6025730-cohort-size-IIDness-30trials-s1'
trials = 9
tests_per_trial = 75

x = [1,16,31,46,61]  # round nums
cohort_size = [2, 5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40]
y = list(range(len(cohort_size)))

# x is a list of tests with smallest cohort size (beginning points for each line)
# option is 1, 2, 3, or 4
# 1: plain connected plot
# 2: plot with standard deviation bars
# 3: plot with max/min bars
# 4: linear regression
def plotx(x, option):
	# iterate through IIDness
	for i in x: 
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

			# iterate though trials
			rounds_all_trials = []
			for z in range(trials): # folders
				# get number of rounds
				filename = 'results/' + batch_name + '/r' + str(z) + '/' + str(batch) + '.' + str(test + (tests_per_trial * z)) + '.s1summary.csv'
				with open(filename,'r') as csvfile:
					data = csv.reader(csvfile, delimiter=',')
					header = next(data)
					for row in data:
						# if not a timeout
						if int(row[0]) < 250:
							rounds_all_trials.append(int(row[0]))

			# get average and stdev num rounds for this cohort size in this IIDness
			rounds.append(sum(rounds_all_trials) / len(rounds_all_trials))
			rounds_std.append(stat.pstdev(rounds_all_trials))
			rounds_min.append(rounds[-1] - min(rounds_all_trials))
			rounds_max.append(max(rounds_all_trials) - rounds[-1])
			print(cohort_size[j], iidness, rounds_all_trials)
			
		# add average rounds vs cohort size to plot
		if option == 1: # normal plot
			plt.plot(cohort_size, rounds, label=str(iidness) + '% IID data')
		elif option == 2: # stdev
			plt.errorbar(cohort_size, rounds, yerr=rounds_std, label=str(iidness) + '% IID data, stdev', capsize=4)
		elif option == 3:  # max/min
			plt.errorbar(cohort_size, rounds, yerr=[rounds_min, rounds_max], label=str(iidness) + '% IID data, max/min', capsize=4)
		elif option == 4:  # linear regression
			model = LinearRegression(fit_intercept=True)
			X = np.array(cohort_size)
			Y = np.array(rounds)
			# fit model
			model.fit(X[:, np.newaxis], Y)
			xfit = np.linspace(2, 40, 100)
			yfit = model.predict(xfit[:, np.newaxis])
			# plot model
			plt.scatter(cohort_size, rounds)
			plt.plot(xfit, yfit, label=str(iidness) + '% IID data, linear model')


# normal plot

plotx(x, 1)
plt.xticks(np.arange(0, 41, step=5))  # Set label locations.
plt.grid(b=True, which='both', axis='y')
plt.xlabel('Cohort Size')
plt.ylabel('Average Number of Rounds to Reach 99% Accuracy')
plt.title("FL on CNN with MNIST and Partially IID Client Data") # LR 0.1
plt.legend()

# plt.suptitle('Batch 6025730: ' + str(trials) + ' Trials', size=20)
plt.savefig('results/' + str(batch) + '_rounds_vs_cohortsize_99.png')


# standard deviation plot

plt.clf()
plotx(x, 2)
plt.xticks(np.arange(0, 41, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Average Number of Rounds to Reach 99% Accuracy')
plt.title("MNIST, Partially IID Data, LR 0.1")
plt.legend()

plt.suptitle('Batch 6025730: ' + str(trials) + ' Trials', size=20)
plt.savefig('results/' + batch_name + '/' + str(batch) + '.rounds_vs_cohortsize_99stdev.png')


# max and min plot

plt.clf()
plotx(x, 3)
plt.xticks(np.arange(0, 41, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Average Number of Rounds to Reach 99% Accuracy')
plt.title("MNIST, Partially IID Data, LR 0.1")
plt.legend()

plt.suptitle('Batch 6025730: ' + str(trials) + ' Trials', size=20)
plt.savefig('results/' + batch_name + '/' + str(batch) + '.rounds_vs_cohortsize_99maxmin.png')


# linear regression

plt.clf()
plotx(x, 4)
plt.xticks(np.arange(0, 41, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Average Number of Rounds to Reach 99% Accuracy')
plt.title("MNIST, Partially IID Data, LR 0.1")
plt.legend()

plt.suptitle('Batch 6025730: ' + str(trials) + ' Trials', size=20)
plt.savefig('results/' + batch_name + '/' + str(batch) + '.rounds_vs_cohortsize_99reg.png')