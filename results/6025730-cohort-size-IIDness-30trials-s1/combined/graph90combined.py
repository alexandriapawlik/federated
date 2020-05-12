import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import statistics as stat
from sklearn.linear_model import LinearRegression

plt.figure(figsize=(12, 12))

batch = 6025730
batch_name = '6025730-cohort-size-IIDness-30trials-s1'
trials = 9
tests_per_trial = 75

x = [1,16,31,46,61]  # round nums
cohort_size = [5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40]
y = list(range(len(cohort_size)))

# other batch
_cohort_size = [2,5,10,15,20,25,30]
_iidness = [0, 20, 40, 60, 80, 100]
_batch = 5944196
_batch_name = '5944196-cohort-size-IIDness-8trials-s1'
_trials = 6
_tests_per_trial = 42

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
			test = i + j + 1

			# iterate though trials
			rounds_all_trials = []
			for z in range(trials): # folders
				# get number of rounds
				all_rounds_nums = []
				filename = 'results/' + batch_name + '/r' + str(z) + '/' + str(batch) + '.' + str(test + (tests_per_trial * z)) + '.s1out.csv'
				with open(filename,'r') as csvfile:
					data = csv.reader(csvfile, delimiter=',')
					header = next(data)
					for row in data:
						all_rounds_nums.append(int(row[0]))
						if float(row[-3]) > 0.9:
							break
				# add if not a timeout
				if all_rounds_nums[-1] < 250:
					rounds_all_trials.append(all_rounds_nums[-1])

			# get data from earlier batch if cohort size has data
			if cohort_size[j] in _cohort_size:
				_test = (_iidness.index(iidness) * 7) + _cohort_size.index(cohort_size[j])

				# iterate though trials
				for z in range(_trials): # folders
					# get number of rounds
					all_rounds_nums = []
					filename = 'results/' + _batch_name + '/r' + str(z) + '/' + str(_batch) + '.' + str(_test + (_tests_per_trial * z)) + '.s1out.csv'
					with open(filename,'r') as csvfile:
						data = csv.reader(csvfile, delimiter=',')
						header = next(data)
						for row in data:
							all_rounds_nums.append(int(row[0]))
							if float(row[-3]) > 0.9:
								break
					# add if not a timeout
					if all_rounds_nums[-1] < 250:
						rounds_all_trials.append(all_rounds_nums[-1])

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
			plt.errorbar(cohort_size, rounds, yerr=rounds_std, label=str(iidness) + '% IID data', capsize=4)
		elif option == 3:  # max/min
			plt.errorbar(cohort_size, rounds, yerr=[rounds_min, rounds_max], label=str(iidness) + '% IID data', capsize=4)
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
			plt.plot(xfit, yfit, label=str(iidness) + '% IID data')


# normal plot

plt.subplot(221)
plotx(x, 1)
plt.xticks(np.arange(0, 41, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Average Number of Rounds to Reach 90% Accuracy')
plt.title("MNIST, Partially IID Data, LR 0.1")
plt.legend()

# standard deviation plot

plt.subplot(222)
plotx(x, 2)
plt.xticks(np.arange(0, 41, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Average Number of Rounds to Reach 90% Accuracy')
plt.title("Standard Deviations")
plt.legend()

# max and min plot

plt.subplot(223)
plotx(x, 3)
plt.xticks(np.arange(0, 41, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Average Number of Rounds to Reach 90% Accuracy')
plt.title("Maximum and Minimum")
plt.legend()

# linear regression

plt.subplot(224)
plotx(x, 4)
plt.xticks(np.arange(0, 41, step=5))  # Set label locations.
plt.xlabel('Cohort Size per Global Round')
plt.ylabel('Average Number of Rounds to Reach 90% Accuracy')
plt.title("Linear Regression")
plt.legend()

plt.suptitle('Batches 5944196 and 6025730: ' + str(_trials) + ' + ' + str(trials) + ' Trials', size=20)
plt.savefig('results/' + batch_name + '/combined/rounds_vs_cohortsize_90.png')