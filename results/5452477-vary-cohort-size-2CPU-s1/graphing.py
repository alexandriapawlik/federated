import numpy as np
import math
import csv
import matplotlib.pyplot as plt

cohort_size = [5,10,20,50]
cohort_size_2 = [5,10,20]

batch = 5420860
batch_name = '5420860-vary-cohort-size-1CPU-s1'
test = [1, 2, 3]

# for full batch:
# number of rounds vs cohort size for all three schemas
rounds_s1 = []
rounds_s3 = []
rounds_s4 = []

for i in range(len(test)):

	if i == 2:
		batch = 5458300
		batch_name = '5458300-vary-cohort-size-1CPU-s1'

	# for each test:
	# accuracy vs round number for all schemas
	round_num = []
	accuracy_s1 = []
	# accuracy_s3 = []
	# accuracy_s4 = []

	filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test[i]) + '.s1out.csv'
	with open(filename,'r') as csvfile:
		data = csv.reader(csvfile, delimiter=',')
		header = next(data)
		for row in data:
			round_num.append(int(row[0]))
			accuracy_s1.append(int(float(row[4]) * 100))
		
	# append last line to full batch data
	rounds_s1.append(round_num[-1])

	# filename = 'results/' + str(batch) + '/' + str(batch) + '.' + str(test[i]) + '.s3out.csv'
	# with open(filename,'r') as csvfile:
	# 	data = csv.reader(csvfile, delimiter=',')
	# 	header = next(data)
	# 	for row in data:
	# 		round_num.append(int(row[0]))
	# 		accuracy_s3.append(int(row[4]) * 100)
		
	# # append last line to full batch data
	# rounds_s3.append(round_num[-1])

	# filename = 'results/' + str(batch) + '/' + str(batch) + '.' + str(test[i]) + '.s4out.csv'
	# with open(filename,'r') as csvfile:
	# 	data = csv.reader(csvfile, delimiter=',')
	# 	header = next(data)
	# 	for row in data:
	# 		round_num.append(int(row[0]))
	# 		accuracy_s4.append(int(row[4]) * 100)
		
	# # append last line to full batch data
	# rounds_s4.append(round_num[-1])

	# get cohort size
	# filename = 'results/' + str(batch) + '/' + str(batch) + '.' + str(test[i]) + '.config.csv'
	# with open(filename,'r') as csvfile:
	# 	data = csv.reader(csvfile, delimiter=',')
	# 	header = next(data)
	# 	cohort_size.append(next(data)[0])

	# data for cohort size specific plot
	# max_round = max(round_num) + 1
	max_round = rounds_s1[-1] + 1
	round_num = range(1, max_round)
	plt.clf()
	plt.plot(round_num, accuracy_s1, label='Schema 1: Clients partially IID')
	# plt.plot(round_num, accuracy_s3, label='Schema 3: Sharding')
	# plt.plot(round_num, accuracy_s4, label='Schema 4: IID')
	plt.xlabel('Round Number')
	plt.ylabel('SCA Accuracy')
	plt.title('Round Accuracy for Cohort Size ' + str(cohort_size[i]) + ', 1 CPU')
	plt.legend()
	# plt.show()
	plt.savefig('results/' + batch_name + '/' + str(batch) + '.' + str(test[i]) + '.accuracy_vs_round.png')


# get 2CPU data

batch = 5452477
batch_name = '5452477-vary-cohort-size-2CPU-s1'
test = [1, 2, 3, 4]

rounds_s1_2 = []

for i in range(len(test)):
	# for each test:
	# accuracy vs round number for all schemas
	round_num = []
	accuracy_s1_2 = []

	filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(test[i]) + '.s1out.csv'
	with open(filename,'r') as csvfile:
		data = csv.reader(csvfile, delimiter=',')
		header = next(data)
		for row in data:
			round_num.append(int(row[0]))
			accuracy_s1_2.append(int(float(row[4]) * 100))
		
	# append last line to full batch data
	rounds_s1_2.append(round_num[-1])

	# data for cohort size specific plot
	# max_round = max(round_num) + 1
	max_round = rounds_s1_2[-1] + 1
	round_num = range(1, max_round)
	plt.clf()
	plt.plot(round_num, accuracy_s1_2, label='Schema 1: Clients partially IID')
	# plt.plot(round_num, accuracy_s3, label='Schema 3: Sharding')
	# plt.plot(round_num, accuracy_s4, label='Schema 4: IID')
	plt.xlabel('Round Number')
	plt.ylabel('SCA Accuracy')
	plt.title('Round Accuracy for Cohort Size ' + str(cohort_size[i]) + ', 2 CPU')
	plt.legend()
	# plt.show()
	plt.savefig('results/' + batch_name + '/' + str(batch) + '.' + str(test[i]) + '.accuracy_vs_round.png')

# data for efficiency by cohort size plot
plt.clf()

plt.ylim(4, 10)

plt.plot(cohort_size_2, rounds_s1, label='1 CPU')
# plt.plot(cohort_size, rounds_s3, label='Schema 3: Sharding')
# plt.plot(cohort_size, rounds_s4, label='Schema 4: IID')
plt.plot(cohort_size, rounds_s1_2, label='2 CPU')
plt.xlabel('Cohort Size')
plt.ylabel('Number of Rounds to Reach 99% Accuracy')
plt.title('Model Efficiency for Varying Cohort Size, Clients with Partially IID Data')
plt.legend()
plt.savefig('results/' + batch_name + '/' + str(batch) + '.rounds_vs_cohortsize.png')