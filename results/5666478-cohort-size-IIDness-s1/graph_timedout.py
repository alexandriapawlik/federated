import numpy as np
import math
import csv
import matplotlib.pyplot as plt

batch = 5666478
batch_name = '5666478-cohort-size-IIDness-s1'

# pull data about unfinished tests
tests = [1,2,3,4,5,6,7,18]
plt.figure(figsize=(16, 8))

# test 1
plt.subplot(241)
i = tests[0]
round_num = []
accuracy = []
filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.s1out.csv'
with open(filename,'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	header = next(data)
	for row in data:
		round_num.append(int(row[0]))
		accuracy.append(int(float(row[4]) * 100))
plt.plot(round_num, accuracy)
plt.title('Test ' + str(i) + ': cohort size 5, 0% IID')

# test 2
plt.subplot(242)
i = tests[1]
round_num = []
accuracy = []
filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.s1out.csv'
with open(filename,'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	header = next(data)
	for row in data:
		round_num.append(int(row[0]))
		accuracy.append(int(float(row[4]) * 100))
plt.plot(round_num, accuracy)
plt.title('Test ' + str(i) + ': cohort size 10, 0% IID')

# test 3
plt.subplot(243)
i = tests[2]
round_num = []
accuracy = []
filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.s1out.csv'
with open(filename,'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	header = next(data)
	for row in data:
		round_num.append(int(row[0]))
		accuracy.append(int(float(row[4]) * 100))
plt.plot(round_num, accuracy)
plt.title('Test ' + str(i) + ': cohort size 15, 0% IID')

# test 4
plt.subplot(244)
i = tests[3]
round_num = []
accuracy = []
filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.s1out.csv'
with open(filename,'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	header = next(data)
	for row in data:
		round_num.append(int(row[0]))
		accuracy.append(int(float(row[4]) * 100))
plt.plot(round_num, accuracy)
plt.title('Test ' + str(i) + ': cohort size 20, 0% IID')

# test 5
plt.subplot(245)
i = tests[4]
round_num = []
accuracy = []
filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.s1out.csv'
with open(filename,'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	header = next(data)
	for row in data:
		round_num.append(int(row[0]))
		accuracy.append(int(float(row[4]) * 100))
plt.plot(round_num, accuracy)
plt.xlabel('Round Number')
plt.ylabel('SCA Accuracy')
plt.title('Test ' + str(i) + ': cohort size 5, 20% IID')

# test 6
plt.subplot(246)
i = tests[5]
round_num = []
accuracy = []
filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.s1out.csv'
with open(filename,'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	header = next(data)
	for row in data:
		round_num.append(int(row[0]))
		accuracy.append(int(float(row[4]) * 100))
plt.plot(round_num, accuracy)
plt.title('Test ' + str(i) + ': cohort size 10, 20% IID')

# test 7
plt.subplot(247)
i = tests[6]
round_num = []
accuracy = []
filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.s1out.csv'
with open(filename,'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	header = next(data)
	for row in data:
		round_num.append(int(row[0]))
		accuracy.append(int(float(row[4]) * 100))
plt.plot(round_num, accuracy)
plt.title('Test ' + str(i) + ': cohort size 15, 20% IID')

# test 18
plt.subplot(248)
i = tests[7]
round_num = []
accuracy = []
filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.s1out.csv'
with open(filename,'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	header = next(data)
	for row in data:
		round_num.append(int(row[0]))
		accuracy.append(int(float(row[4]) * 100))
plt.plot(round_num, accuracy)
plt.title('Test ' + str(i) + ': cohort size 10, 80% IID')

# plt.show()
plt.savefig('results/' + batch_name + '/' + str(batch) + '.accuracy_vs_round_timedout.png')