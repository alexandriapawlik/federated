import numpy as np
import math
import csv
import matplotlib.pyplot as plt

batch = 5699753
batch_name = str(batch)

# test 1
plt.subplot(121)
i = 1
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
ymax = max(accuracy)
plt.axhline(ymax, label='max: ' + str(ymax) + '%', color='g')
plt.legend()
plt.xlabel('Round Number')
plt.ylabel('SCA Accuracy')
plt.title('Test ' + str(i) + ': cohort size 5, 0% IID')

# test 2
plt.subplot(122)
i = 2
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
ymax = max(accuracy)
plt.axhline(ymax, label='max: ' + str(ymax) + '%', color='g')
plt.legend()
plt.xlabel('Round Number')
plt.title('Test ' + str(i) + ': cohort size 10, 0% IID')

# plt.show()
plt.savefig('results/' + batch_name + '/' + str(batch) + '.accuracy_vs_round_timedout.png')