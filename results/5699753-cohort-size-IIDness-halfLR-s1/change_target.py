import numpy as np
import math
import csv

LIM = 0.9 

batch = 5699753
batch_name = '5699753-cohort-size-IIDness-halfLR-s1'

filename = 'results/' + batch_name + '/' + str(batch) + '.all.target' + str(int(LIM * 100)) + '.csv'
with open(filename, 'w', newline='') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['TEST_NUM','IIDNESS','COHORT_SIZE','LR','TOTAL_ROUNDS','AVERAGE_SECONDS_PER_ROUND'])

	# add test 1-8 values to CSV
	# get IIDNESS,COHORT_SIZE,LR,TOTAL_ROUNDS,AVERAGE_SECONDS_PER_ROUND
	for i in range(1,25):
		# pull from old batch
		if i > 8:
			batch = 5666478
			batch_name = '5666478-cohort-size-IIDness-s1'

		iidness = 0
		cohort_size = 0
		LR = 0

		# iidness, cohort size, LR
		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.config.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				iidness = int(row[-3])
				cohort_size = int(row[0])
				LR = float(row[4])

		times = []
		round_num = []

		filename = 'results/' + batch_name + '/' + str(batch) + '.' + str(i) + '.s1out.csv'
		with open(filename,'r') as csvfile:
			data = csv.reader(csvfile, delimiter=',')
			header = next(data)
			for row in data:
				round_num.append(int(row[0]))
				times.append(float(row[-1]))
				if float(row[-3]) >= LIM:
					break
		# add data to CSV
		writer.writerow([i, iidness, cohort_size, LR, round_num[-1], sum(times)/len(times)])