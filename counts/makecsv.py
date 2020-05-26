import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

cols = ['IIDNESS','SEED','iid_av','iid_max','0_av','0_max','1_av','1_max','2_av','2_max','3_av','3_max','4_av','4_max','5_av','5_max','6_av','6_max','7_av','7_max','8_av','8_max','9_av','9_max','noniid_av','std_over_labels']
dataarr = []

iid = list(range(10,100,10))
seed = [1,5,10,18,25]

for i in iid:
	for s in seed:
		thisdata = [i,s]
		filename = 'counts/count.' + str(i) + '.' + str(s) + '.txt'
		file1 = open(filename,"r")
		# read until IID data
		for x in range(16):
			file1.readline()
		# add iid average and max
		a = file1.readline()
		thisdata.append(float(a.split()[1]))
		m = file1.readline()
		thisdata.append(int(m.split()[1]))
		# get to by label section
		file1.readline()
		# 3 line skips per label
		for l in range(10):
			for x in range(3):
				file1.readline()
			# add label average and max
			a = file1.readline()
			thisdata.append(float(a.split()[1]))
			m = file1.readline()
			thisdata.append(int(m.split()[1]))
		# last noniid data
		file1.readline()
		a = file1.readline()
		thisdata.append(float(a.split()[3]))
		# complete file
		dataarr.append(np.array(thisdata))
		file1.close()

# get standard deviations for label averages
for i in range(len(dataarr)):
	label_avs = dataarr[i][np.array([4,6,8,10,12,14,16,18,20,22])] # slice inner numpy arrays
	dataarr[i] = np.append(dataarr[i], np.std(label_avs))

# convert data to numpy array and make dataframe
df = pd.DataFrame(np.array(dataarr), columns=cols)
# print(df)
df.to_csv('counts/counts.csv')

# plot
x = df['IIDNESS']
iid_av = df['iid_av']
noniid_av = df['noniid_av']
noniid_std = df['std_over_labels']

plt.figure(figsize=(16, 8))
plt.subplot(121)

# y1
model1 = LinearRegression(fit_intercept=True)
model1.fit(x[:, np.newaxis], iid_av)
xfit = np.linspace(0, 100, 500)
yfit = model1.predict(xfit[:, np.newaxis])
plt.scatter(x, iid_av)
plt.plot(xfit, yfit, label='IID data average multiplicity')

# y2
model2 = LinearRegression(fit_intercept=True)
model2.fit(x[:, np.newaxis], noniid_av)
xfit = np.linspace(0, 100, 500)
yfit = model2.predict(xfit[:, np.newaxis])
plt.scatter(x, noniid_av, label='Non-IID data average multiplicity')
# plt.plot(xfit, yfit, label='Non-IID data average multiplicity')

plt.title('Average Counts for IID and Non-IID Portions of Client Data')
plt.xlabel('Percent Data IID')
plt.ylabel('Datapoint Count')
plt.legend()


plt.subplot(122)

# y3
model3 = LinearRegression(fit_intercept=True)
model3.fit(x[:, np.newaxis], noniid_std)
xfit = np.linspace(0, 100, 500)
yfit = model3.predict(xfit[:, np.newaxis])
plt.scatter(x, noniid_std)
# plt.plot(xfit, yfit)

plt.title('Standard Deviation of Average Counts for Each of 10 Lables')
plt.xlabel('Percent Data IID')
# plt.ylabel('Datapoint Count')

plt.suptitle('Counting Data Multiplicities for Partially IID Partitioning of MNIST Data to 100 Clients', size=20)
plt.savefig('counts/counts_vs_iidness.png')