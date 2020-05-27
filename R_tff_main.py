each_client_partially_iid = __import__('R_1_each_client_partially_iid')
import sys
from datetime import datetime

# disable CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

p1 = each_client_partially_iid.Partitioner1()

# pass test number = 0 if no test number given
if len(sys.argv) < 2:
	test = 0
else:
	test = int(sys.argv[1])

# pass batch number = 0 if no batch number given
if len(sys.argv) < 3:
	batch = 0
else:
	batch = int(sys.argv[2])

print("Test ", test)
print(datetime.now())
print()

p1.go(test, batch)
print(datetime.now())