each_client_partially_iid = __import__('1_each_client_partially_iid')
some_clients_iid = __import__('2_some_clients_iid')
shard = __import__('3_shard')
iid = __import__('4_iid')
import sys
from datetime import datetime

# disable CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

p1 = each_client_partially_iid.Partitioner1()
# p2 = some_clients_iid.Partitioner2()
p3 = shard.Partitioner3()
p4 = iid.Partitioner4()

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

# # p2.go(test, batch)
# # print(datetime.now())

# p3.go(test, batch)
# print(datetime.now())

# p4.go(test, batch)
# print(datetime.now())