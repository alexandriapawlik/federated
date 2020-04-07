import each_client_partially_iid
import some_clients_iid
import shard
import iid
import sys

# disable CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

p1 = each_client_partially_iid.Partitioner1()
p2 = some_clients_iid.Partitioner2()
p3 = shard.Partitioner3()
p4 = iid.Partitioner4()

# p1.go(int(sys.argv[1]))
# p2.go(int(sys.argv[1]))
p3.go(int(sys.argv[1]))
p4.go(int(sys.argv[1]))