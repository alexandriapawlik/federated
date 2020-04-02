import each_client_partially_iid
import some_clients_iid
import sys

# disable CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

partitioner = each_client_partially_iid.Partitioner1()
partitioner.go(int(sys.argv[1]))

# partitioner = some_clients_iid.Partitioner2()
# partitioner.go(int(sys.argv[1]))