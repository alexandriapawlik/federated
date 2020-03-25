import iid
import by_label
import sys

# disable CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

partitioner = iid.IID_Partitioner()
partitioner.go(int(sys.argv[1]))
# partitioner.go(1)

partitioner = by_label.Label_Partitioner()
partitioner.go(int(sys.argv[1]))
# partitioner.go(1)