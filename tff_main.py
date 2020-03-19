import iid
import by_label
import sys

# disable CPU (enable AVX/FMA) warning on Mac
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print()
print("IID partitioning:")
print("60,000 samples randomly distributed to 100 clients")
print()

partitioner = iid.IID_Partitioner()
partitioner.go(int(sys.argv[1]))
# partitioner.go(1)

print()
print("Non-IID partitioning:")
print("60,000 samples divided by label, split into 200 shards, and randomly distributed to 100 clients")
print()

partitioner = by_label.Label_Partitioner()
partitioner.go(int(sys.argv[1]))
# partitioner.go(1)