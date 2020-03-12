import iid
import by_label
import sys

# print()
# print("IID partitioning:")
# print("60,000 samples randomly distributed to 100 clients")
# print()

# partitioner = iid.IID_Partitioner()
# partitioner.go(int(sys.argv[1]))

print()
print("Non-IID partitioning:")
print("60,000 samples divided by label, split into 200 shards, and randomly distributed to 100 clients")
print()

partitioner = by_label.Label_Partitioner()
partitioner.go(int(sys.argv[1]))