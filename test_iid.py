import iid

print()
print("IID partitioning:")
print("60,000 samples randomly distributed to 100 clients")
print()

partitioner = iid.IID_Partitioner()
partitioner.go()