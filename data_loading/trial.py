import idx2numpy
import numpy as np

file = "/home/domi/ml-training-technique/data/train-images.idx3-ubyte"
file = "/home/domi/ml-training-technique/data/train-labels.idx1-ubyte"
arr = idx2numpy.convert_from_file(file)

for i in range(10):
    print(arr[i])
    print(arr[i].shape)
