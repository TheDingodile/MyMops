import numpy as np
import matplotlib as plt

def mnist():
    # Define a transform to normalize the data
    d = np.load("data/corruptmnist/train_2.npz")
    lst = d.files
    for item in lst:
        print(item)
        print(d[item].shape)
    return train, test

mnist()
