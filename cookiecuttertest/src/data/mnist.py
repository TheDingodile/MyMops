import numpy as np
from os import listdir
import logging
import torch

def mnist(input_path: str):
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    files = listdir(input_path)
    # Define a transform to normalize the data
    inputs = np.empty(shape=(0, 28, 28))
    labels = np.empty(shape=(0))
    test_in = np.empty(shape=(0, 28, 28))
    test_out = np.empty(shape=(0))
    for f in files:
        if f[0:5] == "train":
            inputs = np.concatenate((inputs, np.load("data/raw/" + f)['images']), axis=0)
            labels = np.concatenate((labels, np.load("data/raw/" + f)['labels']), axis=0)
        else:
            test_in = np.concatenate((test_in, np.load("data/raw/" + f)['images']), axis=0)
            test_out = np.concatenate((test_out, np.load("data/raw/" + f)['labels']), axis=0)
    return (inputs, labels), (test_in, test_out)

def load(input_path: str) -> torch.tensor:
    input_train = torch.load(input_path + "/train/inputs")
    labels_train = torch.load(input_path + "/train/labels")
    test_inputs = torch.load(input_path + "/test/inputs")
    test_labels = torch.load(input_path + "/test/labels")
    return torch.tensor(input_train), torch.tensor(labels_train), torch.tensor(test_inputs), torch.tensor(test_labels)