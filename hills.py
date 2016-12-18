from numpy import loadtxt
from glob import glob
import os
from random import sample

hills_train = []
if os.path.exists(os.path.join("hills", "training")):
    for hill in glob(os.path.join("hills", "training", "*.txt")):
        hills_train.append(loadtxt(hill))

hills_test = []
if os.path.exists(os.path.join("hills", "test")):
    for hill in glob(os.path.join("hills", "test", "*.txt")):
        hills_test.append(loadtxt(hill))


def sample_train(N=10):
    return sample(hills_train, N)

def sample_test(N=10):
    return sample(hills_test, N)