

import torch
import torch.utils.data as data_utils
import numpy as np

import h5py
from scipy.io import loadmat


class Scaler:

    def __init__(self, x):
        # Numpy array input to tensor
        x = torch.from_numpy(x).double()

        # Calculate mean and standard deviation of train
        self.mean_vec = torch.mean(x, dim=0)
        self.sd_vec = torch.std(x, dim=0)

    def to(self, device):
        self.mean_vec = self.mean_vec.to(device)
        self.sd_vec = self.sd_vec.to(device)

    def normalize(self, x):
        return (x - self.mean_vec) / self.sd_vec


def np_shuffle_arrays(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def read_mat(path: str, transpose=True, print_dim=False):

    # Read data - different .mat versions: first try h5py, then scipy
    try:
        file = h5py.File(path, 'r')
    except OSError:
        file = loadmat(path)

    # Extract X and labels
    X = np.array(file.get('X'))
    labels = np.array(file.get('y'))

    # Transpose data
    if transpose:
        X = X.transpose()
        labels = labels.transpose()

    if print_dim:
        print('Input data dim:')
        print(' X:      {}'.format(X.shape))
        print(' labels: {}'.format(labels.shape))
