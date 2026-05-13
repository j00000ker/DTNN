"""Data loading and preprocessing utilities for the DTNN stock prediction pipeline."""

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import WeightedRandomSampler


def prepare_x(data):
    """Build feature matrix from raw stock data.

    Takes the first 40 rows, transposes, and appends pairwise products of
    adjacent columns (20 product features), yielding 60 features per time step.
    """
    df1 = data[:40, :].T
    products = df1[:, 0::2] * df1[:, 1::2]
    df1 = np.hstack((df1, products))
    return np.array(df1)


def get_label(data):
    """Extract the last 5 rows as label basis."""
    lob = data[-5:, :].T
    return lob


def data_classification(X, Y, T):
    """Create sliding windows of length T from time-series data."""
    N, D = X.shape
    df = np.array(X)
    dY = np.array(Y)

    dataY = dY[T - 1:N]
    # Build windows via stride tricks (zero-copy view, then copy to make contiguous)
    shape = (N - T + 1, T, D)
    strides = (df.strides[0], df.strides[0], df.strides[1])
    dataX = np.lib.stride_tricks.as_strided(df, shape=shape, strides=strides).copy()

    return dataX, dataY


class StockDataset(data.Dataset):
    """PyTorch Dataset for high-frequency stock data.

    Reads raw stock data, builds sliding windows of length T, and extracts
    labels from column ``k`` (zero-indexed).  Expects labels to be 1-indexed in
    the raw data and converts them to 0-indexed.
    """

    def __init__(self, data_array, k, num_classes, T):
        x = prepare_x(data_array)
        y = get_label(data_array)
        x, y = data_classification(x, y, T)
        y = y[:, k] - 1  # convert from 1-indexed to 0-indexed
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.squeeze(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def get_sampler(labels, num_samples=None):
    """Create a WeightedRandomSampler to handle class imbalance.

    Parameters
    ----------
    labels : torch.Tensor
        Integer class labels.
    num_samples : int or None
        Number of samples per epoch. When None, uses the full dataset size.
    """
    class_counts = [0, 0, 0]
    for i in range(3):
        class_counts[i] = (labels == i).sum()
    print(f'target train 0/1/2: {class_counts[0]}/{class_counts[1]}/{class_counts[2]}')

    weights = np.zeros(len(labels))
    for i in range(len(labels)):
        weights[i] = 1.0 / class_counts[int(labels[i])]

    if num_samples is None:
        num_samples = len(labels)
    sampler = WeightedRandomSampler(weights, num_samples)
    return sampler
