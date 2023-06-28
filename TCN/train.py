import os
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# load packages
import pandas as pd
import pickle
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch import nn, einsum
import torch.nn.functional as F
from TCN.tcn import TemporalConvNet
from torch.utils.data import WeightedRandomSampler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model import DTNN


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
# N, D = X_train.shape

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_sampler(label):
    class_counts=[0,0,0]
    for i in range(3):
        class_counts[i]=(label==i).sum()
    print('target train 0/1/2: {}/{}/{}'.format(class_counts[0], class_counts[1], class_counts[2]))
    weights=np.zeros(len(label))
    for i in range(len(label)):
        weights[i]=1./class_counts[int(label[i].detach())]
    sampler=WeightedRandomSampler(weights,3)
    return sampler




def prepare_x(data):
    df1 = data[:40, :].T
    for i in range(20):
        df1 = np.hstack((df1, (df1[:, 2 * i] * df1[:, 2 * i + 1]).reshape(-1, 1)))
    return np.array(df1)


def get_label(data):
    lob = data[-5:, :].T
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY


def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, k, num_classes, T):
        """Initialization"""
        self.k = k
        self.num_classes = num_classes
        self.T = T

        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:, self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.squeeze(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]


def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs, model_name, device):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in range(epochs):

        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, './' + model_name)
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch + 1}')

    return train_losses, test_losses



if __name__ == '__main__':

    print(device)
    batch_size = 64
    LR = 1e-4
    depth=3
    k=1
    print('depth={},LR={},k={}.'.format(depth,LR,k))
    model_name = 'd3^lr1e-4_pytorch_diff75'
    path=''
    dec_data = np.loadtxt(path +  'Train_Dst_NoAuction_DecPre_CF_7.txt')
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    dec_test1 = np.loadtxt(path +  'Test_Dst_NoAuction_DecPre_CF_7.txt')
    dec_test2 = np.loadtxt(path + 'Test_Dst_NoAuction_DecPre_CF_8.txt')
    dec_test3 = np.loadtxt(path +  'Test_Dst_NoAuction_DecPre_CF_9.txt')
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    print(dec_train.shape, dec_val.shape,
          dec_test.shape
          )

    dataset_train = Dataset(data=dec_train, k=k, num_classes=3, T=100)
    dataset_val = Dataset(data=dec_val, k=k, num_classes=3, T=100)
    dataset_test = Dataset(data=dec_test, k=k, num_classes=3, T=100)
    sampler=get_sampler(dataset_train.y)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    #train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    print(dataset_train.x.shape, dataset_train.y.shape)

    tmp_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)
    for x, y in tmp_loader:
        print(x)
        print(y)
        print(x.shape, y.shape)
        break

    print(dataset_train.x.shape)
    model = DTNN(time_slices=dataset_train.x.shape[1], num_classes=3, dim=dataset_train.x.shape[2],
                    depth=depth, heads=32, mlp_dim=2 * dataset_train.x.shape[2])
    model.to(device)

    summary(model, [1, 100, 60])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    train_losses, val_losses = batch_gd(model, criterion, optimizer,
                                        train_loader, val_loader, epochs=150, model_name=model_name, device=device)

    plt.figure(figsize=(15, 6))
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='validation loss')
    plt.savefig(model_name + '_loss.jpg')


    model = torch.load(model_name)

    n_correct = 0.
    n_total = 0.
    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    test_acc = n_correct / n_total
    print(f"Test acc: {test_acc:.4f}")

    # model = torch.load('best_val_model_pytorch')
    all_targets = []
    all_predictions = []

    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    print('accuracy_score:', accuracy_score(all_targets, all_predictions))
    print(classification_report(all_targets, all_predictions, digits=4))
