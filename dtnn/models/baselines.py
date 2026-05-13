"""Baseline models used for comparison with DTNN.

This module reproduces the models from the paper's comparison experiments.
All models return raw logits for use with ``CrossEntropyLoss``.
"""

import torch
from torch import nn

from dtnn.modules.tabl import TABL


class SVM(nn.Module):
    """Linear classifier (multi-class SVM / logistic regression)."""

    def __init__(self, time_slices, dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(time_slices * dim, num_classes)

    def forward(self, x):
        b, n, d = x.shape
        x = x.reshape(b, n * d)
        return self.linear(x)


class MLP(nn.Module):
    """Two-layer perceptron with LeakyReLU."""

    def __init__(self, time_slices, dim, num_classes, n_hidden=128):
        super().__init__()
        self.hidden = nn.Linear(time_slices * dim, n_hidden)
        self.predict = nn.Linear(n_hidden, num_classes)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        b, n, d = x.shape
        x = x.reshape(b, n * d)
        x = self.activation(self.hidden(x))
        x = self.predict(x)
        return x


class LSTM(nn.Module):
    """Single-layer LSTM followed by a linear classifier."""

    def __init__(self, time_slices, dim, num_classes, n_hidden=40):
        super().__init__()
        self.lstm = nn.LSTM(input_size=dim, hidden_size=n_hidden, batch_first=True)
        self.activation = nn.LeakyReLU()
        self.classifier = nn.Linear(time_slices * n_hidden, num_classes)

    def forward(self, x):
        output, (ht, ct) = self.lstm(x)
        output = self.activation(output)
        b, n, d = output.shape
        output = output.reshape(b, n * d)
        output = self.classifier(output)
        return output


class CNN(nn.Module):
    """Stacked 2D/1D CNN with max-pooling and a linear classifier head."""

    def __init__(self, time_slices, dim, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, dim)),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        # Determine the flattened size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, time_slices, dim)
            flat_size = self.conv(dummy.unsqueeze(1)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat_size, 32),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        return self.head(x)


class CNN_LSTM(nn.Module):
    """CNN feature extractor followed by an LSTM and linear classifier."""

    def __init__(self, time_slices, dim, num_classes):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, dim), padding=(2, 0))
        self.layer2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
        self.layer3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.layer4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.activation = nn.PReLU()
        self.lstm = nn.LSTM(input_size=time_slices, hidden_size=32, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32, 64),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        b, n, d = x.shape
        x = self.layer1(x.unsqueeze(1))
        x = self.layer2(x.squeeze(3))
        x = self.layer3(x)
        x = self.layer4(x)
        output, (ht, ct) = self.lstm(x)
        output = self.activation(output)
        return self.classifier(output.reshape(b, -1))


class C_TABL(nn.Module):
    """Three-layer TABL stack for classification (pure TABL baseline)."""

    def __init__(self, time_slices, dim, num_classes):
        super().__init__()
        self.tabl_stack = nn.Sequential(
            TABL(time_slices, dim, 60, 10),
            nn.ReLU(),
            TABL(60, 10, 120, 5),
            nn.ReLU(),
            TABL(120, 5, 3, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.tabl_stack(x)
        x = x.squeeze(2)
        return x
