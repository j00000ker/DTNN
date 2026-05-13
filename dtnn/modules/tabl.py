"""Temporal Attention-Based Layer (TABL) for bilinear sequence transformation."""

import torch
from torch import nn


class TABL(nn.Module):
    """Temporal Attention-Based Layer.

    Applies a bilinear projection with learned attention over both the time and
    feature axes.  ``beta`` controls the residual proportion of the pre-attention
    signal blended with the attention-weighted signal.

    Parameters
    ----------
    input_len : int
        Input sequence length (time dimension).
    input_dim : int
        Input feature dimension.
    output_len : int
        Output sequence length.
    output_dim : int
        Output feature dimension.
    beta : float
        Blend ratio between attention-weighted and residual signal (default 0.99).
    """

    def __init__(self, input_len, input_dim, output_len, output_dim, beta=0.99):
        super().__init__()
        self.beta = beta

        self.W1 = nn.Parameter(torch.randn(1, output_len, input_len))
        self.W = nn.Parameter(torch.randn(1, input_dim, input_dim))
        self.softmax = nn.Softmax(dim=-1)
        self.W2 = nn.Linear(input_dim, output_dim)
        self.activ = nn.ReLU()

    def forward(self, x):
        # Broadcasting avoids materialising (b, *, *) copies of W1 and W
        x = torch.bmm(self.W1, x)
        E = torch.bmm(x, self.W)
        A = self.softmax(E)
        x = self.beta * (x * A) + (1 - self.beta) * x
        y = self.activ(self.W2(x))
        return y
