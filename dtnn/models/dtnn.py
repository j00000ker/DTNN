"""DTNN: Differential Transformer Neural Network for high-frequency stock movement prediction."""

import torch
from torch import nn
from einops import repeat

from dtnn.modules.transformer import Transformer
from dtnn.modules.tcn import TCN
from dtnn.modules.tabl import TABL


class DTNN(nn.Module):
    """Differential Transformer Neural Network.

    Combines three parallel encoders (TCN, TABL, identity passthrough) whose
    outputs are concatenated, differenced along the time axis, and fed through a
    Transformer encoder with a class token, followed by an MLP head for
    classification.  Returns raw logits (use ``CrossEntropyLoss`` for training).

    Parameters
    ----------
    time_slices : int
        Number of time steps in each input window.
    num_classes : int
        Number of output classes (default 2).
    dim : int
        Feature dimension of the input / internal representation.
    kernel_size : int
        Convolution kernel size for the TCN branch.
    num_channels : list of int or None
        TCN channel sizes. When None, defaults to ``[2*dim, 2*dim, 2*dim]``.
    depth : int
        Number of Transformer encoder layers.
    heads : int
        Number of attention heads.
    mlp_dim : int or None
        Hidden dimension of the Transformer feed-forward block. When None,
        uses ``2 * dim`` (the recommended default for the paper configuration).
    pool : {'cls', 'mean'}
        Pooling method: class token or mean pooling.
    dim_head : int
        Dimension per attention head.
    dropout : float
        Dropout rate after the attention projection.
    emb_dropout : float
        Dropout rate applied after the positional embedding.
    """

    def __init__(
        self,
        *,
        time_slices,
        num_classes=2,
        dim=30,
        kernel_size=2,
        num_channels=None,
        depth=3,
        heads=32,
        mlp_dim=None,
        pool='cls',
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        if pool not in {'cls', 'mean'}:
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}")

        if mlp_dim is None:
            mlp_dim = 2 * dim

        self.pos_embedding = nn.Parameter(torch.zeros(1, time_slices + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        tcn_channel = [2 * dim] * 3 if not num_channels else num_channels
        self.tcn_emb = TCN(dim, dim, tcn_channel, kernel_size=kernel_size, dropout=dropout)
        self.tabl_emb = nn.Sequential(
            TABL(time_slices, dim, time_slices, 2 * dim),
            TABL(time_slices, 2 * dim, 2 * time_slices, 2 * dim),
            TABL(2 * time_slices, 2 * dim, time_slices, dim),
        )
        self.emb = nn.Linear(dim * 3, dim)
        # Identity-init the first `dim` columns so the direct passthrough branch
        # is passed through unchanged at the start of training.  Depends on the
        # torch.cat order placing identity branch first (see forward).
        nn.init.eye_(self.emb.weight)
        nn.init.constant_(self.emb.bias, 0)

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.GELU(),
        )

    def forward(self, x):
        b, n, _ = x.shape

        x1 = self.tcn_emb(x)
        x2 = self.tabl_emb(x)
        x = torch.cat((x, x1, x2), dim=2)

        x = self.emb(x)

        # Sequential differencing along time axis (reversed order).
        # Equivalent to: for i in reversed(range(1, n)):
        #     x[:, i, :] = x[:, i, :] - x[:, i-1, :]
        x = torch.cat([x[:, :1, :], torch.diff(x, dim=1)], dim=1)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)
