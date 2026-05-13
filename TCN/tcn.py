"""Backward-compatible re-export of the canonical TCN classes."""

from dtnn.modules.tcn import Chomp1d, TemporalBlock, TemporalConvNet

__all__ = ["Chomp1d", "TemporalBlock", "TemporalConvNet"]
