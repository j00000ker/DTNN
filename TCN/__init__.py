"""TCN module -- re-exports from the dtnn package for backward compatibility."""

from dtnn.modules.tcn import TemporalConvNet, TemporalBlock, Chomp1d

__all__ = ["TemporalConvNet", "TemporalBlock", "Chomp1d"]
