"""Loss interfaces for the public WCHG-Net-2 release.

Core loss details are intentionally redacted. Class/function names are kept
for readability of the training script and paper terminology.
"""

import torch
import torch.nn as nn


__all__ = [
    "comp_dist",
    "OriTripletLoss",
    "QuarCenterTripletLoss",
]


class RedactedImplementationError(NotImplementedError):
    """Raised when a protected WCHG-Net loss is executed."""


def comp_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise distance placeholder."""
    raise RedactedImplementationError(
        "Distance/loss internals are redacted in WCHG-Net-2."
    )


class OriTripletLoss(nn.Module):
    """Triplet-loss interface retained for script compatibility."""

    def __init__(self, batch_size, margin=0.3):
        super().__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, inputs, targets):
        raise RedactedImplementationError(
            "OriTripletLoss is not included in WCHG-Net-2."
        )


class QuarCenterTripletLoss(nn.Module):
    """Quad-center triplet loss (Lqct), public placeholder."""

    def __init__(self, k_size, margin=0, t=0):
        super().__init__()
        self.k_size = k_size
        self.margin = margin
        self.t = t

    def forward(self, inputs, targets):
        raise RedactedImplementationError(
            "Lqct is redacted in WCHG-Net-2."
        )
