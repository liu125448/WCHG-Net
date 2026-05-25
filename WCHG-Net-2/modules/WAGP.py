"""Public API placeholder for WAGP.

The full Wavelet-Aware Gated Purification implementation is intentionally
redacted in this public release. The class name and constructor signature are
kept so that the repository structure and model wiring remain visible.
"""

import torch
import torch.nn as nn


class RedactedImplementationError(NotImplementedError):
    """Raised when a protected WCHG-Net component is executed."""


class AGP_W_M(nn.Module):
    """Wavelet-Aware Gated Purification (WAGP), public placeholder.

    Args are accepted for API compatibility with the private implementation.
    The forward pass is not included in WCHG-Net-2.
    """

    def __init__(
        self,
        channels: int,
        split_ratios=None,
        rerank_method: str = "learnable",
        wavename: str = "haar",
        fasa_with_ll_attn: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.split_ratios = split_ratios or [1, 1]
        self.rerank_method = rerank_method
        self.wavename = wavename
        self.fasa_with_ll_attn = fasa_with_ll_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RedactedImplementationError(
            "WAGP is redacted in WCHG-Net-2. "
            "This public release keeps only the module interface."
        )
