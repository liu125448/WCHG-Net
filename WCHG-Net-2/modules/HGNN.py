"""Public API placeholders for CMGR/HGNN.

The full Chebyshev Multi-band Graph Readout implementation, including
hypergraph construction, spectral readout, multi-granularity graph readout,
and structural regularization details, is intentionally redacted in this
public release.
"""

import torch
import torch.nn as nn


class RedactedImplementationError(NotImplementedError):
    """Raised when a protected WCHG-Net component is executed."""


def _raise_redacted(name: str):
    raise RedactedImplementationError(
        f"{name} is redacted in WCHG-Net-2. "
        "This public release keeps the interface but omits the implementation."
    )


class HypergraphConv(nn.Module):
    """Hypergraph construction layer, public placeholder."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _raise_redacted("HypergraphConv")


def hyperedge_entropy_regularization(*args, **kwargs):
    """Lher placeholder."""
    _raise_redacted("Lher")


def hyperedge_orthogonality_regularization(*args, **kwargs):
    """Lhor placeholder."""
    _raise_redacted("Lhor")


class ChebyshevMultiBandGraphReadout(nn.Module):
    """Chebyshev Multi-band Graph Readout (CMGR), public placeholder."""

    def __init__(
        self,
        in_channels=1024,
        out_channels=1024,
        features_height=24,
        features_width=12,
        edges=256,
        filters=128,
        cheb_order=6,
        bands=3,
        fuse="concat",
        graph_hidden_dims=(768, 512, 256),
        graph_readout="mean",
        lambda_her=2e-4,
        lambda_hor=1e-4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features_height = features_height
        self.features_width = features_width
        self.edges = edges
        self.filters = filters
        self.cheb_order = cheb_order
        self.bands = bands
        self.fuse = fuse
        self.graph_hidden_dims = graph_hidden_dims
        self.graph_readout = graph_readout
        self.lambda_her = float(lambda_her)
        self.lambda_hor = float(lambda_hor)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        _raise_redacted("CMGR")


CMGR = ChebyshevMultiBandGraphReadout
