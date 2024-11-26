#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Layer normalization module."""

import torch


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, nout, dim=-1, eps=1e-12):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )

class AdaptiveLayerNorm(torch.nn.Module):
    """Adaptive Layer normalization module.

    Args:
        ncond (int): Condition dim size.
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, ncond, nout, dim=-1, eps=1e-12):
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = torch.nn.Linear(ncond, 2 * nout)
        self.norm = LayerNorm(nout, dim, eps=eps)
        self.nout = nout

    def forward(self, x, g):
        weight, bias = torch.split(
            self.project_layer(g),
            split_size_or_sections=self.nout,
            dim=-1,
        )
        return weight * self.norm(x) + bias
