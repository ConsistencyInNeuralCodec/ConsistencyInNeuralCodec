# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Normalization modules."""

import typing as tp

import einops
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return x


class ConditionalNorm(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        norm_dim: int,
        epsilon: Optional[float] = 1e-5,
        base_norm_type: Optional[str] = "time_group_norm",
    ):
        super().__init__()
        self.condition_dim = condition_dim
        self.norm_dim = norm_dim
        self.epsilon = epsilon
        # self.W_scale = nn.Linear(self.condition_dim, self.norm_dim)
        # self.W_bias = nn.Linear(self.condition_dim, self.norm_dim)
        self.condition_linear = nn.Linear(condition_dim, norm_dim * 2)
        self.base_norm_type = base_norm_type
        self.init_base_norm(base_norm_type)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.constant_(self.W_scale.weight, 0.0)
        # torch.nn.init.constant_(self.W_scale.bias, 1.0)
        # torch.nn.init.constant_(self.W_bias.weight, 0.0)
        # torch.nn.init.constant_(self.W_bias.bias, 0.0)
        self.condition_linear.bias.data[self.norm_dim: ] = 0
        self.condition_linear.bias.data[: self.norm_dim] = 1

    def init_base_norm(self, base_norm_type):
        if base_norm_type == "time_group_norm":
            self.base_norm = nn.GroupNorm(num_groups=1, num_channels=self.norm_dim, eps=1e-05, affine=False)
        elif base_norm_type == "layer_norm":
            self.base_norm = nn.LayerNorm(normalized_shape=self.norm_dim, eps=1e-05, elementwise_affine=False)
        elif base_norm_type == "instance_norm":
            self.base_norm = nn.InstanceNorm1d(num_features=self.norm_dim, eps=1e-05, momentum=0.1, affine=False)
        return
    
    def forward_base_norm(self, x):
        # x in (B, C, T)
        if self.base_norm_type == "time_group_norm":
            return self.base_norm(x)
        elif self.base_norm_type == "layer_norm":
            return self.base_norm(x.transpose(1, 2)).transpose(1, 2)
        elif self.base_norm_type == "normal":
            mean = x.mean(dim=1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
            std = (var + self.epsilon).sqrt()
            return (x - mean) / std
        elif self.base_norm_type == "instance_norm":
            return self.base_norm(x)
    
    def forward(self, x, **kwargs):
        # x in (B, C, T)
        x = self.forward_base_norm(x)
        y = x # x in (B, C, T)
        condition_embedding = kwargs.get("condition_embedding", None)
        # scale = self.W_scale(condition_embedding)
        # bias = self.W_bias(condition_embedding)
        if condition_embedding is not None:
            condition = self.condition_linear(condition_embedding).unsqueeze(2)  # (B, 2d, 1)
            gamma, beta = condition.chunk(2, 1)  # (B, d, 1)
        else:
            gamma, beta = 1.0, 0.0
        # print(666, x.shape, condition_embedding.shape, y.shape, scale.shape, bias.shape)
        # print(666, x.sum(-1), condition_embedding.sum(-1), y.sum(-1), scale.sum(-1), bias.sum(-1))
        # y *= scale.unsqueeze(1)
        # y += bias.unsqueeze(1)
        # print(666, y.shape, condition.shape, gamma.shape, beta.shape)
        y = gamma * y + beta
        # y = y.transpose(1, 2) # x in (B, C, T)
        # print(f"666 {self.base_norm} {x.shape} {gamma.shape} {beta.shape}")
        return y
