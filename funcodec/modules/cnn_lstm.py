import sys
sys.path.append("/home/admin_data/user/model") # for Amphion

import torch
from torch import nn
from einops.layers.torch import Rearrange
from Amphion.models.codec.ns3_codec.facodec import (
    ResidualUnit,
    Activation1d,
    SnakeBeta,
)

import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class CNNLSTMConfig(PretrainedConfig):
    model_type = "cnn-lstm"
    is_composition = True

    def __init__(
        self,
        indim: Optional[int] = 128,
        outdim: Optional[int] = 128,
        head: Optional[int] = 1,
        global_pred: Optional[bool] = False,
        seq_len_second: Optional[bool] = True,
        dilation_list: Sequence[int] = [1, 2, 3],
        dropout: Optional[float] = 0.1,
        **kwargs
    ):
        super().__init__(
            indim=indim,
            outdim=outdim,
            head=head,
            global_pred=global_pred,
            seq_len_second=seq_len_second,
            dilation_list=dilation_list,
            dropout=dropout,
            **kwargs
        )

    def to_dict(self):
        return dict(
            indim=self.indim,
            outdim=self.outdim,
            head=self.head,
            global_pred=self.global_pred,
            seq_len_second=self.seq_len_second,
            dilation_list=self.dilation_list,
            dropout=self.dropout,
        )


# def WNConv1d(*args, **kwargs):
#     return weight_norm(nn.Conv1d(*args, **kwargs))


# class ResidualUnit(nn.Module):
#     def __init__(self, dim: int = 16, dilation: int = 1):
#         super().__init__()
#         pad = ((7 - 1) * dilation) // 2
#         self.block = nn.Sequential(
#             Activation1d(activation=SnakeBeta(dim, alpha_logscale=True)),
#             WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
#             Activation1d(activation=SnakeBeta(dim, alpha_logscale=True)),
#             WNConv1d(dim, dim, kernel_size=1),
#         )

#     def forward(self, x):
#         return x + self.block(x)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        indim, outdim, head, 
        global_pred=False, seq_len_second=True, dropout=0.0,
        dilation_list: Sequence[int] = [1, 2, 3],
        **kwargs
    ):
        super().__init__()
        self.global_pred = global_pred
        # self.model = nn.Sequential(
        #     ResidualUnit(indim, dilation=1),
        #     nn.Dropout(p=dropout),
        #     ResidualUnit(indim, dilation=2),
        #     nn.Dropout(p=dropout),
        #     ResidualUnit(indim, dilation=3),
        #     nn.Dropout(p=dropout),
        #     Activation1d(activation=SnakeBeta(indim, alpha_logscale=True)),
        #     Rearrange("b c t -> b t c"),
        # )
        self.model = nn.Sequential()
        for dilation in dilation_list:
            self.model.append(ResidualUnit(indim, dilation=dilation))
            self.model.append(nn.Dropout(p=dropout))
        self.model.append(Activation1d(activation=SnakeBeta(indim, alpha_logscale=True)))
        self.model.append(Rearrange("b c t -> b t c"))
        self.multihead_linear = nn.Linear(indim, outdim * head)
        self.outdim = outdim
        self.head = head
        self.seq_len_second = seq_len_second

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        if self.seq_len_second:
            # x: [B, T, C]
            x = x.transpose(1, 2)
        # x: [B, C, T]
        # print(f"666 CNNLSTM {x.shape}")
        seq_len = x.shape[2]
        x = self.model(x)
        # x: [B, T, C]
        if self.global_pred:
            outs = torch.mean(x, dim=1, keepdim=False)
        else:
            outs = self.multihead_linear(x)
            if self.head == 1:
                outs = outs.view(batch_size, seq_len, self.outdim)
            else:
                outs = outs.view(batch_size, seq_len, self.head, self.outdim)
        return outs
