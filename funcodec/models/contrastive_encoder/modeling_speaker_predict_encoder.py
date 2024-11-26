import sys, importlib
import os, time, random
import logging, warnings
import omegaconf
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .configuration_speaker_predict_encoder import SpeakerPredictEncoderConfig
from ..utils import lengths_to_padding_mask, lengths_to_attention_mask

logger = logging.getLogger(__name__)


def mean_pooling(
    x: torch.Tensor,
    lengths: torch.LongTensor,
    attention_mask: Optional[torch.BoolTensor] = None
):
    """
    x: [bsz, length, dim]
    attention_mask: [bsz, length]
        0: mask
        1: not mask
    """
    if attention_mask is None:
        attention_mask = lengths_to_attention_mask(lengths)
    attention_mask = attention_mask.unsqueeze(-1) # [bsz, length, 1]
    x = x * attention_mask
    # mean_x = x.sum(1) / lengths.clamp(min=1).unsqueeze(-1)
    mean_x = x.sum(1) / lengths.unsqueeze(-1)
    return mean_x


@dataclass
class SpeakerPredictEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    loss: torch.FloatTensor = None


class SpeakerPredictEncoderBaseModel(PreTrainedModel):
    config_class = SpeakerPredictEncoderConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[SpeakerPredictEncoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = SpeakerPredictEncoderConfig(**config)
        super().__init__(config=config)

    @staticmethod
    def build_model(
        config: Optional[Union[SpeakerPredictEncoderConfig, Dict]] = None,
        **kwargs
    ) -> "SpeakerPredictEncoderBaseModel":
        if config is not None:
            if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
                config = SpeakerPredictEncoderConfig(**config)
            if config.encoder_type == "linear":
                return SpeakerLinearPredictor(config=config, **kwargs)
            if config.encoder_type == "cnn_lstm":
                return SpeakerCNNLSTMPredictor(config=config, **kwargs)
            else:
                raise NotImplementedError
        raise NotImplementedError


# class GradientReversal(Function):
#     @staticmethod
#     def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
#         ctx.coeff = coeff
#         output = input * 1.0
#         return output

#     @staticmethod
#     def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
#         return grad_output.neg() * ctx.coeff, None


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


class SpeakerLinearPredictor(SpeakerPredictEncoderBaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        if self.config.linear_dim_list is not None:
            linear_dict = OrderedDict()
            linear_dict["fc_0"] = nn.Linear(self.config.linear_dim_list[0], self.config.linear_dim_list[1])
            for i in range(1, len(self.config.linear_dim_list) - 1):
                linear_dict[f"dropout_{i}"] = nn.Dropout(self.config.dropout)
                linear_dict[f"fc_{i}"] = nn.Linear(self.config.linear_dim_list[i], self.config.linear_dim_list[i + 1])
            self.linear = nn.Sequential(linear_dict)
        else:
            self.linear  = nn.Identity()
        if self.config.alpha_for_gradient_reversal is not None:
            self.alpha_for_gradient_reversal = torch.tensor(self.config.alpha_for_gradient_reversal, requires_grad=False)

    def forward(
        self,
        features: Optional[torch.FloatTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        if self.config.gradient_type == "stop_gradient":
            features = features.detach()
        elif self.config.gradient_type == "reverse_gradient":
            # features = GradientReversal.apply(features)
            features = GradientReversal.apply(features, self.alpha_for_gradient_reversal)
        if self.config.merge_embed == "mean_pooling":
            features = mean_pooling(features, feature_lengths, attention_mask) # batch_size, dim
        logits = self.linear(features) # batch_size, num_speaker
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return SpeakerPredictEncoderOutput(
            last_hidden_state=features,
            logits=logits,
            labels=labels,
            loss=loss,
        )


class SpeakerCNNLSTMPredictor(SpeakerPredictEncoderBaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        from funcodec.modules.cnn_lstm import CNNLSTM
        self.model = CNNLSTM(
            indim=self.config.indim,
            outdim=self.config.outdim,
            head=self.config.head,
            global_pred=True,
            seq_len_second=True,
            dropout=self.config.dropout,
        )
        if self.config.linear_dim_list is not None:
            linear_dict = OrderedDict()
            linear_dict["fc_0"] = nn.Linear(self.config.linear_dim_list[0], self.config.linear_dim_list[1])
            for i in range(1, len(self.config.linear_dim_list) - 1):
                linear_dict[f"dropout_{i}"] = nn.Dropout(self.config.dropout)
                linear_dict[f"fc_{i}"] = nn.Linear(self.config.linear_dim_list[i], self.config.linear_dim_list[i + 1])
            self.linear = nn.Sequential(linear_dict)
        else:
            self.linear  = nn.Identity()
        if self.config.alpha_for_gradient_reversal is not None:
            self.alpha_for_gradient_reversal = torch.tensor(self.config.alpha_for_gradient_reversal, requires_grad=False)

    def forward(
        self,
        features: Optional[torch.FloatTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        if self.config.gradient_type == "stop_gradient":
            features = features.detach()
        elif self.config.gradient_type == "reverse_gradient":
            features = GradientReversal.apply(features, self.alpha_for_gradient_reversal)
        elif self.config.gradient_type == "normal":
            pass
        else:
            raise NotImplementedError
        if self.config.merge_embed == "mean_pooling":
            features = self.model(features) # batch_size, dim
            logits = self.linear(features) # batch_size, num_speaker
        else:
            raise NotImplementedError
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return SpeakerPredictEncoderOutput(
            last_hidden_state=features,
            logits=logits,
            labels=labels,
            loss=loss,
        )
