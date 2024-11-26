import sys, importlib
import os, time, random, math
import logging, warnings
import omegaconf
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .configuration_phoneme_decoder import PhonemeDecoderConfig
from ..utils import lengths_to_padding_mask, lengths_to_attention_mask

logger = logging.getLogger(__name__)


def frame2ph(mel2ph: Sequence, ph_list: Sequence, max_len: int, padding_ph_idx: int = 0, device=torch.device("cpu")):
    # attention_mask = lengths_to_attention_mask(torch.tensor([len(item) for item in mel2ph], dtype=torch.long, device=device))
    mel2ph = [torch.tensor(item, dtype=torch.long, device=device) for item in mel2ph]
    ph_list = [torch.tensor(item, dtype=torch.long, device=device) for item in ph_list]
    mel2ph = torch.nn.utils.rnn.pad_sequence(mel2ph, batch_first=True, padding_value=padding_ph_idx)
    ph_list = torch.nn.utils.rnn.pad_sequence(ph_list, batch_first=True, padding_value=padding_ph_idx)
    ph_tensor = torch.gather(ph_list, dim=1, index=mel2ph)
    return ph_tensor


@dataclass
class PhonemeDecoderOutput(ModelOutput):
    last_hideen_state: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    labels: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None


class PhonemeDecoderPreTrainedModel(PreTrainedModel):
    config_class = PhonemeDecoderConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[PhonemeDecoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = PhonemeDecoderConfig(**config)
        super().__init__(config=config)

    @staticmethod
    def build_model(
        config: Optional[Union[PhonemeDecoderConfig, Dict]] = None,
        **kwargs
    ) -> "PhonemeDecoderPreTrainedModel":
        if config is not None:
            if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
                config = PhonemeDecoderConfig(**config)
            if config.decoder_type == "linear":
                return PhonemeLinear(config=config)
            else:
                raise NotImplementedError
        raise NotImplementedError


class PhonemeLinear(PhonemeDecoderPreTrainedModel):
    def __init__(
        self,
        config: Union[PhonemeDecoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = PhonemeDecoderConfig(**config)
        super().__init__(config=config)
        linear_dict = OrderedDict()
        linear_dict["project_0"] = nn.Linear(self.config.linear_dim_list[0], self.config.linear_dim_list[1])
        for i in range(1, len(self.config.linear_dim_list) - 1):
            linear_dict[f"dropout_{i}"] = nn.Dropout(self.config.dropout)
            linear_dict[f"project_{i}"] = nn.Linear(self.config.linear_dim_list[i], self.config.linear_dim_list[i + 1])
        self.linear = nn.Sequential(linear_dict)
        # self.linear = nn.Sequential(
        #     OrderedDict([
        #                 (f"project_{i}", nn.Linear(self.config.linear_dim_list[i], self.config.linear_dim_list[i+1])) \
        #                     for i in range(len(self.config.linear_dim_list) - 1)
        #             ]
        #     )
        # )

    def forward(
        self,
        phoneme_embeddings: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        logits = self.linear(phoneme_embeddings)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.num_phoneme), labels.view(-1), ignore_index=-100)
        return PhonemeDecoderOutput(
            logits=logits,
            labels=labels,
            loss=loss,
        )
