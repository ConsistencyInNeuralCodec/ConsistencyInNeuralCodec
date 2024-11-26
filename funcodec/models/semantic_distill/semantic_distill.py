import copy
import sys, importlib
import os, time, random, math
from collections import OrderedDict
import logging, warnings
import omegaconf
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from funcodec.models.utils import lengths_to_padding_mask, lengths_to_attention_mask

from phaseaug.phaseaug import PhaseAug
import transformers
from transformers import AutoFeatureExtractor, AutoModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import AutoConfig
from transformers.utils import ModelOutput
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class SemanticDistillConfig(PretrainedConfig):
    def __init__(
        self,
        model_type: Optional[str] = None,
        model_dir: Optional[str] = None,
        model_dim: Optional[int] = None,
        codec_model_dim: Optional[int] = None,
        freeze_model: Optional[bool] = True,
        **kwargs
    ):
        """
        semantic_model_type: hubert
        model_dim: 768
        codec_model_dim: 128
        """
        super().__init__(
            model_type=model_type,
            model_dir=model_dir,
            model_dim=model_dim,
            codec_model_dim=codec_model_dim,
            freeze_model=freeze_model,
            kwargs=kwargs
        )


@dataclass
class SemanticDistillOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None


class SemanticDistillBaseModel(PreTrainedModel):
    config_class = SemanticDistillConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[dict, SemanticDistillConfig],
        **kwargs
    ):
        if isinstance(config, dict):
            config = SemanticDistillConfig(**config)
        super().__init__(config=config)

    @staticmethod
    def build_model(
        config: Union[dict, SemanticDistillConfig],
        **kwargs
    ):
        if isinstance(config, dict):
            config = SemanticDistillConfig(**config)
        if config.model_type == "hubert":
            return HuBERTSemanticDistillModel(config=config)
        else:
            raise NotImplementedError
    

class HuBERTSemanticDistillModel(SemanticDistillBaseModel):
    def __init__(
        self,
        config: Union[dict, SemanticDistillConfig],
        **kwargs
    ):
        super().__init__(config=config)
        if self.config.model_dim != self.config.codec_model_dim:
            self.output_proj = nn.Linear(self.config.model_dim, self.config.codec_model_dim)
        else:
            self.output_proj = nn.Identity()

        self.model = AutoModel.from_pretrained(self.config.model_dir)
        if self.config.freeze_model:
            self.model.eval()

        padding_emebdding = torch.tensor([0.0] * self.config.model_dim, requires_grad=False) # [768]
        self.register_buffer("padding_emebdding", padding_emebdding)
        self.padding_emebdding.requires_grad = False

    def forward(
        self,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        codec_seq_len: Optional[int] = None,
        **kwargs
    ):
        if speech.dim() == 3:
            speech = speech.squeeze(1)
        attention_mask = lengths_to_attention_mask(speech_lengths)
        inputs = transformers.feature_extraction_utils.BatchFeature(data=dict(input_values=speech, attention_mask=attention_mask))
        if self.config.freeze_model:
            with torch.no_grad():
                output = self.model(**inputs, output_hidden_states=True, return_dict=True)
        else:
            output = self.model(**inputs, output_hidden_states=True, return_dict=True)
        batch_size, seq_len, dim = output["last_hidden_state"].shape
        last_hidden_state = output["last_hidden_state"]
        if seq_len < codec_seq_len:
            # print(666, seq_len, codec_seq_len)
            # [768] -> [batch_size, n, 768]
            padding = self.padding_emebdding.unsqueeze(0).unsqueeze(1).expand(batch_size, codec_seq_len - seq_len, self.config.model_dim)
            last_hidden_state = torch.cat([last_hidden_state, padding], dim=1)
        last_hidden_state = self.output_proj(last_hidden_state)
        return SemanticDistillOutput(
            last_hidden_state=last_hidden_state,
        )
