import sys, importlib
import os, time, random, math
import logging, warnings
import omegaconf
from dataclasses import dataclass
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
from .configuration_retrain_model import RetrainModelConfig


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
    module.requires_grad = False
    return module


def unfreeze_module(module):
    module.requires_grad = True
    for param in module.parameters():
        param.requires_grad = True
    module.requires_grad = True
    return module


class RetrainModelPreTrainedModel(PreTrainedModel):
    config_class = RetrainModelConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[RetrainModelConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = RetrainModelConfig(**config)
        super().__init__(config=config)

    def prepare(self, model):
        for module_name in self.config.freeze_modules:
            module = getattr(model, module_name, None)
            if module is not None:
                freeze_module(module)
        for module_name in self.config.retrain_modules:
            module = getattr(model, module_name, None)
            if module is not None:
                unfreeze_module(module)
        return model
