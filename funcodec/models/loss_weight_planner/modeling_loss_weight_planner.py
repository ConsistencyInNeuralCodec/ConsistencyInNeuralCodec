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
from .configuration_loss_weight_planner import LossWeightPlannerConfig


class BaseLossWeightPlanner(PreTrainedModel):
    config_class = LossWeightPlannerConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[LossWeightPlannerConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = LossWeightPlannerConfig(**config)
        super().__init__(config=config)
        self.weight = 0.0

    def forward(self, **kwargs):
        return self.weight

    @classmethod
    def build_model(
        cls,
        config: Optional[Union[LossWeightPlannerConfig, Dict]] = None,
        **kwargs
    ) -> "BaseLossWeightPlanner":
        if config is not None:
            if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
                config = LossWeightPlannerConfig(**config)
            if config.planner_type is None:
                return BaseLossWeightPlanner(config=config)
            elif config.planner_type == "auto_weight":
                return AutoLossWeightPlanner(config=config)
            elif config.planner_type == "step_weight":
                return StepLossWeightPlanner(config=config)
            else:
                raise NotImplementedError
        raise NotImplementedError


class AutoLossWeightPlanner(BaseLossWeightPlanner):
    def __init__(
        self,
        config: Union[LossWeightPlannerConfig, Dict],
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        self.weight = nn.Parameter(torch.ones([])) * self.config.initial_weight

    def forward(self, **kwargs):
        return self.weight


class StepLossWeightPlanner(BaseLossWeightPlanner):
    def __init__(
        self,
        config: Union[LossWeightPlannerConfig, Dict],
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        self.weight = self.config.initial_weight

    def forward(self, **kwargs):
        self.weight *= self.config.step_decay
        return self.weight