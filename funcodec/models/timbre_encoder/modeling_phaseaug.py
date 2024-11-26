import sys, importlib
import os, time, random
import numpy as np
import logging, warnings
import omegaconf
from dataclasses import dataclass
import torch
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from phaseaug.phaseaug import PhaseAug

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .configuration_phaseaug import PhaseaugConfig


class PhaseAug(PreTrainedModel):
    config_class = PhaseaugConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[PhaseaugConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = PhaseaugConfig(**config)
        super().__init__(config=config)
        config_dict = config.to_dict()
        config_dict.pop("freeze", None)
        self.model = PhaseAug(**config_dict)

    def forward(self, speech: torch.Tensor, phi: Optional[torch.Tensor] = None):
        if self.config.freeze:
            with torch.no_grad():
                return self.model(speech, phi=phi).detach()
        else:
            return self.model(speech, phi=phi)