import json
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass

from funcodec.transformers.llama.configuration_llama import LlamaConfig
from funcodec.transformers.llama.modeling_llama import LlamaForCausalLM


@dataclass
class BackboneStrategy:
    backbone: Optional[str] = None
    # FunCodec, transformers
    config_path: Optional[str] = None
    config: Optional[PretrainedConfig] = None

    def __post_init__(self):
        if self.config_path is not None:
            # config = AutoConfig.from_pretrained(self.config_path)
            with open(self.config_path, mode="r+") as f:
                config_dict = json.load(f)
            self.config = LlamaConfig(**config_dict)
    
    def build_model(self):
        return LlamaForCausalLM(config=self.config)