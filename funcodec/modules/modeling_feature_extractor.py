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
import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import AutoConfig
from transformers.utils import ModelOutput
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from funcodec.modules.cnn_lstm import CNNLSTMConfig, CNNLSTM

logger = logging.get_logger(__name__)


class ProjectConfig(PretrainedConfig):
    def __init__(
        self,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
        )


class FeatureExtractorConfig(PretrainedConfig):
    model_type = "feature-extractor"
    is_composition = True

    def __init__(
        self,
        feature_extractor_type: Optional[str] = "cnn_lstm",
        feature_extractor_config: Optional[dict] = {},
        **keargs
    ):
        """
        feature_extractor_type:
            project, cnn_lstm
        """
        if feature_extractor_type == "project" and isinstance(feature_extractor_config, dict):
            feature_extractor_config = ProjectConfig(**feature_extractor_config)
        elif feature_extractor_type == "cnn_lstm" and isinstance(feature_extractor_config, dict):
            feature_extractor_config = CNNLSTMConfig(**feature_extractor_config)
        super().__init__(
            feature_extractor_type=feature_extractor_type,
            feature_extractor_config=feature_extractor_config,
        )


class BaseFeatureExtractor(PretrainedConfig):
    def __init__(
        self,
        config: Union[dict, FeatureExtractorConfig],
        **kwargs
    ):
        if isinstance(config, dict):
            config = FeatureExtractorConfig(**config)
        super().__init__(config=config)


    @staticmethod
    def build_model(
        config: Union[dict, FeatureExtractorConfig],
        **kwargs
    ):
        if isinstance(config, dict):
            config = FeatureExtractorConfig(**config)
        if config.feature_extractor_type == "project":
            raise NotImplementedError
        elif config.feature_extractor_type == "cnn_lstm":
            return CNNLSTM(**config.feature_extractor_config.to_dict())
        raise NotImplementedError
        