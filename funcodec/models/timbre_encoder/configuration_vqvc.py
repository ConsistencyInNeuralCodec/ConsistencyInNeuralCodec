import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .configuration_qformer import QFormerConfig

from funcodec.modules.modeling_feature_extractor import BaseFeatureExtractor, FeatureExtractorConfig

logger = logging.get_logger(__name__)


class VQVCConfig(PretrainedConfig):

    model_type = "vqvc"
    is_composition = True

    def __init__(
        self,
        instance_norm_after_encoder: Optional[bool] = None,
        instance_norm_before_quantization: Optional[bool] = None,
        encoder_output_minus_timbre_features: Optional[bool] = None,
        feature_minus_quant: Optional[bool] = None,
        feature_extractor_before_quantization: Optional[Union[dict, FeatureExtractorConfig]] = None,
        feature_extractor_after_quantization: Optional[Union[dict, FeatureExtractorConfig]] = None,
        **kwargs
    ):
        if isinstance(feature_extractor_before_quantization, dict):
            feature_extractor_before_quantization = FeatureExtractorConfig(**feature_extractor_before_quantization)
        if isinstance(feature_extractor_after_quantization, dict):
            feature_extractor_after_quantization = FeatureExtractorConfig(**feature_extractor_after_quantization)
        super().__init__(
            instance_norm_after_encoder=instance_norm_after_encoder,
            instance_norm_before_quantization=instance_norm_before_quantization,
            encoder_output_minus_timbre_features=encoder_output_minus_timbre_features,
            feature_minus_quant=feature_minus_quant,
            feature_extractor_before_quantization=feature_extractor_before_quantization,
            feature_extractor_after_quantization=feature_extractor_after_quantization,
            **kwargs
        )
