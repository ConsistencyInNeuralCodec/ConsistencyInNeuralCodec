import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from funcodec.models.consistency.perturb_encoder import(
    PhaseAugConfig,
    PerturbEncoderConfig,
)
from funcodec.models.consistency.slice_encoder import SliceEncoderConfig

logger = logging.get_logger(__name__)


class ConsistencyStrategy(PretrainedConfig):
    def __init__(
        self,
        perturb_encoder_config: Optional[Union[dict, PerturbEncoderConfig]] = None,
        slice_encoder_config: Optional[Union[dict, SliceEncoderConfig]] = None,
        **kwargs
    ):
        if perturb_encoder_config is not None and isinstance(perturb_encoder_config, dict):
            perturb_encoder_config = PerturbEncoderConfig(**perturb_encoder_config)
        if slice_encoder_config is not None and isinstance(slice_encoder_config, dict):
            slice_encoder_config = SliceEncoderConfig(**slice_encoder_config)
        super().__init__(
            perturb_encoder_config=perturb_encoder_config,
            slice_encoder_config=slice_encoder_config,
        )