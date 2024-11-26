import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class SpeakerContrastiveEncoderConfig(PretrainedConfig):

    model_type = "speaker-contrastive-encoder"
    is_composition = True

    def __init__(
        self,
        encoder_type: Optional[str] = None,
        encoder_input_type: Optional[str] = "quant_in",
        linear_dim_list: Optional[Sequence] = None,
        dropout: Optional[float] = 0.1,
        loss_type: Optional[str] = "info_nce_loss",
        loss_weight: Optional[Union[float, Sequence[float]]] = 1.0,
        info_nce_loss_reduction: Optional[str] = "mean",
        temperature: Optional[float] = 0.1,
        **kwargs
    ):
        """
        params:
            encoder_type:
                half_speech_encoder
            encoder_input_type:
                quant_in
                quant_out
            loss_type: 
                cosine_similarity:
                contrastive_loss, info_nce_loss:
                cosine_similarity+info_nce_loss: 
            loss_weight:
                1.0
                [1.0, 1.0]
        """
        if isinstance(loss_weight, float):
            loss_weight = [loss_weight]
        super().__init__(
            encoder_type=encoder_type,
            encoder_input_type=encoder_input_type,
            linear_dim_list=linear_dim_list,
            dropout=dropout,
            loss_type=loss_type,
            loss_weight=loss_weight,
            info_nce_loss_reduction=info_nce_loss_reduction,
            temperature=temperature,
            **kwargs
        )
