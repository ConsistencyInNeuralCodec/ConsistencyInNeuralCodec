import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class PhonemeDecoderConfig(PretrainedConfig):

    model_type = "phoneme-decoder"
    is_composition = True

    def __init__(
        self,
        decoder_type: Optional[str] = None,
        loss_weight: Optional[float] = 1.0,
        # for phoneme linear
        linear_dim_list: Optional[Sequence] = None,
        dropout: Optional[float] = 0.1,
        phoneme_embedding_dim: Optional[int] = None,
        num_phoneme: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            decoder_type=decoder_type,
            loss_weight=loss_weight,
            # for phoneme linear
            linear_dim_list=linear_dim_list,
            dropout=dropout,
            phoneme_embedding_dim=phoneme_embedding_dim,
            num_phoneme=num_phoneme,
            **kwargs
        )
