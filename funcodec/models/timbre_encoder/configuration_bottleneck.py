from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class BottleneckConfig(PretrainedConfig):
    def __init__(
        self,
        quantizer_dim: Optional[int] = None,
        quantizer_in_dim: Optional[int] = None,
        quantizer_out_dim: Optional[int] = None,
        linear_before_quantizer: Optional[bool] = None,
        linear_after_quantizer: Optional[bool] = None,
    ):
        super().__init__(
            quantizer_dim=quantizer_dim,
            quantizer_in_dim=quantizer_in_dim,
            quantizer_out_dim=quantizer_out_dim,
            linear_before_quantizer=linear_before_quantizer,
            linear_after_quantizer=linear_after_quantizer,
        )
