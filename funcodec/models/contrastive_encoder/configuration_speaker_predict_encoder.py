import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class SpeakerPredictEncoderConfig(PretrainedConfig):

    model_type = "speaker-predict-encoder"
    is_composition = True

    def __init__(
        self,
        encoder_type: Optional[str] = None,
        encoder_input_type: Optional[str] = "quant_in",
        gradient_type: Optional[str] = None,
        alpha_for_gradient_reversal: Optional[float] = 1.0,
        merge_embed: Optional[str] = "mean_pooling",
        num_speaker: Optional[int] = None,
        dropout: Optional[float] = 0.1,
        loss_weight: Optional[float] = 1.0,
        # for linear
        linear_dim_list: Optional[Sequence] = None,
        # for cnn_lstm
        indim: Optional[int] = 128,
        outdim: Optional[int] = 2456,
        head: Optional[int] = 1,
        global_pred: Optional[bool] = True,
        **kwargs
    ):
        """
        params:
            encoder_type: linear, cnn_lstm
            encoder_input_type:
                quant_in, quant_out
            gradient_type: [None, normal, stop_gradient, reverse_gradient]
            merge_embed: mean_pooling
        """
        if gradient_type is None:
            gradient_type = "normal"
        super().__init__(
            encoder_type=encoder_type,
            encoder_input_type=encoder_input_type,
            gradient_type=gradient_type,
            alpha_for_gradient_reversal=alpha_for_gradient_reversal,
            merge_embed=merge_embed,
            num_speaker=num_speaker,
            dropout=dropout,
            loss_weight=loss_weight,
            # for linear
            linear_dim_list=linear_dim_list,
            # for cnn_lstm
            indim=indim,
            outdim=outdim,
            head=head,
            global_pred=global_pred,
            **kwargs
        )