import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .configuration_qformer import QFormerConfig

logger = logging.get_logger(__name__)


class TimbreExtractorConfig(PretrainedConfig):

    model_type = "timbre-extractor"
    is_composition = True

    def __init__(
        self,
        encoder_type: Optional[str] = None,
        merge_embed: Optional[str] = "qformer",
        merge_with_decoder: Optional[str] = "normal",
        # for wav2vec2
        config_path: Optional[str] = None,
        # for q_former
        qformer_config: Optional[Union[Dict, QFormerConfig]] = None,
        # merge with quantizer output
        merge_with_quant_out: Optional[str] = "cross_attention",
        kdim: Optional[int] = 128,
        vdim: Optional[int] = 128,
        # contrastive learning
        half_speech_contrastive: Optional[bool] = None,
        half_speech_contrastive_loss: Optional[float] = None,
        speech_quant_out_contrastive: Optional[bool] = None,
        speech_quant_out_contrastive_loss: Optional[float] = None,
        alpha_for_speech_quant_out_contrastive_gradient_reversal: Optional[float] = None,
        **kwargs
    ):
        """
        encoder_type: wav2vec2_conformer
        merge_embed: qformer, mean_pooling
        merge_with_quant_out: cross_attention
        """
        if qformer_config is not None and isinstance(qformer_config, dict):
            qformer_config = QFormerConfig(**qformer_config)
        super().__init__(
            encoder_type=encoder_type,
            merge_embed=merge_embed,
            merge_with_decoder=merge_with_decoder,
            # for wav2vec2
            config_path=config_path,
            # for q_former
            qformer_config=qformer_config,
            # merge with quantizer output
            merge_with_quant_out=merge_with_quant_out,
            kdim=kdim,
            vdim=vdim,
            # contrastive learning
            half_speech_contrastive=half_speech_contrastive,
            half_speech_contrastive_loss=half_speech_contrastive_loss,
            speech_quant_out_contrastive=speech_quant_out_contrastive,
            speech_quant_out_contrastive_loss=speech_quant_out_contrastive_loss,
            alpha_for_speech_quant_out_contrastive_gradient_reversal=alpha_for_speech_quant_out_contrastive_gradient_reversal,
            **kwargs
        )
