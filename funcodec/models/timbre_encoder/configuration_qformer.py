import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class QFormerConfig(PretrainedConfig):

    model_type = "qformer"
    is_composition = True

    def __init__(
        self,
        # for transformer decoder layer
        d_model: Optional[int] = 768,
        embed_dim: Optional[int] = 768,
        kdim: Optional[int] = 768,
        vdim: Optional[int] = 768,
        nhead: Optional[int] = 12,
        dim_feedforward: Optional[int] = 3072,
        dropout: Optional[float] = 0.1,
        activation: Optional[str] = "gelu",
        layer_norm_eps: Optional[float] = 1e-5,
        batch_first: Optional[bool] = False,
        norm_first: Optional[bool] = False,
        self_attention_first: Optional[bool] = False,
        ffn_after_cross_attention: Optional[bool] = False,
        # for q former
        num_queries: Optional[int] = 32,
        num_query_layers: Optional[int] = 4,
        num_multimodal_layers: Optional[int] = 0,
        norm: Optional[bool] = False,
        **kwargs
    ):
        super().__init__(
            d_model=d_model,
            embed_dim=embed_dim,
            kdim=kdim,
            vdim=vdim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            self_attention_first=self_attention_first,
            ffn_after_cross_attention=ffn_after_cross_attention,
            num_queries=num_queries,
            num_query_layers=num_query_layers,
            num_multimodal_layers=num_multimodal_layers,
            norm=norm,
            **kwargs
        )