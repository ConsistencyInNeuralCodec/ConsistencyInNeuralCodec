import torch
from torch import nn, Tensor
from torch.nn import (
    MultiheadAttention, 
    Dropout, 
    LayerNorm,
)
from torch.nn.modules.transformer import _get_activation_fn
import torch.nn.functional as F
from dataclasses import dataclass
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from pathlib import Path
import omegaconf
from dataclasses import dataclass, field
from .configuration_qformer import QFormerConfig
from typing import Any, Dict, List, Optional, Tuple, Union, Callable


class QFormerPreTrainedModel(PreTrainedModel):
    config_class = QFormerConfig
    supports_gradient_checkpointing = True


@dataclass
class QFormerOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None


class MultimodalTransformerDecoderLayer(QFormerPreTrainedModel):
    def __init__(
        self, config: QFormerConfig, **kwargs
    ):
        if isinstance(config, dict):
            config = QFormerConfig(**config)
        super().__init__(config=config)
        self.self_attn = nn.MultiheadAttention(
            config.embed_dim, config.nhead, dropout=config.dropout, 
            batch_first=config.batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            config.embed_dim, config.nhead, dropout=config.dropout, batch_first=config.batch_first,
            kdim=config.kdim, vdim=config.vdim, 
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(config.embed_dim, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.embed_dim)
        if self.config.ffn_after_cross_attention:
            self.linear3 = nn.Linear(config.embed_dim, config.dim_feedforward)
            self.dropout4 = nn.Dropout(config.dropout)
            self.linear4 = nn.Linear(config.dim_feedforward, config.embed_dim)
            self.norm4 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

        self.norm_first = config.norm_first
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        # Legacy string support for activation function.
        if isinstance(config.activation, str):
            self.activation = _get_activation_fn(config.activation)
        else:
            self.activation = config.activation

        self.self_attention_first = config.self_attention_first

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self, 
        tgt: Tensor, memory: Tensor, 
        tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            if self.self_attention_first:
                x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            else:
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                if self.config.ffn_after_cross_attention:
                    x = x + self._ff_block_after_mha_block(self.norm4(x))
                x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            if self.self_attention_first:
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            else:
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
                if self.config.ffn_after_cross_attention:
                    x = self.norm4(x + self._ff_block_after_mha_block(x))
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def _ff_block_after_mha_block(self, x: Tensor) -> Tensor:
        x = self.linear4(self.dropout(self.activation(self.linear3(x))))
        return self.dropout4(x)


class QFormerModel(QFormerPreTrainedModel):
    def __init__(
        self, 
        config: QFormerConfig,
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = QFormerConfig(**config)
        super().__init__(config=config)
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        # self.query_embedding = nn.Parameter(torch.zeros(1, config.num_queries, config.embed_dim))
        self.query_embedding = nn.Parameter(torch.randn(1, config.num_queries, config.embed_dim))
        self.norm = None
        if config.norm:
            self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

        query_transformer_layer = MultimodalTransformerDecoderLayer(config=config)
        self.query_transformer_layers = nn.modules.transformer._get_clones(query_transformer_layer, self.config.num_query_layers)

        multimodal_transformer_layer = MultimodalTransformerDecoderLayer(config=config)
        self.multimodal_transformer_layers = nn.modules.transformer._get_clones(multimodal_transformer_layer, self.config.num_multimodal_layers)

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        m1: Optional[Tensor] = None,
        m2: Optional[Tensor] = None,
        m1_key_padding_mask: Optional[Tensor] = None,
        m2_key_padding_mask: Optional[Tensor] = None, 
    ) -> QFormerOutput:
        """
        if batch_first:
            m1: [[b, t_m1, d_m1]]
            m2: [[b, t_m2, d_m2]]
            last_m1: [b, t_m1, d_m1]
            m1_mask: [t_m1, t_m1] or [num_heads_m1, t_m1, t_m1]
            m2_mask: [t_m1, t_m2] or [num_heads_m1, t_m2, t_m1]
            m1_key_padding_mask: [b, t_m1]
            m2_key_padding_mask: [b, t_m2]
        else:
            m1: [[t_m1, b, d_m1]]
            m2: [[t_m2, b, d_m2]]
            last_m1: [t_m1, b, d_m1]
            m1_mask: [t_m1, t_m1] or [num_heads_m1, t_m1, t_m1]
            m2_mask: [t_m1, t_m2] or [num_heads_m1, t_m2, t_m1]
            m1_key_padding_mask: [b, t_m1]
            m2_key_padding_mask: [b, t_m2]      
        """
        hidden_states = []
        if m1 is not None:
            output = self.query_embedding.expand(m1.shape[0], -1, -1)
        elif m2 is not None:
            output = self.query_embedding.expand(m2.shape[0], -1, -1)
        else:
            raise ValueError("all of inputs m1 and m2 are none!")
        for i in range(self.config.num_query_layers):
            output = self.query_transformer_layers[i](
                tgt=output, memory=m1,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=m1_key_padding_mask,
            )
            hidden_states.append(output)
        for i in range(self.config.num_multimodal_layers):
            output = self.multimodal_transformer_layers[i](
                tgt=output, memory=m2,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=m2_key_padding_mask,
            )
            hidden_states.append(output)
        if self.norm is not None:
            output = self.norm(output)
        return QFormerOutput(
            last_hidden_state=output,
            hidden_states=hidden_states,
        )
