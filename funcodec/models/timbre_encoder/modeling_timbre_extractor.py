import sys, importlib
import os, time, random
import logging, warnings
import omegaconf
from dataclasses import dataclass
import torch
from torch import nn
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from transformers.models.wav2vec2_conformer.configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerEncoder,
    Wav2Vec2ConformerModel,
)

from speechbrain.inference.classifiers import EncoderClassifier as SpeechbrainEncoderClassifier
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .configuration_timbre_extractor import TimbreExtractorConfig
from .modeling_qformer import QFormerModel
from ..utils import lengths_to_padding_mask, lengths_to_attention_mask

from funcodec.models.contrastive_encoder.configuration_speaker_contrastive_encoder import SpeakerContrastiveEncoderConfig
from funcodec.models.contrastive_encoder.modeling_speaker_contrastive_encoder import SpeakerContrastiveEncoder

logger = logging.getLogger(__name__)


def mean_pooling(x: torch.Tensor, attention_mask: torch.Tensor):
    """
    x: [bsz, length, dim]
    attention_mask: [bsz, length]
        0: mask
        1: not mask
    """
    attention_mask = attention_mask.unsqueeze(-1) # [bsz, length, 1]
    x = x * attention_mask
    mean_x = x.sum(1) / attention_mask.sum(1).clamp(min=1)
    return mean_x


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


@dataclass
class SpeechQuantOutContrastiveOutput(ModelOutput):
    speaker_embeddings: Optional[torch.FloatTensor] = None
    quant_out: Optional[torch.FloatTensor] = None
    speaker_similarity: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None


@dataclass
class TimbreExtractorOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.FloatTensor] = None
    timbre_output: Optional[torch.FloatTensor] = None
    merged_output: Optional[torch.FloatTensor] = None
    orig_features: Optional[torch.FloatTensor] = None
    half_speech_contrastive_output: Optional[torch.FloatTensor] = None
    speech_quant_out_contrastive_output: Optional[torch.FloatTensor] = None

    def exchange_timbre(self):
        assert self.last_hidden_state.shape[0] == 2
        self.last_hidden_state[[0, 1]] = self.last_hidden_state[[1, 0]]
        if self.merged_output is not None:
            self.merged_output[[0, 1]] = self.merged_output[[1, 0]]
        if self.attention_mask is not None:
            self.attention_mask[[0, 1]] = self.attention_mask.clone()[[1, 0]]
        if self.orig_features is not None:
            self.orig_features[[0, 1]] = self.orig_features.clone()[[1, 0]]


class BaseTimbreExtractorModel(PreTrainedModel):
    config_class = TimbreExtractorConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[TimbreExtractorConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = TimbreExtractorConfig(**config)
        super().__init__(config=config)
        if config.merge_embed == "qformer":
            self.qformer = QFormerModel(config=config.qformer_config)
        if config.half_speech_contrastive:
            half_speakerContrastive_encoder_config = SpeakerContrastiveEncoderConfig(
                encoder_type="half_speech_encoder",
                loss_type="info_nce_loss",
            )
            self.half_speech_contrastive_encoder = SpeakerContrastiveEncoder.build_model(
                config=half_speakerContrastive_encoder_config,
            )
            self.forward_timbre_extractor = lambda half_speech_features: self.forward(half_speech_features.speech, lengths_to_attention_mask(half_speech_features.speech_lengths), half_speech_contrastive=False, speech_quant_out_contrastive=False).last_hidden_state
            self.get_feature_lengths_from_speech_lengths = lambda x: x
        if config.speech_quant_out_contrastive:
            pass

    @staticmethod
    def build_model(
        config: Optional[Union[TimbreExtractorConfig, Dict]] = None,
        **kwargs
    ) -> "BaseTimbreExtractorModel":
        if config is not None:
            if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
                config = TimbreExtractorConfig(**config)
            if config.encoder_type == "wav2vec2_conformer":
                return TimbreExtractorWithConformer(config=config)
            else:
                raise NotImplementedError
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        pass

    def forward_half_speech_contrastive_encoder(
        self,
        speech_features: Optional[torch.FloatTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        # print(f"666 forward_half_speech_contrastive_encoder")
        return self.half_speech_contrastive_encoder(
            forward_timbre_encoder=self.forward_timbre_extractor,
            speech=speech_features,
            speech_lengths=feature_lengths,
            feature_lengths=feature_lengths,
            get_feature_lengths_from_speech_lengths=self.get_feature_lengths_from_speech_lengths,
            speaker_ids=speaker_ids,
        )

    def forward_speech_quant_out_contrastive(
        self,
        speaker_embeddings,
        quant_out: torch.FloatTensor,
        attention_mask: torch.BoolTensor,
        **kwargs
    ):
        """
        speaker_embeddings: [batch_size, dim]
        """
        # print(f"666 forward_speech_quant_out_contrastive")
        speaker_embeddings = speaker_embeddings.detach()
        alpha = torch.tensor([self.config.alpha_for_speech_quant_out_contrastive_gradient_reversal], dtype=torch.float, device=quant_out.device)
        quant_out = GradientReversal.apply(quant_out, alpha)
        with torch.no_grad():
            quant_out_spk_emb = self.forward(
                hidden_states=quant_out,
                attention_mask=attention_mask,
                last_hidden_state_mean_pooling=True,
                half_speech_contrastive=False,
                speech_quant_out_contrastive=False,
            )["merged_output"]
        cosine_logits = torch.cosine_similarity(speaker_embeddings[:, None, :], quant_out_spk_emb[None, :, :], dim=-1) # [batch_size, batch_size]
        loss = 1 - torch.diag(cosine_logits)
        loss = loss.mean()
        return SpeechQuantOutContrastiveOutput(
            speaker_embeddings=speaker_embeddings,
            quant_out=quant_out_spk_emb,
            speaker_similarity=cosine_logits,
            loss=loss,
        )


class TimbreExtractorWithConformer(BaseTimbreExtractorModel):
    def __init__(
        self,
        config: Union[TimbreExtractorConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = TimbreExtractorConfig(**config)
        super().__init__(config=config)
        config = Wav2Vec2ConformerConfig.from_pretrained(config.config_path)
        self.encoder = Wav2Vec2ConformerEncoder(config)
        if self.config.merge_with_quant_out == "linear":
            codec_dim = 128
            self.quant_proj = nn.Linear(codec_dim + self.encoder.config.hidden_size, codec_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        last_hidden_state_mean_pooling: Optional[bool] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
        half_speech_contrastive: Optional[bool] = False,
        speech_quant_out_contrastive: Optional[bool] = False,
        quant_out: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ):
        timbre_output = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        timbre_extractor_output = {
            "timbre_output": timbre_output,
            "orig_features": hidden_states,
        }

        if half_speech_contrastive is None:
            half_speech_contrastive = self.config.half_speech_contrastive is not None
        if half_speech_contrastive:
            timbre_extractor_output["half_speech_contrastive_output"] = self.forward_half_speech_contrastive_encoder(
                speech_features=hidden_states,
                feature_lengths=attention_mask.sum(-1),
                speaker_ids=speaker_ids,
            )

        if self.config.merge_embed == "qformer":
            if padding_mask is None and attention_mask is not None:
                padding_mask = ~attention_mask
            merged_output = self.qformer(
                m1=timbre_output.last_hidden_state,
                m1_key_padding_mask=padding_mask,
            )
            timbre_extractor_output["merged_output"] = merged_output
            timbre_extractor_output["last_hidden_state"] = merged_output.last_hidden_state
        elif self.config.merge_embed == "mean_pooling":
            if last_hidden_state_mean_pooling:
                timbre_extractor_output["merged_output"] = mean_pooling(timbre_output.last_hidden_state, attention_mask)
            timbre_extractor_output["last_hidden_state"] = timbre_output.last_hidden_state
        else:
            raise NotImplementedError

        if speech_quant_out_contrastive is None:
            speech_quant_out_contrastive = self.config.speech_quant_out_contrastive is not None
        if speech_quant_out_contrastive:
            if timbre_extractor_output.get("merged_output", None) is None:
                timbre_extractor_output["merged_output"] = mean_pooling(timbre_output.last_hidden_state, attention_mask)
            speaker_embeddings = timbre_extractor_output["merged_output"]
            timbre_extractor_output["speech_quant_out_contrastive_output"] = self.forward_speech_quant_out_contrastive(
                speaker_embeddings,
                quant_out=quant_out,
                attention_mask=attention_mask,
            )

        return TimbreExtractorOutput(**timbre_extractor_output)
