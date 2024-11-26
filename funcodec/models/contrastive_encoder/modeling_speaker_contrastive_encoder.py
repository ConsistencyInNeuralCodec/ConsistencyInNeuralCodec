import sys, importlib
import os, time, random, math
from collections import OrderedDict
import logging, warnings
import omegaconf
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .configuration_speaker_contrastive_encoder import SpeakerContrastiveEncoderConfig
from ..utils import lengths_to_padding_mask, lengths_to_attention_mask

logger = logging.getLogger(__name__)


def mean_pooling(
    x: torch.Tensor,
    lengths: torch.LongTensor,
    attention_mask: Optional[torch.BoolTensor] = None
):
    """
    x: [bsz, length, dim]
    attention_mask: [bsz, length]
        0: mask
        1: not mask
    """
    # return x[:, 0, :]
    if attention_mask is None:
        attention_mask = lengths_to_attention_mask(lengths)
    attention_mask = attention_mask.unsqueeze(-1) # [bsz, length, 1]
    x = x * attention_mask
    # mean_x = x.sum(1) / lengths.clamp(min=1).unsqueeze(-1)
    mean_x = x.sum(1) / lengths.unsqueeze(-1)
    return mean_x


def mean_pooling_2(
    x: torch.Tensor,
    lengths: torch.LongTensor,
    attention_mask: Optional[torch.BoolTensor] = None
):
    """
    x: [bsz, length, dim]
    attention_mask: [bsz, length]
        0: mask
        1: not mask
    """
    if attention_mask is None:
        attention_mask = lengths_to_attention_mask(lengths)
    for i in range(len(x.shape) - 2):
        attention_mask = attention_mask.unsqueeze(-1) # [bsz, length, 1]
    x = x * attention_mask
    for i in range(len(x.shape) - 2):
        x = x.squeeze(-1)
    # print(666, x.shape, lengths.shape)
    mean_x = x.sum(1) / lengths.clamp(min=1).unsqueeze(-1)
    return mean_x


class SpeakerContrastiveEncoder(PreTrainedModel):
    config_class = SpeakerContrastiveEncoderConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[SpeakerContrastiveEncoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = SpeakerContrastiveEncoderConfig(**config)
        super().__init__(config=config)
        if self.config.linear_dim_list is not None:
            linear_dict = OrderedDict()
            linear_dict["project_0"] = nn.Linear(self.config.linear_dim_list[0], self.config.linear_dim_list[1])
            for i in range(1, len(self.config.linear_dim_list) - 1):
                linear_dict[f"dropout_{i}"] = nn.Dropout(self.config.dropout)
                linear_dict[f"project_{i}"] = nn.Linear(self.config.linear_dim_list[i], self.config.linear_dim_list[i + 1])
            self.linear = nn.Sequential(linear_dict)
            # self.linear = nn.Sequential(
            #     OrderedDict([
            #                 (f"project_{i}", nn.Linear(self.config.linear_dim_list[i], self.config.linear_dim_list[i+1])) \
            #                     for i in range(len(self.config.linear_dim_list) - 1)
            #             ]
            #     )
            # )
        else:
            self.linear  = nn.Identity()

    @staticmethod
    def build_model(
        config: Optional[Union[SpeakerContrastiveEncoderConfig, Dict]] = None,
        **kwargs
    ) -> "SpeakerContrastiveEncoderPreTrainedModel":
        if config is not None:
            if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
                config = SpeakerContrastiveEncoderConfig(**config)
            if config.encoder_type == "half_speech_encoder":
                return SpeakerContrastiveEncoderForHalfSpeech(config=config)
            else:
                raise NotImplementedError
        raise NotImplementedError


@dataclass
class SpeechBatch(ModelOutput):
    speech: Optional[torch.FloatTensor] = None
    speech_lengths: Optional[torch.LongTensor] = None
    feature_lengths: Optional[torch.LongTensor] = None
    _attention_mask: Optional[torch.BoolTensor] = None
    _padding_mask: Optional[torch.BoolTensor] = None

    @property
    def attention_mask(self):
        if self._attention_mask is None:
            self._attention_mask = lengths_to_attention_mask(self.speech_lengths)
        return self._attention_mask

    @property
    def padding_mask(self):
        if self._padding_mask is None:
            self._padding_mask = lengths_to_padding_mask(self.speech_lengths)
        return self._padding_mask


@dataclass
class HalfSpeechBatch(ModelOutput):
    original: Optional[torch.FloatTensor] = None
    first_half: Optional[torch.FloatTensor] = None
    second_half: Optional[torch.FloatTensor] = None
    speaker_ids: Optional[torch.LongTensor] = None


def get_emebddings_split_from_end_positions(
    embeddings: torch.FloatTensor,
    end_positions: torch.LongTensor,
):
    shape = embeddings.shape
    batch_size = shape[0]
    max_length = end_positions.max()
    embeddings = embeddings[:, :max_length]
    range_arr = torch.arange(max_length, device=embeddings.device).unsqueeze(0).expand(batch_size, -1)
    end_positions = end_positions.unsqueeze(1).expand_as(range_arr)
    embeddings[range_arr >= end_positions] = 0
    return embeddings


def get_emebddings_split_from_start_positions(
    embeddings: torch.FloatTensor,
    start_positions: torch.LongTensor,
    valid_length: int,
):
    shape = embeddings.shape
    batch_size = shape[0]
    # 创建用于采集的索引
    batch_indices = torch.arange(batch_size, device=embeddings.device)[:, None]  # 大小为 [batch_size, 1]
    column_indices = torch.arange(valid_length, device=embeddings.device)[None, :] + start_positions[:, None]  # 大小为 [batch_size, valid_length]
    splitted_embeddings = embeddings[batch_indices, column_indices]
    # splitted_embeddings = torch.gather(embeddings, dim=1, index=column_indices)
    return splitted_embeddings


@dataclass
class SpeakerContrastiveEncoderOutput(ModelOutput):
    contrastive_speech: Optional[torch.FloatTensor] = None
    all_features: Optional[torch.FloatTensor] = None # target, positive, negative
    _target_features: Optional[torch.FloatTensor] = None
    target_features_attention_mask: Optional[torch.FloatTensor] = None
    _positive_features: Optional[torch.FloatTensor] = None
    positive_features_attention_mask: Optional[torch.FloatTensor] = None
    _negative_features: Optional[torch.FloatTensor] = None
    negative_features_attention_mask: Optional[torch.FloatTensor] = None

    cosine_logits: Optional[torch.FloatTensor] = None
    logits_attention_mask: Optional[torch.FloatTensor] = None
    positive_logits_attention_mask: Optional[torch.FloatTensor] = None
    _positive_logits: Optional[torch.FloatTensor] = None
    negative_logits_attention_mask: Optional[torch.FloatTensor] = None
    _negative_logits: Optional[torch.FloatTensor] = None

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[torch.FloatTensor] = None

    @property
    def target_features(self):
        if self._target_features is None:
            self._target_features = self.all_features[self.target_features_attention_mask]
        return self._target_features

    @property
    def positive_features(self):
        if self._positive_features is None:
            self._positive_features = self.all_features[self.positive_features_attention_mask]
        return self._positive_features

    @property
    def negative_features(self):
        if self._negative_features is None:
            self._negative_features = self.all_features[self.negative_features_attention_mask]
        return self._negative_features

    @property
    def positive_logits(self):
        if self._positive_logits is None:
            self._positive_logits = self.cosine_logits[self.positive_logits_attention_mask]
        return self._positive_logits

    @property
    def negative_logits(self):
        if self._negative_logits is None:
            self._negative_logits = self.cosine_logits[self.negative_logits_attention_mask]
        return self._negative_logits


class SpeakerContrastiveEncoderForHalfSpeech(SpeakerContrastiveEncoder):
    @staticmethod
    def split_speech_batch(
        speech: torch.FloatTensor,
        speech_lengths: Optional[torch.LongTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        get_feature_lengths_from_speech_lengths=None,
        speaker_ids: Optional[torch.LongTensor] = None,
    ) -> HalfSpeechBatch:
        """
        speech: [batch_size, length]
        """
        if speech_lengths is None:
            speech_lengths = torch.tensor([speech.shape[1]] * speech.shape[0], dtype=torch.long, device=speech.device)
        if speaker_ids is None:
            speaker_ids = torch.arange(start=0, end=speech.shape[0], dtype=torch.long, device=speech.device)
        # if feature_lengths is not None:
        #     indices = feature_lengths >= 2
        #     speech = speech[indices]
        #     speech_lengths = speech_lengths[indices]
        #     feature_lengths = feature_lengths[indices]
        #     speaker_ids = speaker_ids[indices]
        batch_size, _ = speech.shape[:2]
        max_length = speech_lengths.max()
        half_speech_lengths = speech_lengths // 2
        original_speech_batch = SpeechBatch(speech=speech, speech_lengths=speech_lengths, feature_lengths=feature_lengths)

        first_half_speech = get_emebddings_split_from_end_positions(speech.clone(), half_speech_lengths)
        first_half = SpeechBatch(
            speech=first_half_speech,
            speech_lengths=half_speech_lengths,
            feature_lengths=get_feature_lengths_from_speech_lengths(half_speech_lengths),
        )
        second_half_speech = get_emebddings_split_from_start_positions(speech.clone(), half_speech_lengths, (max_length - max_length // 2).item())
        second_half = SpeechBatch(
            speech=second_half_speech,
            speech_lengths=speech_lengths - half_speech_lengths,
            feature_lengths=get_feature_lengths_from_speech_lengths(speech_lengths - half_speech_lengths),
        )
        return HalfSpeechBatch(
            original=original_speech_batch,
            first_half=first_half,
            # first_half=original_speech_batch,
            second_half=second_half,
            speaker_ids=speaker_ids,
        )

    def forward_0(
        self,
        forward_timbre_encoder,
        speech: torch.FloatTensor,
        speech_lengths: Optional[torch.LongTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        get_feature_lengths_from_speech_lengths=None,
        speaker_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        contrastive_speech = self.split_speech_batch(
            speech=speech,
            speech_lengths=speech_lengths,
            feature_lengths=feature_lengths,
            get_feature_lengths_from_speech_lengths=get_feature_lengths_from_speech_lengths,
            speaker_ids=speaker_ids,
        )
        first_half_speech_features = forward_timbre_encoder(contrastive_speech.first_half) # batch_size, length, dim
        first_half_speech_features = mean_pooling(
            first_half_speech_features,
            contrastive_speech.first_half.feature_lengths
        ) # batch_size, dim
        batch_size = first_half_speech_features.shape[0]
        device = first_half_speech_features.device
        first_half_speech_features = self.linear(first_half_speech_features)
        second_half_speech_features = forward_timbre_encoder(contrastive_speech.second_half)
        second_half_speech_features = mean_pooling(
            second_half_speech_features,
            contrastive_speech.second_half.feature_lengths
        ) # batch_size, dim
        second_half_speech_features = self.linear(second_half_speech_features)
        # target与positive相隔batch_size的距离, 其余均为negative
        target_features = first_half_speech_features # batch_size, dim
        positive_features = second_half_speech_features
        all_features = torch.cat([first_half_speech_features, second_half_speech_features], dim=0) # 2xbatch_size, dim

        positive_attention_mask = torch.zeros(batch_size, 2 * batch_size, dtype=torch.bool, device=device)
        indices = torch.arange(batch_size, device=device)
        positive_attention_mask[indices, indices + batch_size] = True

        # [batch_size, dim] * [2xbatch_size, dim] -> [batch_size, 2xbatch_size]
        # target at [0, batch_size), positive at [batch_size, 2xbatch_size)
        # for i-th sample, negative at [..., i, ..., i+batch_size, ...]
        negative_attention_mask = torch.ones(batch_size, 2 * batch_size, dtype=torch.bool, device=device)
        negative_attention_mask[indices, indices] = False
        negative_attention_mask[indices, indices + batch_size] = False
        negative_features_attention_mask = negative_attention_mask.clone()
        cosine_logits = torch.cosine_similarity(target_features[:, None, :], all_features[None, :, :], dim=-1) # batch_size, 2xbatch_size
        positive_logits = cosine_logits[positive_attention_mask] # batch_size, without attention_mask
        # negative_logits = cosine_logits.view(batch_size, -1) # batch_size, 2xbatch_size , with attention_mask

        speaker_id_attention_mask = contrastive_speech.speaker_ids.unsqueeze(1).expand(-1, batch_size)
        speaker_id_attention_mask = (speaker_id_attention_mask != speaker_id_attention_mask.t()).bool()
        speaker_id_attention_mask = torch.cat([speaker_id_attention_mask, speaker_id_attention_mask], dim=-1) # batch_size, 2xbatch_size
        negative_attention_mask = torch.logical_and(negative_attention_mask, speaker_id_attention_mask)
        logits_attention_mask = negative_attention_mask.clone()
        logits_attention_mask[indices, indices + batch_size] = True
        negative_logits = cosine_logits.masked_fill(~negative_attention_mask, -100)

        loss_types = self.config.loss_type.split("+")
        tot_loss = 0.0
        loss_dict = {}
        for i, loss_type in enumerate(loss_types):
            if loss_type == "info_nce_loss":
                labels = indices + batch_size
                _cosine_logits = cosine_logits / self.config.temperature
                _cosine_logits = _cosine_logits.masked_fill(~logits_attention_mask, -100)
                loss = F.cross_entropy(_cosine_logits, labels)
                # print(666, loss, _cosine_logits)
                loss_dict[loss_type] = loss
                tot_loss += self.config.loss_weight[i] * loss

        return SpeakerContrastiveEncoderOutput(
            contrastive_speech=contrastive_speech,
            all_features=all_features,
            _target_features=target_features,
            _positive_features=positive_features,
            negative_features_attention_mask=negative_features_attention_mask,
            cosine_logits=cosine_logits,
            logits_attention_mask=logits_attention_mask,
            positive_logits_attention_mask=positive_attention_mask,
            _positive_logits=positive_logits,
            negative_logits_attention_mask=negative_attention_mask,
            _negative_logits=negative_logits,
            loss=tot_loss,
            loss_dict=loss_dict,
        )


    def forward_1(
        self,
        forward_timbre_encoder,
        speech: torch.FloatTensor,
        speech_lengths: Optional[torch.LongTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        get_feature_lengths_from_speech_lengths=None,
        speaker_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        1xbatch_size
        """
        contrastive_speech = self.split_speech_batch(
            speech=speech,
            speech_lengths=speech_lengths,
            feature_lengths=feature_lengths,
            get_feature_lengths_from_speech_lengths=get_feature_lengths_from_speech_lengths,
            speaker_ids=speaker_ids,
        )
        # print(666, contrastive_speech.speaker_ids)
        first_half_speech_features = forward_timbre_encoder(contrastive_speech.first_half) # batch_size, length, dim
        first_half_speech_features = mean_pooling(
            first_half_speech_features,
            contrastive_speech.first_half.feature_lengths
        ) # batch_size, dim
        batch_size = first_half_speech_features.shape[0]
        device = first_half_speech_features.device
        indices = torch.arange(batch_size, device=device)
        first_half_speech_features = self.linear(first_half_speech_features)
        second_half_speech_features = forward_timbre_encoder(contrastive_speech.second_half)
        # second_half_speech_features = forward_timbre_encoder(contrastive_speech.second_half).detach()
        second_half_speech_features = mean_pooling(
            second_half_speech_features,
            contrastive_speech.second_half.feature_lengths
        ) # batch_size, dim
        second_half_speech_features = self.linear(second_half_speech_features)
        target_features = first_half_speech_features # batch_size, dim
        positive_features = second_half_speech_features # batch_size, dim
        negative_features = first_half_speech_features[None].expand(batch_size, -1, -1) # batch_size, batch_size, dim
        all_features = torch.cat([positive_features[:, None, :], negative_features], dim=1) # batch_size, 1+batch_size, dim
        cosine_logits = torch.cosine_similarity(target_features[:, None, :], all_features, dim=-1) # batch_size, 1+batch_size

        positive_attention_mask = torch.zeros(batch_size, 1 + batch_size, dtype=torch.bool, device=device)
        positive_attention_mask[:, 0] = True
        positive_logits = cosine_logits[positive_attention_mask]

        negative_attention_mask = torch.ones(batch_size, 1 + batch_size, dtype=torch.bool, device=device)
        negative_attention_mask[:, 0] = False
        # negative_attention_mask[:, 1:][indices, indices] = False
        negative_attention_mask[indices, 1 + indices] = False

        speaker_id_attention_mask = contrastive_speech.speaker_ids.unsqueeze(1).expand(-1, batch_size)
        speaker_id_attention_mask = (speaker_id_attention_mask != speaker_id_attention_mask.t()).bool() # batch_size, batch_size
        negative_attention_mask[:, 1:] = torch.logical_and(negative_attention_mask[:, 1:], speaker_id_attention_mask)
        logits_attention_mask = negative_attention_mask.clone()
        logits_attention_mask[:, 0] = True
        negative_logits = cosine_logits.masked_fill(~negative_attention_mask, -100)

        loss_types = self.config.loss_type.split("+")
        tot_loss = 0.0
        loss_dict = {}
        for i, loss_type in enumerate(loss_types):
            if loss_type == "info_nce_loss":
                labels = torch.zeros(batch_size, device=device, dtype=torch.long)
                _cosine_logits = cosine_logits / self.config.temperature
                _cosine_logits = _cosine_logits.masked_fill(~logits_attention_mask, -100)
                loss = F.cross_entropy(_cosine_logits, labels, ignore_index=-100)
                # print(loss, contrastive_speech.first_half.feature_lengths, contrastive_speech.second_half.feature_lengths)
                # print(_cosine_logits)
                loss_dict[loss_type] = loss
                tot_loss += self.config.loss_weight[i] * loss

        return SpeakerContrastiveEncoderOutput(
            contrastive_speech=contrastive_speech,
            all_features=all_features,
            _target_features=target_features,
            _positive_features=positive_features,
            _negative_features=negative_features,
            negative_features_attention_mask=negative_attention_mask,
            cosine_logits=cosine_logits,
            logits_attention_mask=logits_attention_mask,
            positive_logits_attention_mask=positive_attention_mask,
            _positive_logits=positive_logits,
            negative_logits_attention_mask=negative_attention_mask,
            _negative_logits=negative_logits,
            loss=tot_loss,
            loss_dict=loss_dict,
        )

    def forward(
        self,
        forward_timbre_encoder,
        speech: torch.FloatTensor,
        speech_lengths: Optional[torch.LongTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        get_feature_lengths_from_speech_lengths=None,
        speaker_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        2 x batch_size
        """
        contrastive_speech = self.split_speech_batch(
            speech=speech,
            speech_lengths=speech_lengths,
            feature_lengths=feature_lengths,
            get_feature_lengths_from_speech_lengths=get_feature_lengths_from_speech_lengths,
            speaker_ids=speaker_ids,
        )
        # print(666, contrastive_speech.speaker_ids)
        first_half_speech_features = forward_timbre_encoder(contrastive_speech.first_half) # batch_size, length, dim
        first_half_speech_features = mean_pooling(
            first_half_speech_features,
            contrastive_speech.first_half.feature_lengths
        ) # batch_size, dim
        batch_size = first_half_speech_features.shape[0]
        device = first_half_speech_features.device
        indices = torch.arange(2 * batch_size, device=device)
        first_half_speech_features = self.linear(first_half_speech_features)
        second_half_speech_features = forward_timbre_encoder(contrastive_speech.second_half)
        # second_half_speech_features = forward_timbre_encoder(contrastive_speech.second_half).detach()
        second_half_speech_features = mean_pooling(
            second_half_speech_features,
            contrastive_speech.second_half.feature_lengths
        ) # batch_size, dim
        second_half_speech_features = self.linear(second_half_speech_features)
        target_features = torch.cat([first_half_speech_features, second_half_speech_features], dim=0) # 2xbatch_size, dim
        positive_features = torch.cat([second_half_speech_features, first_half_speech_features], dim=0) # 2xbatch_size, dim
        negative_features = target_features[None].expand(2 * batch_size, -1, -1) # 2xbatch_size, 2xbatch_size, dim
        all_features = torch.cat([positive_features[:, None, :], negative_features], dim=1) # 2xbatch_size, 1+2xbatch_size, dim
        cosine_logits = torch.cosine_similarity(target_features[:, None, :], all_features, dim=-1) # 2xbatch_size, 1+2xbatch_size

        positive_attention_mask = torch.zeros(2 * batch_size, 1 + 2 * batch_size, dtype=torch.bool, device=device)
        positive_attention_mask[:, 0] = True
        positive_logits = cosine_logits[positive_attention_mask]

        negative_attention_mask = torch.ones(2 * batch_size, 1 + 2 * batch_size, dtype=torch.bool, device=device)
        negative_attention_mask[:, 0] = False
        # negative_attention_mask[:, 1:][indices, indices] = False
        negative_attention_mask[indices, 1 + indices] = False

        speaker_ids = torch.cat([contrastive_speech.speaker_ids, contrastive_speech.speaker_ids], dim=0)
        speaker_id_attention_mask = speaker_ids.unsqueeze(1).expand(-1, 2 * batch_size)
        speaker_id_attention_mask = (speaker_id_attention_mask != speaker_id_attention_mask.t()).bool() # 2 * batch_size, 2 * batch_size
        negative_attention_mask[:, 1:] = torch.logical_and(negative_attention_mask[:, 1:], speaker_id_attention_mask)
        logits_attention_mask = negative_attention_mask.clone()
        logits_attention_mask[:, 0] = True
        negative_logits = cosine_logits.masked_fill(~negative_attention_mask, -100)

        loss_types = self.config.loss_type.split("+")
        tot_loss = 0.0
        loss_dict = {}
        for i, loss_type in enumerate(loss_types):
            if loss_type == "info_nce_loss":
                labels = torch.zeros(2 * batch_size, device=device, dtype=torch.long)
                _cosine_logits = cosine_logits / self.config.temperature
                _cosine_logits = _cosine_logits.masked_fill(~logits_attention_mask, -100)
                loss = F.cross_entropy(_cosine_logits, labels, ignore_index=-100, reduction=self.config.info_nce_loss_reduction)
                # print(loss, contrastive_speech.first_half.feature_lengths, contrastive_speech.second_half.feature_lengths)
                # print(_cosine_logits)
                loss_dict[loss_type] = loss
                tot_loss += self.config.loss_weight[i] * loss

        return SpeakerContrastiveEncoderOutput(
            contrastive_speech=contrastive_speech,
            all_features=all_features,
            _target_features=target_features,
            _positive_features=positive_features,
            _negative_features=negative_features,
            negative_features_attention_mask=negative_attention_mask,
            cosine_logits=cosine_logits,
            logits_attention_mask=logits_attention_mask,
            positive_logits_attention_mask=positive_attention_mask,
            _positive_logits=positive_logits,
            negative_logits_attention_mask=negative_attention_mask,
            _negative_logits=negative_logits,
            loss=tot_loss,
            loss_dict=loss_dict,
        )