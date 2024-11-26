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

from .configuration_frame_contrastive_encoder import FrameContrastiveEncoderConfig
from .modeling_contrastive_encoder import PhonemeIntervalList
from ..utils import lengths_to_padding_mask, lengths_to_attention_mask

logger = logging.getLogger(__name__)


@dataclass
class FrameContrastiveEncoderOutput(ModelOutput):
    orig_features: Optional[torch.FloatTensor] = None
    orig_perturbed_features: Optional[torch.FloatTensor] = None
    features: Optional[torch.FloatTensor] = None
    feature_lengths: Optional[torch.FloatTensor] = None
    perturbed_features: Optional[torch.FloatTensor] = None
    phoneme_interval_list_batch: Optional[torch.FloatTensor] = None
    all_features: Optional[torch.FloatTensor] = None
    feature_lengths: Optional[torch.FloatTensor] = None
    cosine_logits: Optional[torch.FloatTensor] = None
    logits_attention_mask: Optional[torch.FloatTensor] = None
    positive_logits_attention_mask: Optional[torch.FloatTensor] = None
    negative_logits_attention_mask: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[torch.FloatTensor] = None
    code_indices: Optional[torch.FloatTensor] = None
    perturbed_code_indices: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    target: Optional[torch.FloatTensor] = None


@dataclass
class MergedPhonemeOutput(ModelOutput):
    features: Optional[torch.FloatTensor] = None
    feature_lengths: Optional[torch.FloatTensor] = None
    phoneme_interval_list_batch: Optional[torch.FloatTensor] = None


class BaseFrameContrastiveEncoder(PreTrainedModel):
    config_class = FrameContrastiveEncoderConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[FrameContrastiveEncoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = SpeakerContrastiveEncoderConfig(**config)
        super().__init__(config=config)
        self.init_feature_extractor()

    def init_feature_extractor(self):
        if self.config.feature_extractor_type == "cnn_lstm":
            from funcodec.modules.cnn_lstm import CNNLSTM, CNNLSTMConfig
            self.config.feature_extractor_config = CNNLSTMConfig(**self.config.feature_extractor_config)
            self.feature_extractor = CNNLSTM(**self.config.feature_extractor_config.to_dict())
        elif self.config.feature_extractor_type == "mlp":
            from funcodec.modules.cnn_lstm import CNNLSTM, CNNLSTMConfig
            self.config.feature_extractor_config = CNNLSTMConfig(**self.config.feature_extractor_config)
            out_dim = self.config.feature_extractor_config.outdim
            self.feature_extractor = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.GroupNorm(1, out_dim, affine=False),
                nn.ELU(inplace =True),
                nn.Linear(out_dim, out_dim),
                nn.GroupNorm(1, out_dim, affine=False),
                nn.ELU(inplace=True),
                nn.Linear(out_dim, out_dim),
                nn.GroupNorm(1, out_dim, affine=False),
            )
        elif self.config.feature_extractor_type == "linear":
            linear_dict = OrderedDict()
            linear_dict["project_0"] = nn.Linear(self.config.linear_dim_list[0], self.config.linear_dim_list[1])
            for i in range(1, len(self.config.linear_dim_list) - 1):
                linear_dict[f"dropout_{i}"] = nn.Dropout(self.config.dropout)
                linear_dict[f"project_{i}"] = nn.Linear(self.config.linear_dim_list[i], self.config.linear_dim_list[i + 1])
                # todo: activation func
            self.feature_extractor = nn.Sequential(linear_dict)
        else:
            self.feature_extractor = nn.Identity()

    @staticmethod
    def build_model(
        config: Optional[Union[FrameContrastiveEncoderConfig, Dict]] = None,
        **kwargs
    ) -> "BaseFrameContrastiveEncoder":
        if config is not None:
            if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
                config = FrameContrastiveEncoderConfig(**config)
            if config.encoder_type == "simple_frame_contrastive_encoder":
                return FrameContrastiveEncoder(config=config)
            elif config.encoder_type == "simclr_frame_contrastive_encoder":
                return SimclrFrameContrastiveEncoder(config=config)
            elif config.encoder_type == "simsiam_frame_contrastive_encoder":
                return SimsiamFrameContrastiveEncoder(config=config)
            elif config.encoder_type == "distill_frame_contrastive_encoder":
                return DistillFrameContrastiveEncoder(config=config)
            else:
                raise NotImplementedError
        raise NotImplementedError

    @staticmethod
    def merge_phoneme_features(
        features: torch.FloatTensor,
        phoneme_interval_list_batch: Optional[Sequence[Sequence[List]]] = None, # [[[s1, e1], [s2, e2]], [[s1, e1]]]
    ) -> MergedPhonemeOutput:
        batch_size, max_length, dim = features.size()
        avg_features_batch = []
        feature_lengths = []
        for i in range(batch_size):
            phoneme_intervals = phoneme_interval_list_batch[i]
            avg_features = []
            # 对每个区间求取特征的平均值
            for start, end in phoneme_intervals:
                interval_features = features[i, start:end]
                avg_feature = torch.mean(interval_features, dim=0)
                avg_features.append(avg_feature)
            avg_features = torch.stack(avg_features, dim=0)
            feature_lengths.append(avg_features.shape[0])
            avg_features_batch.append(avg_features)
        avg_features_batch = torch.nn.utils.rnn.pad_sequence(avg_features_batch, batch_first=True, padding_value=0.0)
        feature_lengths = torch.tensor(feature_lengths, dtype=torch.long, device=features.device)
        return MergedPhonemeOutput(
            features=avg_features_batch,
            feature_lengths=feature_lengths,
            phoneme_interval_list_batch=phoneme_interval_list_batch,
        )

    def get_features(
        self,
        encoder_features: Optional[torch.FloatTensor] = None,
        encoder_perturbed_features: Optional[torch.FloatTensor] = None,
        quantizer_features: Optional[torch.FloatTensor] = None,
        quantizer_perturbed_features: Optional[torch.FloatTensor] = None,
    ):
        features, perturbed_features = None, None
        if self.config.features_type == "encoder":
            features = encoder_features
        elif self.config.features_type == "quantizer":
            features = quantizer_features
        if self.config.perturbed_features_type == "encoder":
            perturbed_features = encoder_perturbed_features
        elif self.config.perturbed_features_type == "quantizer":
            perturbed_features = quantizer_perturbed_features
        return features, perturbed_features


class SimpleFrameContrastiveEncoder(BaseFrameContrastiveEncoder):
    def __init__(
        self,
        config: Union[FrameContrastiveEncoderConfig, Dict],
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
        else:
            self.linear  = nn.Identity()

    def forward(
        self,
        features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        feature_lengths: torch.LongTensor,
        mel2ph: Optional[Sequence[Sequence[List]]] = None,
        **kwargs,
    ) -> FrameContrastiveEncoderOutput:
        """
        features: batch_size, length, dim
        """
        batch_size = features.shape[0]
        if self.config.merge_phoneme_features:
            # clip_audio_from_left_side: true
            phoneme_interval_list_batch = []
            for idx in range(batch_size):
                phoneme_interval_list = PhonemeIntervalList(mel2ph=mel2ph[idx])
                phoneme_interval_list = [
                    [phoneme_interval.start, phoneme_interval.end] 
                    for phoneme_interval in phoneme_interval_list.phoneme_interval_list
                ]
                phoneme_interval_list_batch.append(phoneme_interval_list)
            merged_phoneme_output = self.merge_phoneme_features(features, phoneme_interval_list_batch)
            merged_perturbed_phoneme_output = self.merge_phoneme_features(perturbed_features, phoneme_interval_list_batch)
            features = merged_phoneme_output.features
            perturbed_features = merged_perturbed_phoneme_output.features
            feature_lengths = merged_phoneme_output.feature_lengths
        batch_size, max_length, dim = features.shape
        device = features.device
        if self.config.detach_features is not None:
            for detach_features in self.config.detach_features:
                if detach_features == "normal_features":
                    features = features.detach()
                elif detach_features == "perturbed_features":
                    perturbed_features = perturbed_features.detach()
        length_indices = torch.arange(max_length, device=device)
        mask_length_indices = torch.cat([length_indices, length_indices], dim=0)[None, :] >= feature_lengths[:, None]
        batch_indices = torch.arange(batch_size, device=device)
        padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=max_length)
        all_features = torch.cat([features, perturbed_features], dim=1) # batch_size, 2 x length, dim
        all_features = self.linear(all_features)
        cosine_logits = torch.cosine_similarity(all_features[:, None, :, :], all_features[:, :, None, :], dim=-1) # batch_size, 2 x length

        positive_attention_mask = torch.zeros(batch_size, 2 * max_length, 2 * max_length, dtype=torch.bool, device=device)
        positive_attention_mask[:, length_indices, length_indices + max_length] = True
        positive_attention_mask[:, length_indices + max_length, length_indices] = True
        positive_attention_mask[mask_length_indices, :] = False
        positive_attention_mask[mask_length_indices[:, None, :].expand(-1, 2 * max_length, -1)] = False

        logits_attention_mask = torch.ones(batch_size, 2 * max_length, 2 * max_length, dtype=torch.bool, device=device)
        logits_attention_mask[:, length_indices, length_indices] = False
        logits_attention_mask[:, length_indices + max_length, length_indices + max_length] = False
        logits_attention_mask[mask_length_indices, :] = False
        logits_attention_mask[mask_length_indices[:, None, :].expand(-1, 2 * max_length, -1)] = False

        negative_attention_mask = logits_attention_mask.clone()
        negative_attention_mask[:, length_indices, length_indices + max_length] = False
        negative_attention_mask[:, length_indices + max_length, length_indices] = False

        loss_types = self.config.loss_type.split("+")
        tot_loss = 0.0
        loss_dict = {}
        labels = torch.cat([length_indices + max_length, length_indices], dim=0)[None, :].expand(batch_size, -1)
        labels = labels.masked_fill(torch.cat([padding_mask, padding_mask], dim=1), -100) # token_ids with -100 will be masked
        encoder_output = {
            "features": features,
            "feature_lengths": feature_lengths,
            "perturbed_features": perturbed_features,
            "all_features": all_features,
            "feature_lengths": feature_lengths,
            "cosine_logits": cosine_logits,
            "logits_attention_mask": logits_attention_mask,
            "positive_logits_attention_mask": positive_attention_mask,
            "negative_logits_attention_mask": negative_attention_mask,
        }
        if self.config.merge_phoneme_features:
            encoder_output["phoneme_interval_list_batch"] = phoneme_interval_list_batch
        for i, loss_type in enumerate(loss_types):
            if loss_type == "info_nce_loss":
                _cosine_logits = cosine_logits / self.config.temperature
                # _cosine_logits = _cosine_logits.masked_fill(~logits_attention_mask, -100)
                _cosine_logits = _cosine_logits.masked_fill(~logits_attention_mask, float("-inf"))
                encoder_output["cosine_logits"] = _cosine_logits
                loss = F.cross_entropy(_cosine_logits, labels, ignore_index=-100, reduction=self.config.info_nce_loss_reduction)
                if self.config.info_nce_loss_reduction is not None and self.config.info_nce_loss_reduction == "sum":
                    loss /= features.shape[0] # loss(sum) / batch_size
                loss_dict[loss_type] = loss
                tot_loss += self.config.loss_weight[i] * loss
        return FrameContrastiveEncoderOutput(
            loss=tot_loss,
            loss_dict=loss_dict,
            **encoder_output,
        )


FrameContrastiveEncoder = SimpleFrameContrastiveEncoder


class SimclrFrameContrastiveEncoder(BaseFrameContrastiveEncoder):
    def __init__(
        self,
        config: Union[FrameContrastiveEncoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = SpeakerContrastiveEncoderConfig(**config)
        super().__init__(config=config)

    def forward(
        self,
        features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        feature_lengths: torch.LongTensor,
        mel2ph: Optional[Sequence[Sequence[List]]] = None,
        code_indices: Optional[torch.LongTensor] = None,
        perturbed_code_indices: Optional[torch.LongTensor] = None,
        training_step: Optional[int] = None,
        **kwargs,
    ) -> FrameContrastiveEncoderOutput:
        """
        features: batch_size, length, dim
        """
        batch_size = features.shape[0]
        orig_features = features.clone()
        orig_perturbed_features = perturbed_features.clone()
        if self.config.merge_phoneme_features:
            # clip_audio_from_left_side: true
            phoneme_interval_list_batch = []
            for idx in range(batch_size):
                phoneme_interval_list = PhonemeIntervalList(mel2ph=mel2ph[idx])
                phoneme_interval_list = [
                    [phoneme_interval.start, phoneme_interval.end] 
                    for phoneme_interval in phoneme_interval_list.phoneme_interval_list
                ]
                phoneme_interval_list_batch.append(phoneme_interval_list)
            merged_phoneme_output = self.merge_phoneme_features(features, phoneme_interval_list_batch)
            merged_perturbed_phoneme_output = self.merge_phoneme_features(perturbed_features, phoneme_interval_list_batch)
            features = merged_phoneme_output.features
            perturbed_features = merged_perturbed_phoneme_output.features
            feature_lengths = merged_phoneme_output.feature_lengths
            
        features = self.feature_extractor(features)
        perturbed_features = self.feature_extractor(perturbed_features)
        batch_size, max_length, dim = features.shape
        device = features.device
        if self.config.detach_features is not None:
            for detach_features in self.config.detach_features:
                if detach_features == "normal_features":
                    features = features.detach()
                elif detach_features == "perturbed_features":
                    # print("666, perturbed_features")
                    perturbed_features = perturbed_features.detach()
        length_indices = torch.arange(max_length, device=device)
        mask_length_indices = torch.cat([length_indices, length_indices], dim=0)[None, :] >= feature_lengths[:, None]
        batch_indices = torch.arange(batch_size, device=device)
        padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=max_length)
        all_features = torch.cat([features, perturbed_features], dim=1) # batch_size, 2 x length, dim
        cosine_logits = torch.cosine_similarity(all_features[:, None, :, :], all_features[:, :, None, :], dim=-1, eps=1e-06) # batch_size, 2 x length

        positive_attention_mask = torch.zeros(batch_size, 2 * max_length, 2 * max_length, dtype=torch.bool, device=device)
        positive_attention_mask[:, length_indices, length_indices + max_length] = True
        positive_attention_mask[:, length_indices + max_length, length_indices] = True
        positive_attention_mask[mask_length_indices, :] = False
        positive_attention_mask[mask_length_indices[:, None, :].expand(-1, 2 * max_length, -1)] = False

        logits_attention_mask = torch.ones(batch_size, 2 * max_length, 2 * max_length, dtype=torch.bool, device=device)
        logits_attention_mask[:, length_indices, length_indices] = False
        logits_attention_mask[:, length_indices + max_length, length_indices + max_length] = False
        logits_attention_mask[mask_length_indices, :] = False
        logits_attention_mask[mask_length_indices[:, None, :].expand(-1, 2 * max_length, -1)] = False

        negative_attention_mask = logits_attention_mask.clone()
        negative_attention_mask[:, length_indices, length_indices + max_length] = False
        negative_attention_mask[:, length_indices + max_length, length_indices] = False

        loss_types = self.config.loss_type
        tot_loss = 0.0
        loss_dict = {}
        labels = torch.cat([length_indices + max_length, length_indices], dim=0)[None, :].expand(batch_size, -1)
        labels = labels.masked_fill(torch.cat([padding_mask, padding_mask], dim=1), -100) # token_ids with -100 will be masked
        encoder_output = {
            "orig_features": orig_features,
            "orig_perturbed_features": orig_perturbed_features,
            "features": features,
            "feature_lengths": feature_lengths,
            "perturbed_features": perturbed_features,
            "all_features": all_features,
            "feature_lengths": feature_lengths,
            "cosine_logits": cosine_logits,
            "logits_attention_mask": logits_attention_mask,
            "positive_logits_attention_mask": positive_attention_mask,
            "negative_logits_attention_mask": negative_attention_mask,
            "code_indices": code_indices,
            "perturbed_code_indices": perturbed_code_indices,
        }
        if self.config.merge_phoneme_features:
            encoder_output["phoneme_interval_list_batch"] = phoneme_interval_list_batch
        for i, loss_type in enumerate(loss_types):
            if loss_type == "info_nce_loss":
                _cosine_logits = cosine_logits / self.config.temperature
                # _cosine_logits = _cosine_logits.masked_fill(~logits_attention_mask, -100)
                _cosine_logits = _cosine_logits.masked_fill(~logits_attention_mask, float("-inf"))
                encoder_output["cosine_logits"] = _cosine_logits
                loss = F.cross_entropy(_cosine_logits, labels, ignore_index=-100, reduction=self.config.info_nce_loss_reduction)
                if self.config.info_nce_loss_reduction is not None and self.config.info_nce_loss_reduction == "sum":
                    loss /= features.shape[0] # loss(sum) / batch_size
                loss_dict[loss_type] = loss
                tot_loss += self.config.loss_weight[i] * loss
            elif loss_type == "mse_loss":
                _features = features.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
                _perturbed_features = perturbed_features.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
                # _features = F.normalize(_features, dim=-1)
                # _perturbed_features = F.normalize(_perturbed_features, dim=-1)
                loss = nn.functional.mse_loss(_features, _perturbed_features, reduction=self.config.mse_loss_reduction)
                loss_dict[loss_type] = loss
                tot_loss += self.config.loss_weight[i] * loss

        if self.config.loss_weight_planner_config is not None:
            if self.config.loss_weight_planner_config.weight < self.config.loss_weight_planner_config.end:
                self.config.loss_weight_planner_config.weight += self.config.loss_weight_planner_config.linear_increase_per_step
            self.config.loss_weight[0] = self.config.loss_weight_planner_config.weight

        return FrameContrastiveEncoderOutput(
            loss=tot_loss,
            loss_dict=loss_dict,
            **encoder_output,
        )


class SimsiamFrameContrastiveEncoder(BaseFrameContrastiveEncoder):
    def __init__(
        self,
        config: Union[FrameContrastiveEncoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = SpeakerContrastiveEncoderConfig(**config)
        super().__init__(config=config)

    def asymmetric_cosine_similarity(self, z, p, feature_lengths):
        batch_size, max_length, dim = z.shape
        device = z.device
        z = z.detach()
        padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=max_length)
        attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=max_length)
        cosine_logits = torch.cosine_similarity(z, p, dim=-1) # batch_size, max_length
        cosine_logits = cosine_logits.masked_fill(padding_mask, 0.0)
        # print(666, cosine_logits.sum(), attention_mask.sum(), cosine_logits.shape, cosine_logits)
        return cosine_logits.sum() / attention_mask.sum()

    def simsiam_forward(self, z1, z2, p1, p2, feature_lengths):
        loss = -self.asymmetric_cosine_similarity(z2, p1, feature_lengths) - self.asymmetric_cosine_similarity(z1, p2, feature_lengths)
        return loss / 2

    def forward(
        self,
        features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        feature_lengths: torch.LongTensor,
        mel2ph: Optional[Sequence[Sequence[List]]] = None,
        code_indices: Optional[torch.LongTensor] = None,
        perturbed_code_indices: Optional[torch.LongTensor] = None,
        training_step: Optional[int] = None,
        **kwargs,
    ) -> FrameContrastiveEncoderOutput:
        """
        features: batch_size, length, dim
        """
        batch_size = features.shape[0]
        orig_features = features.clone()
        orig_perturbed_features = perturbed_features.clone()
        if self.config.merge_phoneme_features:
            # clip_audio_from_left_side: true
            phoneme_interval_list_batch = []
            for idx in range(batch_size):
                phoneme_interval_list = PhonemeIntervalList(mel2ph=mel2ph[idx])
                phoneme_interval_list = [
                    [phoneme_interval.start, phoneme_interval.end] 
                    for phoneme_interval in phoneme_interval_list.phoneme_interval_list
                ]
                phoneme_interval_list_batch.append(phoneme_interval_list)
            merged_phoneme_output = self.merge_phoneme_features(features, phoneme_interval_list_batch)
            merged_perturbed_phoneme_output = self.merge_phoneme_features(perturbed_features, phoneme_interval_list_batch)
            features = merged_phoneme_output.features
            perturbed_features = merged_perturbed_phoneme_output.features
            feature_lengths = merged_phoneme_output.feature_lengths
            
        features = self.feature_extractor(features)
        perturbed_features = self.feature_extractor(perturbed_features)

        encoder_output = {
            "orig_features": orig_features,
            "orig_perturbed_features": orig_perturbed_features,
            "features": features,
            "feature_lengths": feature_lengths,
            "perturbed_features": perturbed_features,
            # "cosine_logits": cosine_logits,
            "code_indices": code_indices,
            "perturbed_code_indices": perturbed_code_indices,
        }
        if self.config.merge_phoneme_features:
            encoder_output["phoneme_interval_list_batch"] = phoneme_interval_list_batch
        loss_dict = {}
        loss_dict["asymmetric_cosine_loss"] = self.simsiam_forward(
            z1=orig_features, z2=orig_perturbed_features,
            p1=features, p2=perturbed_features,
            feature_lengths=feature_lengths
        )
        tot_loss = self.config.loss_weight[0] * loss_dict["asymmetric_cosine_loss"]

        if self.config.loss_weight_planner_config is not None:
            if self.config.loss_weight_planner_config.weight < self.config.loss_weight_planner_config.end:
                self.config.loss_weight_planner_config.weight += self.config.loss_weight_planner_config.linear_increase_per_step
            self.config.loss_weight[0] = self.config.loss_weight_planner_config.weight

        return FrameContrastiveEncoderOutput(
            loss=tot_loss,
            loss_dict=loss_dict,
            **encoder_output,
        )        


class DistillFrameContrastiveEncoder(BaseFrameContrastiveEncoder):
    def __init__(
        self,
        config: Union[FrameContrastiveEncoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = SpeakerContrastiveEncoderConfig(**config)
        super().__init__(config=config)
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction=self.config.cross_entropy_loss_reduction,
            label_smoothing=self.config.cross_entropy_label_smoothing,
        )

    def forward(
        self,
        features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        feature_lengths: torch.LongTensor,
        mel2ph: Optional[Sequence[Sequence[List]]] = None,
        code_indices: Optional[torch.LongTensor] = None,
        perturbed_code_indices: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> FrameContrastiveEncoderOutput:
        """
        features: batch_size, length, dim
        """ 
        features = self.feature_extractor(features) # batch_size, length, rvq_dim
        perturbed_features = self.feature_extractor(perturbed_features) # batch_size, length, rvq_dim
        if self.config.detach_features is not None:
            for detach_features in self.config.detach_features:
                if detach_features == "normal_features":
                    features = features.detach()
                elif detach_features == "perturbed_features":
                    perturbed_features = perturbed_features.detach()
        batch_size, max_length, dim = features.shape
        device = features.device
        padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=max_length)

        loss_types = self.config.loss_type.split("+")
        tot_loss = 0.0
        loss_dict = {}
        encoder_output = {
            "features": features,
            "feature_lengths": feature_lengths,
            "perturbed_features": perturbed_features,
            "feature_lengths": feature_lengths,
            "code_indices": code_indices,
            "perturbed_code_indices": perturbed_code_indices,
        }
        for i, loss_type in enumerate(loss_types):
            if loss_type == "cross_entropy_loss":
                logits = features.view(-1, self.config.vocab_size)
                target = perturbed_code_indices.detach().masked_fill(padding_mask, -100).view(-1)
                # print(666, logits.shape, target.shape)
                loss = self.cross_entropy_loss(logits, target)
                encoder_output["logits"] = logits
                encoder_output["target"] = target
                loss_dict[loss_type] = loss
                tot_loss += self.config.loss_weight[i] * loss
        return FrameContrastiveEncoderOutput(
            loss=tot_loss,
            loss_dict=loss_dict,
            **encoder_output,
        )
