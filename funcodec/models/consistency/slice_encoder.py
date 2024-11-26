import copy
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
from phaseaug.phaseaug import PhaseAug
import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import AutoConfig
from transformers.utils import ModelOutput
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class SliceEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        slice_interval_type: Optional[str] = "random",
        static_slice_interval: Optional[Sequence[int]] = None,
        split_interval_duration: Optional[float] = None,
        split_interval_percentage: Optional[float] = 0.4,
        feature_types: Optional[str] = ["quant_out"],
        loss_types: Optional[Sequence[str]] = ["mse_loss"],
        loss_weights: Optional[Sequence[float]] = [1.0],
        mse_loss_reduction: Optional[str] = "mean",
        target_sr: Optional[int] = 16_000,
        ds_rate: Optional[int] = 320,
        update_step: Optional[int] = None,
        adaptive_loss_weight: Optional[str] = None,
        adaptive_loss_weight_clamp: Optional[Sequence[Union[str, float]]] = None,
        last_encoder_output_layer: Optional[int] = None,
        feature_extractor_type: Optional[str] = None,
        feature_extractor_config: Optional[PretrainedConfig] = None,
        **kwargs
    ):
        """
        slice_interval_type:
            static
            static_duration
            random (percentage)
        static_slice_interval:
            [12, 16] # for features
        split_interval_duration:
            0.3s
        features_type:
            encoder_output, quant_in, quant_out, decoder_input, sub_quants
        loss_types:
            mse_loss, asymmetric_mse_loss
        adaptive_loss_weight:
            codebook0_entropy
        last_encoder_output_layer:
            -1: last layer output of the encoder
        """
        super().__init__(
            slice_interval_type=slice_interval_type,
            static_slice_interval=static_slice_interval,
            split_interval_duration=split_interval_duration,
            split_interval_percentage=split_interval_percentage,
            feature_types=feature_types,
            loss_types=loss_types,
            loss_weights=loss_weights,
            mse_loss_reduction=mse_loss_reduction,
            target_sr=target_sr,
            ds_rate=ds_rate,
            update_step=update_step,
            adaptive_loss_weight=adaptive_loss_weight,
            adaptive_loss_weight_clamp=adaptive_loss_weight_clamp,
            last_encoder_output_layer=last_encoder_output_layer,
            feature_extractor_type=feature_extractor_type,
            feature_extractor_config=feature_extractor_config,
        )


@dataclass
class SliceInterval(ModelOutput):
    start_split: Optional[torch.Tensor] = None
    end_split: Optional[torch.Tensor] = None
    start_time: Optional[torch.Tensor] = None
    end_time: Optional[torch.Tensor] = None
    start_point: Optional[torch.Tensor] = None
    end_point: Optional[torch.Tensor] = None
    split_interval_lengths: Optional[torch.Tensor] = None


@dataclass
class SliceEncoderOutput(ModelOutput):
    slice_speech: Optional[torch.Tensor] = None
    speech_lengths: Optional[torch.Tensor] = None
    speech_attention_mask: Optional[torch.Tensor] = None

    gathered_features: Optional[torch.Tensor] = None
    slice_features: Optional[torch.Tensor] = None
    feature_lengths: Optional[torch.Tensor] = None
    feature_attention_mask: Optional[torch.Tensor] = None

    gathered_code_indices: Optional[torch.FloatTensor] = None
    slice_code_indices: Optional[torch.FloatTensor] = None

    slice_interval: Optional[torch.Tensor] = None

    def mse_loss(self, reduction: Optional[str] = "mean"):
        loss = F.mse_loss(self.gathered_features, self.slice_features, reduction=reduction)
        return loss


def codec_entropy_encoding(
    codebook_embeds: Optional[torch.FloatTensor] = None,
    output_features: Optional[torch.FloatTensor] = None,
    dist: Optional[torch.FloatTensor] = None,
    index: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
):
    """
    codebook_embeds: [1, dim, codebook_size]
    output_features: [batch_size, seq_len, dim]
    """
    if dist is None:
        dist = -(
            output_features.pow(2).sum(2, keepdim=True)
            - 2 * output_features @ codebook_embeds
            + codebook_embeds.pow(2).sum(1, keepdim=True)
        )
    # for numerically stable
    dist = dist - torch.max(dist, dim=-1, keepdim=True).values.detach() # [batch_size, seq_len, codebook_size]
    # pred_acc = (torch.argmax(dist, dim=-1) == index).sum() / index.numel()
    # loss = F.cross_entropy(dist.transpose(1, 2), index, reduction="mean")
    # print(loss)
    p = F.log_softmax(dist, dim=-1)
    y = F.one_hot(index, num_classes=codebook_embeds.shape[-1])
    entropy = -(y.float() * p).sum(dim=-1) # [batch_size, seq_len]
    if attention_mask is not None:
        entropy = entropy * attention_mask
    # loss = entropy.sum()
    return entropy


def codec_dist_logits(
    codebook_embeds: Optional[torch.FloatTensor] = None,
    output_features: Optional[torch.FloatTensor] = None,
    dist: Optional[torch.FloatTensor] = None,
    index: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
):
    """
    codebook_embeds: [1, dim, codebook_size]
    output_features: [batch_size, seq_len, dim]
    """
    if dist is None:
        dist = -(
            output_features.pow(2).sum(2, keepdim=True)
            - 2 * output_features @ codebook_embeds
            + codebook_embeds.pow(2).sum(1, keepdim=True)
        )
    # for numerically stable
    dist = dist - torch.max(dist, dim=-1, keepdim=True).values.detach() # [batch_size, seq_len, codebook_size]
    return dist


class SliceEncoder(PreTrainedModel):
    config_class = SliceEncoderConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[dict, SliceEncoderConfig],
        **kwargs
    ):
        if isinstance(config, dict):
            config = SliceEncoderConfig(**config)
        super().__init__(config=config)
        self.tokens_per_second = int(self.config.target_sr / self.config.ds_rate)
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
        else:
            self.feature_extractor = nn.Identity()
    
    def get_slice_interval_from_split(self, start_split: int, end_split: int):
        start_time = start_split / tokens_per_second
        end_time = end_split / tokens_per_second
        start_point = self.config.ds_rate * start_split
        end_point = self.config.ds_rate * end_split
        return SliceInterval(
            start_split=start_split, end_split=end_split,
            start_time=start_time, end_time=end_time,
            start_point=start_point, end_point=end_point,
        )

    def sample_split_intervals(
        self,
        feature_lengths: torch.LongTensor,
        **kwargs
    ) -> SliceInterval:
        batch_size = feature_lengths.shape[0]
        device = feature_lengths.device
        if self.config.slice_interval_type == "static":
            start_split, end_split = self.config.static_slice_interval
            split_interval = self.get_slice_interval_from_split(start_split, end_split)
            slice_interval = [slice_interval for _ in range(batch_size)]
        elif self.config.slice_interval_type in ["static_duration", "random"]:
            if self.config.slice_interval_type == "static_duration":
                split_interval_length = int(self.config.split_interval_duration * self.tokens_per_second)
                split_interval_lengths = torch.tensor([split_interval_length] * batch_size, dtype=torch.long, device=device)
                split_interval_lengths = torch.where(split_interval_lengths == 0, feature_lengths, split_interval_lengths)
                split_interval_lengths = torch.where(split_interval_lengths >= feature_lengths, feature_lengths, split_interval_lengths)
            elif self.config.slice_interval_type == "random":
                split_interval_lengths = (feature_lengths * self.config.split_interval_percentage).floor()
                split_interval_lengths = torch.where(split_interval_lengths == 0, feature_lengths, split_interval_lengths)
            start_positions = torch.rand(batch_size).to(device) * (feature_lengths - split_interval_lengths)
            end_positions = start_positions + split_interval_lengths
            end_positions = torch.where(end_positions >= feature_lengths, feature_lengths, end_positions)
            start_positions = start_positions.long()
            end_positions = end_positions.long()
            split_intervals = torch.stack((start_positions, end_positions.long()), dim=1)
            time_intervals = split_intervals / self.tokens_per_second
            point_intervals = split_intervals * self.config.ds_rate
            slice_interval = SliceInterval(
                start_split=split_intervals[:, 0],
                end_split=split_intervals[:, 1],
                start_time=time_intervals[:, 0],
                end_time=time_intervals[:, 1],
                start_point=point_intervals[:, 0], 
                end_point=point_intervals[:, 1], 
                split_interval_lengths=split_intervals[:, 1] - split_intervals[:, 0],
            )
        else:
            raise NotImplementedError()
        return slice_interval

    def gather_speech(
        self,
        speech: torch.FloatTensor,
        speech_lengths: torch.LongTensor,
        slice_interval: SliceInterval,
    ):
        batch_size, seq_len = speech.shape
        device = speech.device
        start_point = slice_interval.start_point
        end_point = slice_interval.end_point
        point_interval_lengths = end_point - start_point

        interval_length = point_interval_lengths.max().item()
        indices = torch.arange(interval_length).expand(batch_size, interval_length).to(device)
        start_point = start_point.unsqueeze(-1)
        adjusted_indices = start_point + indices
        attention_mask = adjusted_indices < end_point.unsqueeze(-1).expand(batch_size, interval_length)
        attention_masked_indices = adjusted_indices * attention_mask
        gathered_speech = torch.gather(speech, 1, attention_masked_indices)
        gathered_speech = gathered_speech * attention_mask.type_as(gathered_speech)
        return SliceEncoderOutput(
            slice_speech=gathered_speech,
            speech_lengths=point_interval_lengths,
            speech_attention_mask=attention_mask,
            # slice_interval=slice_interval,
        )

    def gather_features(
        self,
        features: Optional[torch.FloatTensor] = None,
        slice_features: Optional[torch.FloatTensor] = None,
        code_indices: Optional[torch.FloatTensor] = None,
        slice_code_indices: Optional[torch.FloatTensor] = None,
        slice_feature_lengths: Optional[torch.LongTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        slice_interval: Optional[SliceInterval] = None,
    ):
        if features.dim() == 3:
            sub_quant_first = False # [batch_size, seq_len, dim]
            batch_size, seq_len, dim = features.shape
        elif features.dim() == 4:
            sub_quant_first = True  # [num_sub_quants, batch_size, seq_len, dim]
            num_sub_quants, batch_size, seq_len, dim = features.shape
            # print(666, features.shape)
        device = features.device
        if slice_interval is None:
            slice_interval = self.sample_split_intervals(feature_lengths)
        start_split = slice_interval.start_split
        end_split = slice_interval.end_split
        split_interval_lengths = slice_interval.split_interval_lengths
        interval_length = split_interval_lengths.max().item()
        if not sub_quant_first:
            indices = torch.arange(interval_length).expand(batch_size, interval_length).to(device)
        else:
            indices = torch.arange(interval_length).expand(num_sub_quants, batch_size, interval_length).to(device)
        start_split = start_split.unsqueeze(-1)
        adjusted_indices = start_split + indices
        # print(adjusted_indices)
        attention_mask = adjusted_indices < end_split.unsqueeze(-1).expand(batch_size, interval_length)
        # print(attention_mask)
        attention_masked_indices = adjusted_indices * attention_mask
        # print(attention_masked_indices)
        if not sub_quant_first:
            gathered_features = torch.gather(features, 1, attention_masked_indices.unsqueeze(-1).expand(-1, -1, dim))
        else:
            gathered_features = torch.gather(features, 2, attention_masked_indices.unsqueeze(-1).expand(-1, -1, -1, dim))
            attention_mask = attention_mask.expand(num_sub_quants, -1, -1)
        # print(666, gathered_features.shape, slice_features.shape)
        return SliceEncoderOutput(
            gathered_features=gathered_features * attention_mask.unsqueeze(-1).type_as(gathered_features),
            slice_features=slice_features * attention_mask.unsqueeze(-1).type_as(slice_features),
            feature_lengths=split_interval_lengths,
            feature_attention_mask=attention_mask,
            slice_interval=slice_interval,
        )

    def gather_code_indices(
        self,
        code_indices: Optional[torch.FloatTensor] = None,
        slice_code_indices: Optional[torch.FloatTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        slice_feature_lengths: Optional[torch.LongTensor] = None,
        slice_interval: Optional[SliceInterval] = None,
    ):
        num_codebooks, batch_size, seq_len = code_indices.shape
        device = code_indices.device
        if slice_interval is None:
            slice_interval = self.sample_split_intervals(feature_lengths)
        start_split = slice_interval.start_split
        end_split = slice_interval.end_split
        split_interval_lengths = slice_interval.split_interval_lengths
        interval_length = split_interval_lengths.max().item()
        indices = torch.arange(interval_length).expand(num_codebooks, batch_size, interval_length).to(device)
        start_split = start_split.unsqueeze(-1)
        adjusted_indices = start_split + indices
        attention_mask = adjusted_indices < end_split.unsqueeze(-1).expand(batch_size, interval_length)
        # print(666, code_indices.shape, slice_code_indices.shape, adjusted_indices.shape, attention_mask.shape)
        gathered_code_indices = torch.gather(code_indices, 2, adjusted_indices)
        return SliceEncoderOutput(
            gathered_code_indices=gathered_code_indices,
            slice_code_indices=slice_code_indices,
            feature_lengths=split_interval_lengths,
            feature_attention_mask=attention_mask,
            slice_interval=slice_interval,
        )

    def asymmetric_mse_loss_for_one_side(self, z, p):
        batch_size, max_length, dim = z.shape
        device = z.device
        z = z.detach()
        diff = z - p # batch_size, max_length, dim
        mse_loss = diff ** 2
        return mse_loss

    def asymmetric_mse_loss(self, z1, z2, p1, p2, attention_mask):
        batch_size, max_length, dim = z1.shape
        mse_loss = self.asymmetric_mse_loss_for_one_side(z2, p1) + self.asymmetric_mse_loss_for_one_side(z1, p2)
        mse_loss = mse_loss * attention_mask.unsqueeze(-1)
        mse_loss = mse_loss.sum() / attention_mask.sum() / dim
        return mse_loss / 2

    def forward(
        self,
        speech: Optional[torch.FloatTensor] = None,
        slice_speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        quant_in: Optional[torch.FloatTensor] = None,
        slice_quant_in: Optional[torch.FloatTensor] = None,
        quant_out: Optional[torch.FloatTensor] = None,
        slice_quant_out: Optional[torch.FloatTensor] = None,
        sub_quants: Optional[torch.FloatTensor] = None,
        slice_sub_quants: Optional[torch.FloatTensor] = None,
        code_indices: Optional[torch.FloatTensor] = None,
        slice_code_indices: Optional[torch.FloatTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        slice_feature_lengths: Optional[torch.LongTensor] = None,
        slice_interval: Optional[SliceInterval] = None,
        codebook: Optional[torch.FloatTensor] = None,
        forward_step: Optional[int] = None,
        **kwargs
    ):
        if slice_interval is None:
            slice_interval = self.sample_split_intervals(feature_lengths)
        return_dict = OrderedDict()
        return_dict["loss"] = torch.tensor([0.0], device=speech.device)
        return_dict["loss_dict"] = OrderedDict()
        for i, feature_type in enumerate(self.config.feature_types):
            if feature_type == "quant_in":
                features = quant_in
                slice_features = slice_quant_in
            elif feature_type == "quant_out":
                features = quant_out
                slice_features = slice_quant_out
            elif feature_type == "sub_quants":
                features = sub_quants
                slice_features = slice_sub_quants
            if feature_type not in return_dict:
                output = self.gather_features(
                    features=features,
                    slice_features=slice_features,
                    feature_lengths=feature_lengths,
                    slice_feature_lengths=slice_feature_lengths,
                    slice_interval=slice_interval,
                )
                return_dict[feature_type] = output
            for loss_type in self.config.loss_types:
                if loss_type in ("mse_loss", "asymmetric_mse_loss"):
                    if self.config.adaptive_loss_weight is None:
                        tot = return_dict[feature_type].feature_attention_mask.sum()
                        if loss_type == "mse_loss":
                            mse_loss = output.mse_loss(reduction=self.config.mse_loss_reduction)
                            return_dict["loss_dict"]["mse_loss"] = mse_loss
                        elif loss_type == "asymmetric_mse_loss":
                            z1 = return_dict[feature_type].gathered_features
                            z2 = return_dict[feature_type].slice_features
                            p1 = self.feature_extractor(z1) + z1
                            p2 = self.feature_extractor(z2) + z2
                            mse_loss = self.asymmetric_mse_loss(z1, z2, p1, p2, return_dict[feature_type].feature_attention_mask)
                            return_dict["loss_dict"]["asymmetric_mse_loss"] = mse_loss
                        if self.config.update_step is None or self.config.update_step <= forward_step:
                            return_dict["loss"] += self.config.loss_weights[i] * mse_loss
                        if "code_indices" not in return_dict:
                            return_dict["code_indices"] = self.gather_code_indices(
                                code_indices=code_indices,
                                slice_code_indices=slice_code_indices,
                                feature_lengths=feature_lengths,
                                slice_feature_lengths=slice_feature_lengths,
                                slice_interval=slice_interval,
                            )
                        return_dict["loss_dict"]["quantizer0_consistency"] = ((return_dict["code_indices"].gathered_code_indices[0] == return_dict["code_indices"].slice_code_indices[0]) * return_dict[feature_type].feature_attention_mask).sum() / tot
                    elif self.config.adaptive_loss_weight == "codebook0_entropy":
                        if "code_indices" not in return_dict:
                            return_dict["code_indices"] = self.gather_code_indices(
                                code_indices=code_indices,
                                slice_code_indices=slice_code_indices,
                                feature_lengths=feature_lengths,
                                slice_feature_lengths=slice_feature_lengths,
                                slice_interval=slice_interval,
                            )
                        if "sub_quants" not in return_dict:
                            return_dict["sub_quants"] = self.gather_features(
                                features=sub_quants,
                                slice_features=slice_sub_quants,
                                feature_lengths=feature_lengths,
                                slice_feature_lengths=slice_feature_lengths,
                                slice_interval=slice_interval,
                            )
                        entropy1 = codec_entropy_encoding(
                            codebook_embeds=codebook[0].t().unsqueeze(0),
                            output_features=return_dict[feature_type].slice_features,
                            index=return_dict["code_indices"].gathered_code_indices[0],
                            attention_mask=return_dict[feature_type].feature_attention_mask
                        )
                        entropy2 = codec_entropy_encoding(
                            codebook_embeds=codebook[0].t().unsqueeze(0),
                            output_features=return_dict[feature_type].gathered_features,
                            index=return_dict["code_indices"].slice_code_indices[0],
                            attention_mask=return_dict[feature_type].feature_attention_mask
                        )
                        tot = return_dict[feature_type].feature_attention_mask.sum()
                        # tot = return_dict[feature_type].feature_attention_mask.shape[0] * return_dict[feature_type].feature_attention_mask.shape[1]
                        return_dict["loss_dict"]["entropy1"] = entropy1.sum() / tot
                        return_dict["loss_dict"]["entropy2"] = entropy2.sum() / tot
                        weight = (entropy1 + entropy2).detach()
                        weight = self.config.loss_weights[i] * weight
                        if self.config.adaptive_loss_weight_clamp is not None:
                            weight = torch.clamp(weight, min=self.config.adaptive_loss_weight_clamp[0], max=self.config.adaptive_loss_weight_clamp[1])
                            weight = return_dict[feature_type].feature_attention_mask * weight
                        return_dict["weight"] = weight
                        diff = return_dict[feature_type].gathered_features - return_dict[feature_type].slice_features
                        mse_loss = diff ** 2
                        dim = return_dict[feature_type].gathered_features.shape[-1]
                        return_dict["loss_dict"]["mse_loss"] = mse_loss.sum() / tot / dim
                        mse_loss = weight.unsqueeze(-1) * mse_loss
                        if self.config.mse_loss_reduction == "mean":
                            mse_loss = mse_loss.sum() / tot / dim
                        elif self.config.mse_loss_reduction == "sum":
                            mse_loss = mse_loss.sum()
                        if self.config.update_step is None or self.config.update_step >= forward_step:
                            return_dict["loss"] += mse_loss
                        return_dict["loss_dict"]["quantizer0_consistency"] = ((return_dict["code_indices"].gathered_code_indices[0] == return_dict["code_indices"].slice_code_indices[0]) * return_dict[feature_type].feature_attention_mask).sum() / tot
                        
        return return_dict
