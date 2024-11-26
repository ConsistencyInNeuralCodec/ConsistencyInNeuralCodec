import sys, importlib
import os, time, random, math
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
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .configuration_contrastive_encoder import ContrastiveEncoderConfig
from ..utils import lengths_to_padding_mask, lengths_to_attention_mask

logger = logging.getLogger(__name__)


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    lengths: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    modified from transformers.models.wav2vec2.modeling_wav2vec2
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = lengths.detach().tolist() if lengths is not None else [sequence_length for _ in range(batch_size)]

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    sequence_length_range = np.arange(sequence_length)

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # avoid sampling the same positive vector, but keep the distribution uniform
        sampled_indices[sampled_indices >= feature_indices] += 1

        # remap to actual indices
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # correct for batch size
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


def compute_contrastive_logits(
    positive_feature: torch.FloatTensor,
    negative_features: torch.FloatTensor,
    predicted_features: torch.FloatTensor,
    temperature: int = 0.1,
):
    """
    Compute logits for contrastive loss based using cosine similarity as the distance measure between
    `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
    positive_features: [1, batch_size, num_frames, dim]
    negative_features: [num_negatives, batch_size, num_frames, dim]
    predicted_features: [batch_size, num_frames, dim]
    """
    target_features = torch.cat([positive_feature, negative_features], dim=0)
    logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(target_features)
    # apply temperature
    logits = logits / temperature
    return logits


@dataclass
class ContrastiveEncoderOutput(ModelOutput):
    positive_indices: Optional[Union[Sequence, torch.Tensor]] = None
    positive_features: torch.FloatTensor = None
    negative_indices: Optional[Union[Sequence, torch.Tensor]] = None
    negative_features: torch.FloatTensor = None
    predicted_features: torch.FloatTensor = None
    mask_time_indices: torch.FloatTensor = None
    sampled_negative_indices: torch.FloatTensor = None
    loss: torch.FloatTensor = None


@dataclass
class PhonemeInterval(ModelOutput):
    # [start_idx, end_idx)
    phoneme_idx: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None
    num: Optional[int] = None


class PhonemeIntervalList:
    def __init__(
        self,
        mel2ph:Optional[Sequence] = None,
        phoneme_interval_list=None,
    ):
        self.mel2ph = mel2ph
        if phoneme_interval_list is None:
            phoneme_interval_list = self.get_phoneme_interval_list(mel2ph)
        self.phoneme_interval_list = phoneme_interval_list
    
    @staticmethod
    def get_phoneme_interval_list(mel2ph):
        start_idx = 0
        phoneme_interval_list = []

        for i in range(1, len(mel2ph)):
            if mel2ph[i] != mel2ph[i - 1]:
                end_idx = i
                phoneme_interval = PhonemeInterval(
                    phoneme_idx=mel2ph[start_idx],
                    start=start_idx, end=end_idx, num=end_idx - start_idx,
                )
                phoneme_interval_list.append(phoneme_interval)
                start_idx = i
        end_idx = len(mel2ph)
        phoneme_interval = PhonemeInterval(
            phoneme_idx=mel2ph[start_idx],
            start=start_idx, end=end_idx, num=end_idx - start_idx,
        )
        phoneme_interval_list.append(phoneme_interval)
        return phoneme_interval_list

    def __len__(self):
        return len(self.phoneme_interval_list)

    def __getitem__(self, item: int):
        return self.phoneme_interval_list[item]

    def __iter__(self):
        # setattr(self, "idx", 0)
        return self
        
    # def __next__(self):
    #     for i in range(len(self)):
    #         yield self[i]
    #     raise StopIteration()

    def __next__(self):
        if not hasattr(self, "idx"):
            setattr(self, "idx", 0)
        if self.idx < len(self):
            item = self[self.idx]
            self.idx += 1
            return item
        else:
            raise StopIteration()

    def prev_neighbors(self, item, num_neighbors):
        assert num_neighbors <= len(self) - 1
        item += len(self)
        phoneme_interval_list = self.phoneme_interval_list + self.phoneme_interval_list
        return phoneme_interval_list[item - num_neighbors: item]

    def next_neighbors(self, item, num_neighbors):
        assert num_neighbors <= len(self) - 1
        phoneme_interval_list = self.phoneme_interval_list + self.phoneme_interval_list
        return phoneme_interval_list[item: item + num_neighbors]

    def tolist(self):
        indices = []
        for phoneme_interval in self.phoneme_interval_list:
            indices += list(range(phoneme_interval.start, phoneme_interval.end))
        return indices


@dataclass
class PositiveExample(ModelOutput):
    feature: Optional[torch.FloatTensor] = None # [seq_len, dim]
    predicted_positive_pairs: Optional[Sequence[Union[Sequence, torch.FloatTensor]]] = None # [[1,2], [2, 1], [3, 4]]


@dataclass
class NegativeExample(ModelOutput):
    feature: Optional[torch.FloatTensor] = None # [seq_len, dim]
    negative_indices_list: Optional[Sequence[Union[Sequence, torch.FloatTensor]]] = None


@dataclass
class ContrastivePair(ModelOutput):
    # predicted_features: Optional[torch.FloatTensor] = None
    # predicted_indices: Optional[torch.FloatTensor] = None
    # positive_features: Optional[torch.FloatTensor] = None
    # positive_indices: Optional[torch.FloatTensor] = None
    # negative_features: Optional[torch.FloatTensor] = None
    # negative_indices: Optional[Union[torch.FloatTensor, Sequence[torch.FloatTensor]]] = None
    # cat_negative_features: Optional[bool] = True
    ori_predicted_indices: Optional[torch.FloatTensor] = None
    ori_positive_indices: Optional[torch.FloatTensor] = None
    ori_negative_indices: Optional[torch.FloatTensor] = None
    predicted_features: Optional[torch.FloatTensor] = None
    predicted_indices: Optional[torch.FloatTensor] = None
    positive_features: Optional[torch.FloatTensor] = None
    positive_indices: Optional[torch.FloatTensor] = None
    negative_features: Optional[torch.FloatTensor] = None
    negative_indices: Optional[torch.FloatTensor] = None
    cat_negative_features: Optional[torch.FloatTensor] = None
    attention_mask_for_negative_indices: Optional[torch.FloatTensor] = None
    batch_size: Optional[torch.FloatTensor] = None
    sampled_feature_lengths: Optional[torch.FloatTensor] = None
    phoneme_interval_list_batch: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict[str, torch.FloatTensor]] = None

    @property
    def sampled_predicted_features(self):
        return self.predicted_features[self.predicted_indices]

    @property
    def sampled_positive_features(self):
        return self.positive_features[self.positive_indices]

    @property
    def sampled_negative_features(self):
        if self.cat_negative_features == True or self.attention_mask_for_negative_indices is not None:
            return self.negative_features[self.negative_indices]
        else:
            sampled_negative_features = [self.negative_features[_negative_indices] for _negative_indices in self.negative_indices]
            return sampled_negative_features


    @property
    def cosine_similarity(self):
        sampled_predicted_features = self.sampled_predicted_features
        sampled_positive_features = self.sampled_positive_features
        cat_negative_features = getattr(self, "cat_negative_features", None)
        if cat_negative_features is None:
            negative_indices_length = [len(_negative_indices) for _negative_indices in self.negative_indices]
            if max(negative_indices_length) == min(negative_indices_length):
                cat_negative_features = True
            else:
                cat_negative_features = False
        sampled_negative_features = self.sampled_negative_features
        if cat_negative_features == True or self.attention_mask_for_negative_indices is not None:
            # print(666, sampled_positive_features[:, None, :].shape, self.negative_indices.shape, sampled_negative_features.shape)
            cos_sim_logits = torch.cosine_similarity(
                sampled_predicted_features[:, None, :],
                torch.cat([sampled_positive_features[:, None, :], sampled_negative_features], dim=1),
                dim=-1,
            )
        else:
            cos_sim_logits = []
            for idx, _sampled_negative_features in enumerate(self.sampled_negative_features):
                cos_sim_logit = torch.cosine_similarity(
                    self.sampled_predicted_features[idx][None, :],
                    torch.cat([self.sampled_positive_features[idx][None, :], _sampled_negative_features], dim=0),
                    dim=-1,
                )
                cos_sim_logits.append(cos_sim_logit)
        return cos_sim_logits

    def infonce_loss(
        self,
        cos_sim_logits: Optional[torch.FloatTensor] = None,
        temperature: Optional[int] = 1.0,
        info_nce_loss_reduction: Optional[str] = "mean",
        batch_size: Optional[int] = 1.0,
    ):
        if cos_sim_logits is None:
            cos_sim_logits = self.cosine_similarity
        if isinstance(cos_sim_logits, torch.Tensor):
            labels = torch.zeros(cos_sim_logits.shape[0], dtype=torch.long, device=cos_sim_logits.device)
            loss = F.cross_entropy(cos_sim_logits / temperature, labels, reduction=info_nce_loss_reduction)
        elif self.attention_mask_for_negative_indices is not None:
            labels = torch.zeros(cos_sim_logits.shape[0], dtype=torch.long, device=cos_sim_logits.device)
            cos_sim_logits = cos_sim_logits / temperature
            # cos_sim_logits[:, 1:] = torch.where(self.attention_mask_for_negative_indices, cos_sim_logits[:, 1:], -100)
            cos_sim_logits[:, 1:] = cos_sim_logits[:, 1:].masked_fill(~self.attention_mask_for_negative_indices, float("-inf"))
            loss = F.cross_entropy(cos_sim_logits, labels, reduction=info_nce_loss_reduction)
        else:
            loss = 0.0
            for _cos_sim_logits in cos_sim_logits:
                labels = torch.zeros(1, dtype=torch.long, device=_cos_sim_logits.device)
                _cos_sim_logits = _cos_sim_logits[None]
                _loss = F.cross_entropy(_cos_sim_logits / temperature, labels)
                loss += _loss
            if info_nce_loss_reduction is None or info_nce_loss_reduction == "mean":
                loss /= len(cos_sim_logits)
        if info_nce_loss_reduction is not None and info_nce_loss_reduction == "sum":
            loss = loss / batch_size
        return loss

    def cosine_similarity_loss(
        self,
        cos_sim_logits: Optional[torch.FloatTensor] = None,
    ):
        if cos_sim_logits is None:
            cos_sim_logits = self.cosine_similarity
        if isinstance(cos_sim_logits, torch.Tensor):
            intra_frame_loss = cos_sim_logits[:, :1]
            inter_frame_loss = cos_sim_logits[:, 1:]
            if self.attention_mask_for_negative_indices is not None:
                inter_frame_loss = inter_frame_loss.masked_fill(~self.attention_mask_for_negative_indices, 0.0)
                # print("tensor, 666")
        elif self.attention_mask_for_negative_indices is not None:
            intra_frame_loss = cos_sim_logits[:, :1]
            inter_frame_loss = cos_sim_logits[:, 1:]
            # inter_frame_loss = torch.where(self.attention_mask_for_negative_indices, cos_sim_logits[:, 1:], 0.0)
            inter_frame_loss = inter_frame_loss.masked_fill(~self.attention_mask_for_negative_indices, 0.0)
            # print("attention_mask_for_negative_indices, 666")
        else:
            intra_frame_loss, inter_frame_loss = 0.0, 0.0
            for _cos_sim_logits in cos_sim_logits:
                intra_frame_loss += _cos_sim_logits[:1].sum()
                inter_frame_loss += _cos_sim_logits[1:].sum()
        loss = -intra_frame_loss.sum() + inter_frame_loss.sum()
        return loss


class ContrastiveEncoderPreTrainedModel(PreTrainedModel):
    config_class = ContrastiveEncoderConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[ContrastiveEncoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = ContrastiveEncoderConfig(**config)
        super().__init__(config=config)
        self.project = nn.ModuleDict()
        for prefix in ("positive", "negative", "predicted"):
            if getattr(config, f"{prefix}_project", False):
                prefix_in_dim = getattr(config, f"{prefix}_in_dim")
                prefix_out_dim = getattr(config, f"{prefix}_out_dim")
                self.project[prefix] = nn.Linear(prefix_in_dim, prefix_out_dim)
        self.init_feature_extractor()

    def init_feature_extractor(self):
        if self.config.feature_extractor_type == "cnn_lstm":
            from funcodec.modules.cnn_lstm import CNNLSTM, CNNLSTMConfig
            self.config.feature_extractor_config = CNNLSTMConfig(**self.config.feature_extractor_config)
            self.feature_extractor = CNNLSTM(**self.config.feature_extractor_config.to_dict())
        else:
            self.feature_extractor = nn.Identity()

    @staticmethod
    def build_contrastive_encoder(
        config: Optional[Union[ContrastiveEncoderConfig, Dict]] = None,
        **kwargs
    ) -> "ContrastiveEncoderPreTrainedModel":
        if config is not None:
            if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
                config = ContrastiveEncoderConfig(**config)
            if config.encoder_type == "transformers.Wav2Vec2":
                return ContrastiveEncoderForWav2Vec2(config=config)
            elif config.encoder_type in ("SICF", "SpeakerIndependentContentFeature"):
                return ContrastiveEncoderForSICF(config=config)
            else:
                raise NotImplementedError
        raise NotImplementedError


class ContrastiveEncoderForWav2Vec2(ContrastiveEncoderPreTrainedModel):  
    def forward(
        self,
        encoder_features: Optional[torch.Tensor] = None,
        quantized_features: Optional[torch.Tensor] = None,
        feature_lengths: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> ContrastiveEncoderOutput:
        """
        encoder_features: [batch_size, time_step, hidden_size]
        quantized_features: [batch_size, time_step, hidden_size]
        """
        batch_size, sequence_length, hidden_size = encoder_features.shape
        if mask_time_indices is None:
            mask_time_indices = _compute_mask_indices(
                shape=(batch_size, sequence_length), 
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                lengths=feature_lengths,
            )
        if sampled_negative_indices is None:
            sampled_negative_indices = _sample_negative_indices(
                features_shape=(batch_size, sequence_length),
                num_negatives=self.config.num_negatives,
                mask_time_indices=mask_time_indices,
            )
        mask_time_indices = torch.tensor(data=mask_time_indices, device=encoder_features.device, dtype=torch.long)
        sampled_negative_indices = torch.tensor(data=sampled_negative_indices, device=encoder_features.device, dtype=torch.long)
        prefix_features = {}
        for prefix in ("positive", "negative", "predicted"):
            prefix_type = getattr(self.config, f"{prefix}_type")
            if prefix_type == "encoder_output":
                prefix_features[f"{prefix}_features"] = encoder_features
            elif prefix_type == "quantized":
                prefix_features[f"{prefix}_features"] = quantized_features
            if getattr(self.config, f"{prefix}_project", False):
                prefix_features[f"{prefix}_features"] = self.project[prefix](prefix_features[f"{prefix}_features"])
        positive_features, negative_features, predicted_features = prefix_features["positive_features"], prefix_features["negative_features"], prefix_features["predicted_features"]
        # 3. sample K negatives (distractors) quantized states for contrastive loss
        # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
        # sample negative quantized vectors BTC => (BxT)C
        negative_features = negative_features.reshape(-1, hidden_size)[sampled_negative_indices.long().view(-1)]
        negative_features = negative_features.view(
            batch_size, sequence_length, -1, hidden_size
        ).permute(2, 0, 1, 3) # [num_negatives, B, T, C]
        logits = compute_contrastive_logits(
            positive_features[None, :],
            negative_features,
            predicted_features,
            self.config.temperature,
        )
        neg_is_pos = (positive_features == negative_features).all(-1)
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
        # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
        logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
        target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
        contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
        contrastive_loss *= self.config.loss_weight[0]
        return ContrastiveEncoderOutput(
            positive_features=positive_features,
            negative_features=negative_features,
            predicted_features=predicted_features,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
            loss=contrastive_loss,
        )


def random_choice_for_arange(size: int, quantity: int, device: Optional[torch.device] = torch.device("cpu"), replace: Optional[bool] = False):
    if not replace or quantity > 1:
        return torch.randperm(size, dtype=torch.long, device=device)[:quantity]
    else:
        return torch.randint(low=0, high=size, size=(quantity,), dtype=torch.long, device=device)


def random_choice_for_array(array: torch.Tensor, quantity: int, replace: Optional[bool] = False):
    """
    array: shape N
    """
    idx_tensor = random_choice_for_arange(
        size=array.shape[0], quantity=quantity, device=array.device, replace=replace
    )
    return array[idx_tensor]


def random_shuffle_for_array(array: torch.Tensor):
    return random_choice_for_array(array, array.shape[0])


def sample_with_order(array: torch.Tensor, quantity: int):
    """
    array: shape N
    """
    sampled_indices = random_choice_for_arange(size=array.shape[0], quantity=quantity, device=array.device, replace=False)
    sorted_sampled_indices = torch.sort(sampled_indices)
    sampled_array = array[sorted_sampled_indices]
    return sampled_array


def append_random_sample(array: torch.Tensor):
    if array.shape[0] % 2 != 0:
        # 随机采样一个数字（除去最后一个数字）
        sampled_tensor = random_choice_for_array(array[:-1], 1, False)
        array = torch.cat([array, sampled_tensor], dim=-1)
    return array


def sample_indices(
    feature_length: int,
    num_samples: int,
    mask_indices: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    if mask_indices is not None and device is None:
        device = mask_indices.device
    # sampled_indices = torch.zeros(size=(feature_length, num_samples), dtype=torch.long, device=device)
    mask_indices = mask_indices.bool() if mask_indices is not None else torch.zeros(feature_length, dtype=torch.bool, device=device)
    accessible_indices = torch.where(mask_indices == False)[0]
    # print(666, mask_indices.shape, accessible_indices.shape)
    num_accessible = accessible_indices.shape[0]
    replace = False if num_accessible >= num_samples else True
    sampled_indices_flat = random_choice_for_array(accessible_indices, num_samples, replace=replace)
    return sampled_indices_flat


def get_indices(
    feature_length: int,
    mask_indices: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    if mask_indices is not None and device is None:
        device = mask_indices.device
    mask_indices = mask_indices.bool() if mask_indices is not None else torch.zeros(feature_length, dtype=torch.bool, device=device)
    accessible_indices = torch.where(mask_indices == False)[0]
    # print(666, mask_indices.shape, accessible_indices.shape)
    return accessible_indices


class PositiveSampler:
    @staticmethod
    def sample(
        sample_positive_strategy: str,
        sample_positive_quantity: str,
        phoneme_interval: PhonemeInterval,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> PositiveExample:
        # print(f"666 PositiveSampler device = {device}")
        feature_idx_range = torch.arange(phoneme_interval.start, phoneme_interval.end, device=device)
        if isinstance(sample_positive_quantity, str):
            if sample_positive_quantity == "all":
                pass
            else:
                raise NotImplementedError()
        elif isinstance(sample_positive_quantity, float):
            percentage = sample_positive_quantity
            quantity = percentage * feature_idx_range.shape[0]
            quantity = math.ceil(quantity)
            feature_idx_range = sample_with_order(feature_idx_range, quantity)
        elif isinstance(sample_positive_quantity, int):
            feature_idx_range = sample_with_order(feature_idx_range, sample_positive_quantity)
        else:
            raise NotImplementedError()
        if sample_positive_strategy == "random":
            feature_idx_range = random_shuffle_for_array(feature_idx_range)
        elif sample_positive_strategy == "neighbor":
            pairs = []
            feature_idx_range = append_random_sample(feature_idx_range)
            for i in range(0, feature_idx_range.shape[0], 2):
                pair = [feature_idx_range[i], feature_idx_range[i+1]]
                pairs.append(pair)
        elif sample_positive_strategy == "neighbor_all":
            pairs = []
            # feature_idx_range = append_random_sample(feature_idx_range)
            for i in range(0, feature_idx_range.shape[0] - 1):
                pair = [feature_idx_range[i], feature_idx_range[i + 1]]
                pairs.append(pair)
        else:
            raise NotImplementedError()
        return PositiveExample(predicted_positive_pairs=pairs)


class NegativeSampler:
    @staticmethod
    def sample(
        sample_negative_strategy: str,
        sample_negative_quantity: Union[str, float, int],
        phoneme_interval: PhonemeInterval,
        num_predicted_positive_pairs: int,
        feature_length: int,
        # for neighbor
        num_negative_neighbors: Optional[int] = None,
        item: Optional[int] = None,
        phoneme_interval_list: Optional[PhonemeIntervalList] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> NegativeExample:
        # print(f"666 NegativeSampler device = {device}")
        features_shape = (num_predicted_positive_pairs, feature_length)
        mask_indices = torch.zeros(features_shape, device=device)
        mask_indices[:, phoneme_interval.start: phoneme_interval.end] = 1
        if isinstance(sample_negative_quantity, str):
            if sample_negative_quantity == "all":
                quantity = feature_length - phoneme_interval.num
            else:
                raise NotImplementedError()
        elif isinstance(sample_negative_quantity, float):
            percentage = quantity
            quantity = percentage * (feature_length - phoneme_interval.num)
            quantity = math.ceil(quantity)
        elif isinstance(sample_negative_quantity, int):
            quantity = sample_negative_quantity
        else:
            raise NotImplementedError()
        negative_indices_list = []
        for i in range(num_predicted_positive_pairs):
            if sample_negative_strategy == "random":
                # negative_indices = sample_indices(feature_length, quantity, mask_indices[i], device=device)
                negative_indices = get_indices(feature_length, mask_indices[i], device=device)
            elif sample_negative_strategy == "neighbor":
                prev_neighbors = phoneme_interval_list.prev_neighbors(item, num_negative_neighbors)
                next_neighbors = phoneme_interval_list.next_neighbors(item, num_negative_neighbors)
                neighbors = PhonemeIntervalList(phoneme_interval_list=prev_neighbors+next_neighbors).tolist()
                replace = False if len(neighbors) >= quantity else True
                negative_indices = random_choice_for_array(torch.tensor(neighbors, device=device), quantity, replace=replace)
            else:
                raise NotImplementedError()
            negative_indices_list.append(negative_indices) # [tensor([...]), tensor([...])]
        return NegativeExample(negative_indices_list=negative_indices_list)


class ContrastiveSampler:
    @staticmethod
    def sample_from_single_batch(
        predicted_features: torch.FloatTensor,
        positive_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        sample_positive_strategy: str,
        sample_positive_quantity: str,
        sample_negative_strategy: str,
        sample_negative_quantity: str,
        feature_length: int,
        phoneme_interval_list: PhonemeIntervalList,
        cat_negative_features: Optional[bool] = None,
        # for neighbor sampler
        num_negative_neighbors: Optional[int] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> ContrastivePair:
        # feature_length = predicted_features.shape[0] # [seq_len, dim]
        predicted_indices = []
        positive_indices = []
        negative_indices = []
        for i, phoneme_interval in enumerate(phoneme_interval_list):
            # print(f"666 {i} {phoneme_interval}")
            positive_example = PositiveSampler.sample(
                sample_positive_strategy,
                sample_positive_quantity,
                phoneme_interval,
                # device=device,
            )
            predicted_indices += [pred_idx for pred_idx, pos_idx in positive_example.predicted_positive_pairs]
            positive_indices += [pos_idx for pred_idx, pos_idx in positive_example.predicted_positive_pairs]
            num_predicted_positive_pairs = len(positive_example.predicted_positive_pairs)
            negative_example = NegativeSampler.sample(
                sample_negative_strategy,
                sample_negative_quantity,
                phoneme_interval,
                num_predicted_positive_pairs,
                feature_length,
                # for neighbor
                num_negative_neighbors,
                i,
                phoneme_interval_list,
                # device=predicted_features.device,
            )
            negative_indices += negative_example.negative_indices_list # [tensor([...]), tensor([...])]
        if cat_negative_features is None:
            negative_indices_length = [_negative_indices.shape[0] for _negative_indices in negative_indices]
            if max(negative_indices_length) == min(negative_indices_length):
                cat_negative_features = True
            else:
                cat_negative_features = False
        predicted_indices = torch.tensor(predicted_indices, dtype=torch.long, device=device)
        positive_indices = torch.tensor(positive_indices, dtype=torch.long, device=device)
        if cat_negative_features == True:
            negative_indices = torch.stack(negative_indices, dim=0)
        return ContrastivePair(
            predicted_features=predicted_features,
            predicted_indices=predicted_indices,
            positive_features=positive_features,
            positive_indices=positive_indices,
            negative_features=negative_features,
            negative_indices=negative_indices,
            cat_negative_features=cat_negative_features,
            batch_size=1,
            sampled_feature_lengths=[predicted_indices.shape[0]],
        )


    @staticmethod
    def sample(
        predicted_features: torch.FloatTensor,
        positive_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        sample_positive_strategy: str,
        sample_positive_quantity: Union[str, int, float],
        sample_negative_strategy: str,
        sample_negative_quantity: Union[str, int, float],
        feature_lengths: Sequence[int],
        phoneme_interval_list_batch: Sequence[PhonemeIntervalList],
        cat_negative_features: Optional[bool] = None,
        apply_attention_mask_for_negative_indices: Optional[bool] = None,
        # for neighbor sampler
        num_negative_neighbors: Optional[int] = None,
    ) -> ContrastivePair:
        """
        predicted_features: [batch_size, seq_len, dim]
        """
        contrastive_pair_batch = []
        batch_size = predicted_features.shape[0]
        max_feature_length = predicted_features.shape[1]
        sampled_feature_lengths = []
        available_batch_idx_list = []
        for batch_idx in range(batch_size):
            # if isinstance(sample_negative_quantity, int) and sample_negative_quantity + 20 > feature_lengths[batch_idx]:
            #     continue
            available_batch_idx_list.append(batch_idx)
            contrastive_pair = ContrastiveSampler.sample_from_single_batch(
                predicted_features=predicted_features[batch_idx],
                positive_features=positive_features[batch_idx],
                negative_features=negative_features[batch_idx],
                sample_positive_strategy=sample_positive_strategy,
                sample_positive_quantity=sample_positive_quantity,
                sample_negative_strategy=sample_negative_strategy,
                sample_negative_quantity=sample_negative_quantity,
                feature_length=feature_lengths[batch_idx],
                phoneme_interval_list=phoneme_interval_list_batch[batch_idx],
                cat_negative_features=cat_negative_features,
                # for neighbor sampler
                num_negative_neighbors=num_negative_neighbors,
                device=predicted_features.device,
            )
            contrastive_pair_batch.append(contrastive_pair)
            sampled_feature_lengths += contrastive_pair.sampled_feature_lengths
        # all features
        # predicted_features = rearrange(predicted_features, "b t d -> (b t) d")
        # positive_features = rearrange(positive_features, "b t d -> (b t) d")
        # negative_features = rearrange(negative_features, "b t d -> (b t) d")
        # sampled features
        # predicted_features = torch.cat([contrastive_pair.predicted_features for contrastive_pair in contrastive_pair_batch], dim=0)
        # positive_features = torch.cat([contrastive_pair.positive_features for contrastive_pair in contrastive_pair_batch], dim=0)
        # negative_features = torch.cat([contrastive_pair.negative_features for contrastive_pair in contrastive_pair_batch], dim=0)
        # sampled features
        predicted_features = rearrange(predicted_features[available_batch_idx_list], "b t d -> (b t) d")
        positive_features = rearrange(positive_features[available_batch_idx_list], "b t d -> (b t) d")
        negative_features = rearrange(negative_features[available_batch_idx_list], "b t d -> (b t) d")
        predicted_indices = torch.cat([contrastive_pair.predicted_indices + batch_idx * max_feature_length for batch_idx, contrastive_pair in enumerate(contrastive_pair_batch)])
        positive_indices = torch.cat([contrastive_pair.positive_indices + batch_idx * max_feature_length for batch_idx, contrastive_pair in enumerate(contrastive_pair_batch)])
        if cat_negative_features == True:
            # [[tensor([...]), tensor([...])], [tensor([...]), tensor([...])]] -> tensor([[...], [...]])
            # batch_size, num_phoneme_interval, num_negative_indices -> batch_size x num_phoneme_interval, num_negative_indices
            negative_indices = torch.cat([contrastive_pair.negative_indices + batch_idx * max_feature_length for batch_idx, contrastive_pair in enumerate(contrastive_pair_batch)], dim=0)
        else:
            # negative_indices = [contrastive_pair.negative_indices + batch_idx * max_feature_length for batch_idx, contrastive_pair in enumerate(contrastive_pair_batch)]
            negative_indices = []
            for batch_idx, single_batch_contrastive_pair in enumerate(contrastive_pair_batch):
                single_batch_negative_indices = [negative_indices + batch_idx * max_feature_length for negative_indices in single_batch_contrastive_pair.negative_indices]
                negative_indices += single_batch_negative_indices
        attention_mask_for_negative_indices = None
        if apply_attention_mask_for_negative_indices:
            lens = [_negative_indices.shape[0] for _negative_indices in negative_indices]
            lens = torch.tensor(lens, dtype=torch.long, device=predicted_features.device)
            attention_mask_for_negative_indices = lengths_to_attention_mask(lens)
            if cat_negative_features != True:
                negative_indices = torch.nn.utils.rnn.pad_sequence(negative_indices, batch_first=True, padding_value=0)
                # print(666, predicted_features.shape, lens, negative_indices.shape)
        return ContrastivePair(
            ori_predicted_indices=[contrastive_pair.predicted_indices for contrastive_pair in contrastive_pair_batch],
            ori_positive_indices=[contrastive_pair.positive_indices for contrastive_pair in contrastive_pair_batch],
            ori_negative_indices=[contrastive_pair.negative_indices for contrastive_pair in contrastive_pair_batch],
            predicted_features=predicted_features,
            predicted_indices=predicted_indices,
            positive_features=positive_features,
            positive_indices=positive_indices,
            negative_features=negative_features,
            negative_indices=negative_indices,
            cat_negative_features=cat_negative_features,
            batch_size=len(contrastive_pair_batch),
            sampled_feature_lengths=sampled_feature_lengths,
            attention_mask_for_negative_indices=attention_mask_for_negative_indices,
        )


class ContrastiveEncoderForSICF(ContrastiveEncoderPreTrainedModel):
    def forward(
        self,
        encoder_features: Optional[torch.Tensor] = None,
        quantized_features: Optional[torch.Tensor] = None,
        feature_lengths: Optional[torch.Tensor] = None,
        mel2ph: Optional[Sequence[Sequence[List]]] = None,
        phoneme_interval_list_batch: Optional[Sequence[PhonemeIntervalList]] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> ContrastiveEncoderOutput:
        """
        encoder_features: [batch_size, time_step, hidden_size]
        quantized_features: [batch_size, time_step, hidden_size]
        """
        if encoder_features is not None:
            batch_size, sequence_length, hidden_size = encoder_features.shape
        elif quantized_features is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape
        prefix_features = {}
        for prefix in ("positive", "negative", "predicted"):
            prefix_type = getattr(self.config, f"{prefix}_type")
            if prefix_type == "encoder_output":
                prefix_features[f"{prefix}_features"] = encoder_features
            elif prefix_type == "quantized":
                prefix_features[f"{prefix}_features"] = quantized_features
            if getattr(self.config, f"{prefix}_project", False):
                prefix_features[f"{prefix}_features"] = self.project[prefix](prefix_features[f"{prefix}_features"])
            prefix_features[f"{prefix}_features"] = self.feature_extractor(prefix_features[f"{prefix}_features"])
        positive_features, negative_features, predicted_features = prefix_features["positive_features"], prefix_features["negative_features"], prefix_features["predicted_features"]
        if phoneme_interval_list_batch is None:
            if mel2ph is not None:
                phoneme_interval_list_batch = []
                for idx in range(batch_size):
                    phoneme_interval_list = PhonemeIntervalList(mel2ph=mel2ph[idx])
                    phoneme_interval_list = PhonemeIntervalList(
                        phoneme_interval_list=[
                            phoneme_interval for phoneme_interval in phoneme_interval_list.phoneme_interval_list if phoneme_interval.num > 1
                        ]
                    )
                    # print(f"666 len_phoneme_interval_list = {len(phoneme_interval_list)}")
                    phoneme_interval_list_batch.append(phoneme_interval_list)
            else:
                raise ValueError("one of `mel2ph` or `phoneme_interval_list_batch` should not be none!")
        contrastive_output = ContrastiveSampler.sample(
            predicted_features=predicted_features,
            positive_features=positive_features,
            negative_features=negative_features,
            sample_positive_strategy=self.config.sample_positive_strategy,
            sample_positive_quantity=self.config.sample_positive_quantity,
            sample_negative_strategy=self.config.sample_negative_strategy,
            sample_negative_quantity=self.config.sample_negative_quantity,
            feature_lengths=feature_lengths,
            phoneme_interval_list_batch=phoneme_interval_list_batch,
            cat_negative_features=self.config.cat_negative_features,
            apply_attention_mask_for_negative_indices=self.config.apply_attention_mask_for_negative_indices,
            # for neighbor sampler
            num_negative_neighbors=self.config.num_negative_neighbors,
        )
        cos_sim_logits = contrastive_output.cosine_similarity
        tot_loss = 0.0
        loss_types = self.config.loss_type.split("+")
        loss_dict = {}
        for i, loss_type in enumerate(loss_types):
            if loss_type in ("contrastive_loss", "info_nce_loss"):
                loss = contrastive_output.infonce_loss(
                    cos_sim_logits,
                    self.config.temperature,
                    self.config.info_nce_loss_reduction,
                    batch_size=positive_features.shape[0],
                )
                loss_dict[loss_type] = loss
                tot_loss += self.config.loss_weight[i] * loss
            elif loss_type == "cosine_similarity":
                loss = contrastive_output.cosine_similarity_loss(cos_sim_logits)
                if self.config.cosine_similarity_reduction == "mean":
                    tot_sampled_feature_lengths = sum(contrastive_output.sampled_feature_lengths)
                    loss = loss / tot_sampled_feature_lengths
                elif self.config.cosine_similarity_reduction == "sum":
                    loss = loss / positive_features.shape[0]
                loss_dict[loss_type] = loss
                tot_loss += self.config.loss_weight[i] * loss
        return ContrastivePair(
            phoneme_interval_list_batch=phoneme_interval_list_batch,
            loss=tot_loss,
            loss_dict=loss_dict,
            **contrastive_output
        )
