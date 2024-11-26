# Copyright 2023 Kai Hu
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Neural Codec Language Models."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import typing as tp
import math
import random
import numpy as np
from copy import deepcopy

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types
from collections import OrderedDict

from funcodec.modules.nets_utils import make_pad_mask
from funcodec.modules.embedding import ScaledPositionalEncoding
from funcodec.models.encoder.transformer_encoder import TransformerAdaptiveEncoder
from funcodec.train.abs_espnet_model import AbsESPnetModel
from funcodec.torch_utils.device_funcs import force_gatherable
from funcodec.losses.label_smoothing_loss import SequenceCrossEntropy
from funcodec.losses.label_smoothing_loss import calc_topk_accuracy
from funcodec.losses.label_smoothing_loss import LabelSmoothingLoss
from funcodec.modules.sampling_utils import topk_sampling
from funcodec.layers.mask_along_axis import MaskAlongAxisVariableMaxWidth
import logging
from funcodec.utils.hinter import hint_once

from funcodec.models.utils import lengths_to_padding_mask, lengths_to_attention_mask
from funcodec.models.dynamic.decoder import (
    MergeStrategy, apply_attention_mask_for_hidden, 
    list_mle, select_embeds, get_past_key_values_with_lens,
    modify_with_extra_mask_interval,
)
from funcodec.transformers.strategy import BackboneStrategy
from transformers.cache_utils import SinkCache # , StaticCache


def row_major_flatten(code_vocab_size, y, y_lens=None):
    if len(y.size()) == 2 or y.size(2) == 1:
        return y, y_lens

    bz = y.size(0)
    # oi = (i−1 mod Q)*N
    offsets = np.remainder(np.arange(0, y.size(1) * y.size(2)), y.size(2)) * code_vocab_size
    offsets = torch.from_numpy(offsets).long().to(y.device)
    if y_lens is not None:
        y_lens = y_lens * y.size(2)
    y = y.reshape(bz, -1, 1) + offsets.reshape(1, -1, 1)

    return y, y_lens

def row_major_reshape(code_vocab_size, y, target_shape: tuple):
    bz, seqlen, channels = target_shape
    assert y.size(0) == bz
    assert y.size(1) >= seqlen * channels
    y = y[:, :seqlen * channels]
    # oi = (i−1 mod Q)*N
    offsets = np.remainder(np.arange(0, seqlen * channels), channels) * code_vocab_size
    offsets = torch.from_numpy(offsets).long().to(y.device)

    y = y.reshape(bz, -1, 1) - offsets.reshape(1, -1, 1)
    y = y.reshape(target_shape)

    return y

class AliLinguisticEmbedding(nn.Module):
    """
    linguistic embedding with Ali's g2p tools.
    """
    def __init__(
            self,
            ling_unit_size: dict,
            ling_unit_pad: dict,
            embedding_dim: int,
    ):
        super(AliLinguisticEmbedding, self).__init__()

        nb_ling_sy = ling_unit_size["sy"]
        nb_ling_tone = ling_unit_size["tone"]
        nb_ling_syllable_flag = ling_unit_size["syllable_flag"]
        nb_ling_ws = ling_unit_size["word_segment"]

        self.sy_emb = nn.Embedding(nb_ling_sy, embedding_dim)
        self.tone_emb = nn.Embedding(nb_ling_tone, embedding_dim)
        self.syllable_flag_emb = nn.Embedding(nb_ling_syllable_flag, embedding_dim)
        self.ws_emb = nn.Embedding(nb_ling_ws, embedding_dim)

        self.d_emb = embedding_dim

        self.ling_sy_pad = ling_unit_pad["sy"]
        self.ling_tone_pad = ling_unit_pad["tone"]
        self.ling_syllable_flag_pad = ling_unit_pad["syllable_flag"]
        self.ling_ws_pad = ling_unit_pad["word_segment"]

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        inputs_sy = x[:, :, 0]
        inputs_tone = x[:, :, 1]
        inputs_syllable_flag = x[:, :, 2]
        inputs_ws = x[:, :, 3]

        if mask is not None:
            inputs_sy = inputs_sy.masked_fill_(mask, self.ling_sy_pad)
            inputs_tone = inputs_tone.masked_fill_(mask, self.ling_tone_pad)
            inputs_syllable_flag = inputs_syllable_flag.masked_fill_(mask, self.ling_syllable_flag_pad)
            inputs_ws = inputs_ws.masked_fill_(mask, self.ling_ws_pad)

        # Lookup table
        sy_embedding = self.sy_emb(inputs_sy.clone())
        tone_embedding = self.tone_emb(inputs_tone.clone())
        syllable_flag_embedding = self.syllable_flag_emb(inputs_syllable_flag.clone())
        ws_embedding = self.ws_emb(inputs_ws.clone())

        ling_embedding = sy_embedding + tone_embedding + syllable_flag_embedding + ws_embedding

        return ling_embedding


class T2CARLM(nn.Module):
    """
    text-to-code (semantic / acoustic) auto-regression language model
    text-to-coarse_acoustic
    """

    def __init__(
        self,
        ling_unit_size: Dict[str, int],
        ling_unit_pad: Dict[str, int],
        d_model: int,
        nhead: int,
        num_layers: int,
        code_vocab_size: int,
        num_group: int = 1,
        d_cond: int = -1,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        norm_before: bool = True,
        conditioning_language_id: bool = False,
        lang_type_lst: List[str] = ["ch", "en"],
        conditioning_style_emb: bool = False,
        style_emb_size: int = 128,
        prompt_lens: Optional[int] = None,
        merge_strategy: Optional[MergeStrategy] = None,
        backbone_strategy: Optional[BackboneStrategy] = None,
        text_code_embed: Optional[bool] = None,
        **kwargs,
    ):
        """
        Args:
        """
        super(T2CARLM, self).__init__()
        self.num_group = num_group
        self.code_vocab_size = code_vocab_size

        self.prompt_lens = prompt_lens
        self.merge_strategy = merge_strategy
        if merge_strategy is not None and isinstance(merge_strategy, dict):
            self.merge_strategy = MergeStrategy(**merge_strategy)
        self.backbone_strategy = backbone_strategy
        if backbone_strategy is not None:
            logging.info(f"apply backbone_strategy for text-to-acoustic")

        self.text_code_embed = text_code_embed
        if text_code_embed:
            self.special_embed = nn.Embedding(2, d_model)
            self.special_id = OrderedDict()
            self.special_id["text"] = 0
            self.special_id["code"] = 0

        if conditioning_language_id:
            self.lang_embed = nn.Embedding(len(lang_type_lst), d_model)
        else:
            self.lang_embed = None

        if conditioning_style_emb:
            self.style_embed = nn.Linear(style_emb_size, d_model)
        else:
            self.style_embed = None

        # decoder-only
        if self.backbone_type == "FunCodec":
            self.decoder = TransformerAdaptiveEncoder(
                input_size=d_model,
                cond_size=d_cond,
                output_size=d_model,
                attention_heads=nhead,
                linear_units=4 * d_model,
                num_blocks=num_layers,
                dropout_rate=dropout_rate,
                positional_dropout_rate=positional_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                input_layer=None,
                normalize_before=norm_before,
            )
            self.project_layer = nn.Linear(d_model, code_vocab_size * num_group + 1) # 1 --> EOS
        elif self.backbone_type == "transformers":
            self.decoder = self.backbone_strategy.build_model()
            # self.project_layer = self.decoder.lm_head
            self.project_layer = self.decoder.lm_head = nn.Linear(d_model, code_vocab_size * num_group + 1) # 1 --> EOS
        else:
            raise NotImplementedError
        self.loss_cls = SequenceCrossEntropy(normalize_length=kwargs.get("loss_normalize_length", True))

        if self.backbone_type == "FunCodec":
            self.text_embed = AliLinguisticEmbedding(ling_unit_size, ling_unit_pad, d_model)
            self.text_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate)

            self.code_embed = nn.Embedding(code_vocab_size * num_group + 1 + 1, d_model) # 1 --> EOS && PAD, 1 --> BOS
            self.code_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate)

            self._code_eos_id = code_vocab_size * num_group     # 1024: EOS && PAD
            self._code_bos_id = code_vocab_size * num_group + 1 # 1025: BOS
        elif self.backbone_type == "transformers":
            self.text_embed = AliLinguisticEmbedding(ling_unit_size, ling_unit_pad, d_model)
            # self.code_embed = self.decoder.get_input_embeddings()
            self.code_embed = nn.Embedding(code_vocab_size * num_group + 1 + 1, d_model)
            self.decoder.set_input_embeddings(self.code_embed)
            self._code_eos_id = self.backbone_strategy.config.eos_token_id
            self._code_bos_id = self.backbone_strategy.config.bos_token_id
        else:
            raise NotImplementedError

    @property
    def backbone_type(self):
        if self.backbone_strategy is None:
            return "FunCodec"
        return self.backbone_strategy.backbone
    
    def make_prefix_attn_mask(self, x_mask, y_mask=None):
        device = x_mask.device

        x_max_len = x_mask.size(1)
        if y_mask is None:
            y_max_len = 1 # 1 --> BOS
        else:
            # 这里为什么加1？
            y_max_len = y_mask.size(1) + 1 # 1 --> BOS

        x_attn_mask = F.pad(
            torch.zeros((x_max_len, x_max_len), dtype=torch.bool, device=device),
            (0, y_max_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_max_len, y_max_len, dtype=torch.bool, device=device),
                diagonal=1,
            ),
            (x_max_len, 0),
            value=False,
        )
        xy_attn_mask = torch.cat([x_attn_mask, y_attn_mask], dim=0)

        if y_mask is None:
            xy_padding_mask = F.pad(x_mask, (0, 1), value=False) # 1 --> BOS
        else:
            xy_padding_mask = torch.cat(
                        [x_mask, F.pad(y_mask, (1, 0), value=False)], dim=1
                    ) # 1 --> BOS

        # merge key padding and attention masks
        xy_attn_mask = xy_attn_mask.logical_or(xy_padding_mask.unsqueeze(1))

        return xy_attn_mask

    def is_stop(self, top1_sample, sample, max_len, x_len, y_len, multiplier=16.0):
        is_stop = (sample == self._code_eos_id)
        is_stop = is_stop or (y_len > max_len) or (y_len > x_len * multiplier)
        # is_stop = is_stop or (y_len > x_len and torch.argmax(logits, dim=-1)[
        #     0] == self._code_eos_id)
        is_stop = is_stop or (y_len > x_len and top1_sample == self._code_eos_id)

        return is_stop

    def forward(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        xlens: torch.Tensor,
        ylens: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_attention_mask: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        device = xs.device
        stats = {}

        # Flattening Pattern
        ys, ylens = row_major_flatten(self.code_vocab_size, ys, ylens)

        if self.style_embed is not None:
            assert "stl" in kwargs
            stl = self.style_embed(kwargs["stl"]).unsqueeze(1)
        else:
            stl = None

        if self.lang_embed is not None:
            assert "lang" in kwargs
            lang = self.lang_embed(kwargs["lang"]).unsqueeze(1)
        else:
            lang = None
        # Batch * Seq_len [False, ... , True (pad is True)]
        xmasks = make_pad_mask(xlens).to(device)
        ymasks = make_pad_mask(ylens).to(device)
        omasks = make_pad_mask(ylens + 1).to(device) # 1 --> BOS
        attn_masks = self.make_prefix_attn_mask(xmasks, ymasks)

        xs = self.text_embed(xs, xmasks)
        if self.backbone_type == "FunCodec":
            xs = self.text_pos_enc(xs)
        if self.text_code_embed:
            text_ids = torch.full_like(size=(xs.shape[0], xs.shape[1]), fill_value=self.special_id["text"], device=device)
            xs = xs + self.special_embed()
        xs_max_len = xlens.max()

        # pad y with Both BOS and EOS
        ys = ys[...,0].masked_fill_(ymasks, self._code_eos_id)
        ys = F.pad(ys, (1, 0), value=self._code_bos_id)
        ys = F.pad(ys, (0, 1), value=self._code_eos_id)
        ys_input = ys[:, :-1]
        ys_target = ys[:, 1:]

        ys_input_emb = self.code_embed(ys_input)
        if stl is not None:
            ys_input_emb[:, 0:1, :] = ys_input_emb[:, 0:1, :] + stl
        if lang is not None:
            ys_input_emb = ys_input_emb + lang
        if self.backbone_type == "FunCodec":
            ys_input_emb = self.code_pos_enc(ys_input_emb)

        xy_input_emb = torch.cat([xs, ys_input_emb], dim=1)

        if self.merge_strategy is not None:
            attn_masks = self.merge_strategy.merge_frame_window(
                attention_mask=attn_masks, xlens=xlens + 1 + self.prompt_lens,
            )

        if self.backbone_type == "FunCodec":
            decoder_output = self.decoder(xy_input_emb, None, None, attn_masks)
            hidden = decoder_output.last_hidden_state
            logits = self.project_layer(hidden[:, xs_max_len:])
        elif self.backbone_type == "transformers":
            decoder_output = self.decoder(
                inputs_embeds=xy_input_emb, 
                # attention_mask=~attn_masks.unsqueeze(1).bool(), 
                attention_mask=(~attn_masks.unsqueeze(1)).int(), 
                output_hidden_states=True, output_attentions=output_attentions,
                return_dict=True, return_logits=False,
            )
            hidden = decoder_output["hidden_states"][-1]
            # logits = decoder_output["logits"][:, xs_max_len:]
            logits = self.project_layer(hidden[:, xs_max_len:])
        loss = self.loss_cls(logits, ys_target, masks=omasks)

        if "dist" in kwargs:
            dist = kwargs["dist"]
            dist_lengths = kwargs["dist_lengths"]
            # logging.info(f"666 dist = {dist.shape} {dist[:, 0].argmax(-1)}, ys_target = {ys_target.shape} {ys_target[:, 0]} {ys_target[:, -1]}")
            # logging.info(f"666 dist = {dist.shape} dist[:, 0] = {dist[:, 0].max(-1)}, logits = {logits.shape} logits[:, 0] = {logits[:, 0].max(-1)}")
            # dist[:, 0].argmax(-1) == ys_target[:, 0]
            codebook_dim = 128
            first_k = 10
            attention_mask = lengths_to_attention_mask(dist_lengths)
            list_mle_loss = list_mle(
                y_pred=logits[:, :-1, :self.code_vocab_size], y_true=dist / codebook_dim, first_k=first_k,
                attention_mask=attention_mask, lengths=dist_lengths
            )
            stats["list_mle_loss"] = list_mle_loss.item()
            stats["pred_loss"] = loss.item()

            first_k_acc = torch.topk(dist, k=first_k, dim=-1)[1] == torch.topk(logits[:, :-1, :self.code_vocab_size], k=first_k, dim=-1)[1]
            first_k_acc = first_k_acc * attention_mask.unsqueeze(-1)
            first_k_acc = first_k_acc.sum() / first_k / dist_lengths.sum()
            stats["first_k_acc"] = first_k_acc.item()
            loss += list_mle_loss
            # loss = list_mle_loss

        topk_acc = calc_topk_accuracy(logits.detach(), ys_target, ~omasks,
                                      topk=kwargs.get("topk_lst", (10, 1)))
        stats["top10_acc"] = topk_acc[0].item()
        stats["top1_acc"] = topk_acc[1].item()
        stats["loss"] = loss.item()

        return (loss, stats)

    # TODO support batch-inference
    def inference(
        self,
        xs: torch.Tensor,
        xs_prefix: Union[torch.Tensor, None],
        ys_prefix: Union[torch.Tensor, None],
        top_k: int = -100,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_steps: int = 1000,
        topk_sampling_strategy: Optional[Dict] = None,
        fast_inference: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_attention_mask: Optional[bool] = None,
        extra_mask_interval: Optional[Sequence] = None,
        **kwargs,
    ) -> torch.Tensor:

        if fast_inference:
            return self.fast_inference(
                xs=xs,
                xs_prefix=xs_prefix,
                ys_prefix=ys_prefix,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_steps=max_steps,
                topk_sampling_strategy=topk_sampling_strategy,
                fast_inference=fast_inference,
                output_attentions=output_attentions,
                extra_mask_interval=extra_mask_interval,
                **kwargs,
            )

        device = xs.device
        batch_size = xs.size(0)
        xlens = torch.zeros((batch_size), dtype=torch.long).to(device) + xs.size(1)
        if self.style_embed is not None:
            assert "stl" in kwargs
            stl = self.style_embed(kwargs["stl"]).unsqueeze(1)
        else:
            stl = None

        if self.lang_embed is not None:
            assert "lang" in kwargs
            lang = self.lang_embed(kwargs["lang"]).unsqueeze(1)
        else:
            lang = None

        if ys_prefix is not None:
            ys_prefix, _ = row_major_flatten(self.code_vocab_size, ys_prefix)
            # print(f"666 continual, ys_prefix = {ys_prefix.shape}")

        if xs_prefix is not None:
            # continual generation
            assert ys_prefix is not None
            infer_mode = "continual"
            xlens_prefix = torch.zeros_like(xlens) + xs_prefix.size(1)
            ylens_prefix = torch.zeros_like(xlens) + ys_prefix.size(1)

            xs_pad = torch.cat([xs_prefix, xs], dim=1)
            xlens = xlens + xlens_prefix
            ys_pad = ys_prefix.squeeze(-1)
            ylens = ylens_prefix
        else:
            # generate from scratch
            infer_mode = "generate"
            xs_pad = xs
            ys_pad = None
            ylens = None

        xs_pad = self.text_embed(xs_pad)
        if self.backbone_type == "FunCodec":
            xs_pad = self.text_pos_enc(xs_pad)

        if infer_mode == "continual":
            ys_pad = F.pad(ys_pad, (1, 0), value=self._code_bos_id)
        else:
            ylens = torch.zeros_like(xlens)
            ys_pad = torch.zeros((batch_size, 1), device=device).long() + self._code_bos_id

        prompts = ys_pad

        xmasks = make_pad_mask(xlens).to(device)
        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        logits_list = []
        loop_step = 0
        orig_ylens = ylens
        decoder_outputs = []
        while True:
            steps = ys_pad.shape[1] - prompts.shape[1]
            ymasks = make_pad_mask(ylens).to(device) if infer_mode == "continual" else None
            attn_masks = self.make_prefix_attn_mask(xmasks, ymasks)

            ys_input_emb = self.code_embed(ys_pad)
            if stl is not None:
                ys_input_emb[:, 0:1, :] = ys_input_emb[:, 0:1, :] + stl
            if lang is not None:
                ys_input_emb = ys_input_emb + lang
            if self.backbone_type == "FunCodec":
                ys_input_emb = self.code_pos_enc(ys_input_emb)

            xy_input_emb = torch.cat([xs_pad, ys_input_emb], dim=1)

            # with window_size
            if loop_step == 0:
                pass
                # print(f"xy_input_emb = {xy_input_emb.shape}, attn_masks = {attn_masks.shape}")
                print(f"ar: xlens = {xlens}, ylens = {ylens}, xy_input_emb = {xy_input_emb.shape}, prefix_lens = {xlens + ylens + self.prompt_lens}")
                # print(attn_masks)
            if self.merge_strategy is not None:
                attn_masks = self.merge_strategy.merge_frame_window(
                    attention_mask=attn_masks,
                    xlens=torch.tensor([xlens + orig_ylens], dtype=torch.long, device=device),
                    training=False,
                )
                # print(f"xlens = {xlens}, orig_ylens = {orig_ylens}, prompt_lens = {self.prompt_lens}")
                # print(attn_masks[:, -1, :])
            # only for test and debug
            if extra_mask_interval is not None:
                _extra_mask_interval = deepcopy(extra_mask_interval)
                flag = False
                if _extra_mask_interval[0] < 0:
                    if loop_step < -_extra_mask_interval[1]:
                        pass
                    elif loop_step < -_extra_mask_interval[0]:
                        _extra_mask_interval[0] = 0
                        flag = True
                        prefix_len = attn_masks.shape[1]
                    elif loop_step >= -_extra_mask_interval[0]:
                        flag = True
                        prefix_len = attn_masks.shape[1]
                else:
                    prefix_len = xlens + orig_ylens
                if flag:
                    attn_masks = modify_with_extra_mask_interval(attn_masks, _extra_mask_interval, prefix_len=prefix_len, fill_value=1)
                    # print(f"666 loop_step = {loop_step}, attn_masks = {attn_masks.shape}, extra_mask_interval = {_extra_mask_interval}")

            if self.backbone_type == "FunCodec":
                decoder_output = self.decoder(xy_input_emb, None, None, attn_masks)
                hidden = decoder_output.last_hidden_state
                logits = self.project_layer(hidden[:, -1])
            elif self.backbone_type == "transformers":
                # print(f"666 dtype = {attn_masks.unsqueeze(1).dtype}, {attn_masks.unsqueeze(1).size()}")
                # print(~attn_masks.unsqueeze(1).bool())
                decoder_output = self.decoder(
                    inputs_embeds=xy_input_emb, 
                    # attention_mask=~attn_masks.unsqueeze(1).bool(), 
                    attention_mask=(~attn_masks.unsqueeze(1)).int(), 
                    output_hidden_states=True, output_attentions=output_attentions, 
                    return_dict=True, return_logits=False,
                )
                hidden = decoder_output["hidden_states"][-1]
                logits = self.project_layer(hidden[:, -1])

            # decoder_outputs.append(decoder_output)
            decoder_outputs = [decoder_output]

            if self.merge_strategy is not None:
                pass
                # hidden = apply_attention_mask_for_hidden(hidden, attn_masks[:, -1, :])

            offsets = (steps % self.num_group) * self.code_vocab_size
            sub_logits = logits[:, offsets:offsets + self.code_vocab_size]
            if (steps % self.num_group) == 0:
                sub_logits = torch.cat([sub_logits, logits[:, -1:]], dim=-1)

            top1_samples = torch.argmax(sub_logits, dim=-1)[0]

            # constrained sampling to avoid stopping generation at the beginning due to the random sampling schedule.
            if (steps % self.num_group) == 0 and steps <= self.num_group * xs.size(1):
                sub_logits = sub_logits[...,:-1]

            logits_list.append(sub_logits.detach().cpu())
            # print(f"666 loop_step = {loop_step}, sub_logits = {sub_logits.shape}")

            samples = topk_sampling(
                sub_logits, top_k=top_k,
                top_p=top_p, temperature=temperature,
                topk_sampling_strategy=topk_sampling_strategy,
            )

            if samples[0, 0] == self.code_vocab_size:
                samples[0, 0] = self.num_group * self.code_vocab_size
            else:
                samples[0, 0] = samples[0, 0] + offsets

            loop_step += 1
            if self.is_stop(top1_samples, samples[0, 0], self.num_group*max_steps, self.num_group*xs.size(1), steps):
                print(f"Text-To-Code EOS [{prompts.shape[1]} -> {ys_pad.shape[1]}], loop_step = {loop_step}")
                break

            ys_pad = torch.cat([ys_pad, samples], dim=1)
            ylens = ylens + 1

        codes = ys_pad[:, 1:] # exclude BOS
        codes = row_major_reshape(self.code_vocab_size, codes, (1, codes.size(1)//self.num_group, self.num_group))

        # return codes
        return {
            "codes": codes,
            "logits_list": logits_list,
            "decoder_outputs": decoder_output,
            "attention_mask": attn_masks if output_attention_mask else None,
        }

    def fast_inference(
        self,
        xs: torch.Tensor,
        xs_prefix: Union[torch.Tensor, None],
        ys_prefix: Union[torch.Tensor, None],
        top_k: int = -100,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_steps: int = 1000,
        topk_sampling_strategy: Optional[Dict] = None,
        output_attentions: Optional[bool] = None,
        output_attention_mask: Optional[bool] = None,
        extra_mask_interval: Optional[Sequence] = None,
        **kwargs,
    ) -> torch.Tensor:

        device = xs.device
        batch_size = xs.size(0)
        xlens = torch.zeros((batch_size), dtype=torch.long).to(device) + xs.size(1)
        if self.style_embed is not None:
            assert "stl" in kwargs
            stl = self.style_embed(kwargs["stl"]).unsqueeze(1)
        else:
            stl = None

        if self.lang_embed is not None:
            assert "lang" in kwargs
            lang = self.lang_embed(kwargs["lang"]).unsqueeze(1)
        else:
            lang = None

        if ys_prefix is not None:
            ys_prefix, _ = row_major_flatten(self.code_vocab_size, ys_prefix)
            # print(f"666 continual, ys_prefix = {ys_prefix.shape}")

        if xs_prefix is not None:
            # continual generation
            assert ys_prefix is not None
            infer_mode = "continual"
            xlens_prefix = torch.zeros_like(xlens) + xs_prefix.size(1)
            ylens_prefix = torch.zeros_like(xlens) + ys_prefix.size(1)

            xs_pad = torch.cat([xs_prefix, xs], dim=1)
            xlens = xlens + xlens_prefix
            ys_pad = ys_prefix.squeeze(-1)
            ylens = ylens_prefix
        else:
            # generate from scratch
            infer_mode = "generate"
            xs_pad = xs
            ys_pad = None
            ylens = None

        xs_pad = self.text_embed(xs_pad)
        if self.backbone_type == "FunCodec":
            xs_pad = self.text_pos_enc(xs_pad)

        if infer_mode == "continual":
            ys_pad = F.pad(ys_pad, (1, 0), value=self._code_bos_id)
        else:
            ylens = torch.zeros_like(xlens)
            ys_pad = torch.zeros((batch_size, 1), device=device).long() + self._code_bos_id

        prompts = ys_pad

        xmasks = make_pad_mask(xlens).to(device)
        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        logits_list = []
        loop_step = 0
        orig_ylens = ylens
        decoder_outputs = []
        xy_prompt_outputs = None # xs_prefix + ys_prefix + initial_tokens (prompt_lens)
        xy_lens = xlens + ylens
        past_key_values = None
        while True:
            steps = ys_pad.shape[1] - prompts.shape[1]
            ymasks = make_pad_mask(ylens).to(device) if infer_mode == "continual" else None
            attn_masks = self.make_prefix_attn_mask(xmasks, ymasks)

            ys_input_emb = self.code_embed(ys_pad)
            if stl is not None:
                ys_input_emb[:, 0:1, :] = ys_input_emb[:, 0:1, :] + stl
            if lang is not None:
                ys_input_emb = ys_input_emb + lang
            if self.backbone_type == "FunCodec":
                ys_input_emb = self.code_pos_enc(ys_input_emb)

            xy_input_emb = torch.cat([xs_pad, ys_input_emb], dim=1)

            # with window_size
            if loop_step == 0:
                pass
                # print(f"xy_input_emb = {xy_input_emb.shape}, attn_masks = {attn_masks.shape}")
                print(f"fast ar: xlens = {xlens}, ylens = {ylens}, orig_ylens = {orig_ylens}, xy_input_emb = {xy_input_emb.shape}, prefix_lens = {xlens + ylens}, attn_masks = {attn_masks.shape}")
                # print(attn_masks)
            if self.merge_strategy is not None:
                attn_masks = self.merge_strategy.merge_frame_window(
                    attention_mask=attn_masks,
                    xlens=torch.tensor([xlens + orig_ylens], dtype=torch.long, device=device),
                    training=False,
                )
                # print(f"xlens = {xlens}, orig_ylens = {orig_ylens}, prompt_lens = {self.prompt_lens}")
                # print(f"loop = {loop_step}, attn_masks = {attn_masks.shape}\n{attn_masks[:, -1, :].int()}")

            if self.backbone_type == "FunCodec":
                # not support fast_inference
                decoder_output = self.decoder(xy_input_emb, None, None, attn_masks)
                hidden = decoder_output.last_hidden_state
                logits = self.project_layer(hidden[:, -1])
            elif self.backbone_type == "transformers":
                # only support merge window
                # print(f"666 dtype = {attn_masks.unsqueeze(1).dtype}, {attn_masks.unsqueeze(1).size()}")
                # print(~attn_masks.unsqueeze(1).bool())
                if loop_step == 0:
                    decoder_output = self.decoder(
                        inputs_embeds=xy_input_emb, 
                        attention_mask=(~attn_masks.unsqueeze(1)).int(), 
                        # past_key_values=past_key_values,
                        output_hidden_states=True, output_attentions=output_attentions, 
                        return_dict=True, return_logits=False,
                    )
                    past_key_values = decoder_output.past_key_values
                else: 
                    attn_masks = attn_masks[:, -1, :]
                    # print("attn", attn_masks.shape, attn_masks.int().cpu().numpy().tolist())
                    # kv_seq_len = past_key_values[0][0].shape[-2]
                    cur_position_id = xlens + orig_ylens + loop_step
                    position_ids = torch.tensor([[cur_position_id]], dtype=torch.long, device=device)

                    if self.merge_strategy is None or loop_step < self.merge_strategy.frame_window_size:
                        pass
                    else:
                        past_key_values = get_past_key_values_with_lens(past_key_values, prefix_lens=xlens + orig_ylens, cur_lens=self.merge_strategy.frame_window_size - 1)
                        attn_masks = torch.cat([attn_masks[:, :xlens + orig_ylens], attn_masks[:, -self.merge_strategy.frame_window_size:]], dim=1)
                    # print(f"333 loop_step = {loop_step}, attn_masks = {attn_masks.shape}, position_ids = {position_ids}, kv_seq_len = {past_key_values[0][0].shape}")
                    decoder_output = self.decoder(
                        inputs_embeds=xy_input_emb[:, -1:, :], 
                        position_ids=position_ids,
                        attention_mask=(~attn_masks).int(), 
                        # attention_mask=(~attn_masks[:, None, -1:, :]).int(), 
                        use_cache=True,
                        cache_type="sink",
                        past_key_values=past_key_values,
                        output_hidden_states=True, output_attentions=output_attentions, 
                        return_dict=True, return_logits=False,
                    )
                    past_key_values = decoder_output.past_key_values
                    
                hidden = decoder_output["hidden_states"][-1]
                logits = self.project_layer(hidden[:, -1])

            # decoder_outputs.append(decoder_output)
            decoder_outputs = [decoder_output]

            if self.merge_strategy is not None:
                pass
                # hidden = apply_attention_mask_for_hidden(hidden, attn_masks[:, -1, :])

            offsets = (steps % self.num_group) * self.code_vocab_size
            sub_logits = logits[:, offsets:offsets + self.code_vocab_size]
            if (steps % self.num_group) == 0:
                sub_logits = torch.cat([sub_logits, logits[:, -1:]], dim=-1)

            top1_samples = torch.argmax(sub_logits, dim=-1)[0]

            # constrained sampling to avoid stopping generation at the beginning due to the random sampling schedule.
            if (steps % self.num_group) == 0 and steps <= self.num_group * xs.size(1):
                sub_logits = sub_logits[...,:-1]

            logits_list.append(sub_logits.detach().cpu())
            # print(f"666 loop_step = {loop_step}, sub_logits = {sub_logits.shape}")

            samples = topk_sampling(
                sub_logits, top_k=top_k,
                top_p=top_p, temperature=temperature,
                topk_sampling_strategy=topk_sampling_strategy,
            )

            if samples[0, 0] == self.code_vocab_size:
                samples[0, 0] = self.num_group * self.code_vocab_size
            else:
                samples[0, 0] = samples[0, 0] + offsets

            loop_step += 1
            if self.is_stop(top1_samples, samples[0, 0], self.num_group*max_steps, self.num_group*xs.size(1), steps):
                print(f"Text-To-Code EOS [{prompts.shape[1]} -> {ys_pad.shape[1]}], loop_step = {loop_step}")
                break

            ys_pad = torch.cat([ys_pad, samples], dim=1)
            ylens = ylens + 1

        codes = ys_pad[:, 1:] # exclude BOS
        codes = row_major_reshape(self.code_vocab_size, codes, (1, codes.size(1)//self.num_group, self.num_group))

        # return codes
        return {
            "codes": codes,
            "logits_list": logits_list,
            "decoder_outputs": decoder_output,
            "attention_mask": attn_masks if output_attention_mask else None,
        }


class RVQCNARPredictor(nn.Module):
    """
    code (semantic / acoustic) to RVQ acoustic tokens non-auto-regression predictor
    """

    def __init__(
        self,
        ling_unit_size: Dict[str, int],
        ling_pad_size: Dict[str, int],
        d_model: int,
        nhead: int,
        num_layers: int,
        code_vocab_size: int,
        d_cond: int = -1,
        cond_mode: int = 0,
        dropout_rate: float = 0.1,
        num_rvq: int = 8,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        norm_before: bool = True,
        conditioning_style_emb: bool = False,
        style_emb_size: int = 128,
        syn_random: bool = False,
        tied_output: bool = False,
        start_stage: int = 1,
        merge_strategy: Optional[MergeStrategy] = None,
        backbone_strategy: Optional[BackboneStrategy] = None,
        prompt_lens: Optional[int] = None,
        text_code_embed: Optional[bool] = None,
        nar_attention_direction: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
        """
        super(RVQCNARPredictor, self).__init__()
        self.rng = random.Random(0) if syn_random else random.Random()
        self.num_rvq = num_rvq
        self.cond_mode = cond_mode
        assert cond_mode in [0, 1] # 0: Adaptive LN 1: input embedding adding
        self.start_stage = start_stage

        self.merge_strategy = merge_strategy
        if merge_strategy is not None and isinstance(merge_strategy, dict):
            self.merge_strategy = MergeStrategy(**merge_strategy)
        self.backbone_strategy = backbone_strategy
        if backbone_strategy is not None:
            self.backbone_strategy = deepcopy(self.backbone_strategy)
            # self._code_pad_id
            self.backbone_strategy.config.pad_token_id = code_vocab_size
            if self.backbone_strategy.config.norm_type == "LayerNorm":
                self.backbone_strategy.config.norm_type = "AdaptiveLayerNorm"
            if self.cond_mode == 0:
                # pass
                self.backbone_strategy.config.condition_size = d_cond
            logging.info(f"apply backbone_strategy for non-autoregressive")
        self.prompt_lens = prompt_lens

        if nar_attention_direction is None:
            nar_attention_direction = "bidirection"
            assert nar_attention_direction in ["bidirection", "unidirection"]
        self.nar_attention_direction = nar_attention_direction
        print(f"666 nar_attention_direction = {nar_attention_direction}, self.nar_attention_direction = {self.nar_attention_direction}")

        if conditioning_style_emb:
            self.style_embed = nn.Linear(style_emb_size, d_model)
        else:
            self.style_embed = None

        if self.backbone_type == "FunCodec":
            self.text_embed = AliLinguisticEmbedding(ling_unit_size, ling_pad_size, d_model)
            # self.text_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate)
            self.text_pos_enc = ScaledPositionalEncoding(d_model, 0.0, alpha_requires_grad=False)
            self.code_embed_lst = nn.ModuleList([
                nn.Embedding(code_vocab_size + 1, d_model) for _ in range(num_rvq) # 1 --> PAD
            ])
            # self.code_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate)
            self.code_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate, alpha_requires_grad=False)
        elif self.backbone_type == "transformers":
            # get_input_embeddings()
            self.text_embed = AliLinguisticEmbedding(ling_unit_size, ling_pad_size, d_model)
            self.code_embed_lst = nn.ModuleList([
                nn.Embedding(code_vocab_size + 1, d_model) for _ in range(num_rvq) # 1 --> PAD
            ])
        else:
            raise NotImplementedError

        self._code_pad_id = code_vocab_size
        self.stage_embed = nn.Embedding(num_rvq - start_stage, d_model)

        # one pass encoder
        if self.backbone_type == "FunCodec":
            self.decoder = TransformerAdaptiveEncoder(
                input_size=d_model,
                cond_size=d_cond,
                output_size=d_model,
                attention_heads=nhead,
                linear_units=4 * d_model,
                num_blocks=num_layers,
                dropout_rate=dropout_rate,
                positional_dropout_rate=positional_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                input_layer=None,
                normalize_before=norm_before,
            )
        elif self.backbone_type == "transformers":
            self.decoder = self.backbone_strategy.build_model()
            self.decoder.set_input_embeddings(None)
        else:
            raise NotImplementedError

        if tied_output:
            self.project_layer_lst = nn.ModuleList([
                nn.Linear(d_model, code_vocab_size + 1, bias=False) for _ in range(num_rvq - start_stage)]
            ) # 1 --> PAD
            for i in range(0, num_rvq - start_stage):
                self.project_layer_lst[i].weight = self.code_embed_lst[i + start_stage].weight
        else:
            self.project_layer_lst = nn.ModuleList([
                nn.Linear(d_model, code_vocab_size) for _ in range(num_rvq - start_stage)]
            )

        self.loss_cls = SequenceCrossEntropy(
            normalize_length=kwargs.get("loss_normalize_length", True),
            ignore_index=code_vocab_size
        )

        self.stage_select_mode = kwargs.get("nar_stage_select_mode", 0)

    @property
    def backbone_type(self):
        if self.backbone_strategy is None:
            return "FunCodec"
        return self.backbone_strategy.backbone

    def make_prefix_attn_mask(self, x_mask, z_mask, y_mask):
        device = x_mask.device

        x_max_len = x_mask.size(1)
        z_max_len = z_mask.size(1)
        y_max_len = y_mask.size(1)
        tot_max_len = x_max_len + z_max_len + y_max_len

        # x_attn_mask + z_attn_mask: bi, y_attn_mask: causal
        xz_attn_mask = F.pad(
            torch.zeros((x_max_len + z_max_len, x_max_len + z_max_len), dtype=torch.bool, device=device),
            (0, y_max_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_max_len, y_max_len, dtype=torch.bool, device=device),
                diagonal=1,
            ),
            (x_max_len + z_max_len, 0),
            value=False,
        )
        xzy_attn_mask = torch.cat([xz_attn_mask, y_attn_mask], dim=0)

        # x_attn_mask: bi, z_attn_mask + y_attn_mask: causal
        # x_attn_mask = F.pad(
        #     torch.zeros((x_max_len, x_max_len), dtype=torch.bool, device=device),
        #     (0, z_max_len + y_max_len),
        #     value=True,
        # )
        # zy_attn_mask = F.pad(
        #     torch.triu(
        #         torch.ones(z_max_len + y_max_len, z_max_len + y_max_len, dtype=torch.bool, device=device),
        #         diagonal=1,
        #     ),
        #     (x_max_len, 0),
        #     value=False,
        # )
        # xzy_attn_mask = torch.cat([x_attn_mask, zy_attn_mask], dim=0)

        xzy_padding_mask = torch.cat([x_mask, z_mask, y_mask], dim=1)
        xzy_attn_mask = xzy_attn_mask.logical_or(xzy_padding_mask.unsqueeze(1))

        return xzy_attn_mask

    def make_bi_attn_mask(self, x_mask, z_mask, y_mask):
        device = x_mask.device

        x_max_len = x_mask.size(1)
        y_max_len = y_mask.size(1)
        z_max_len = z_mask.size(1)
        tot_max_len = x_max_len + z_max_len + y_max_len

        xzy_attn_mask = torch.zeros((tot_max_len, tot_max_len), dtype=torch.bool, device=device)
        xzy_padding_mask = torch.cat([x_mask, z_mask, y_mask], dim=1)
        
        # merge key padding and attention masks
        xzy_attn_mask = xzy_attn_mask.logical_or(xzy_padding_mask.unsqueeze(1))
        return xzy_attn_mask

    def select_nar_stage(self):
        num_nar_layers = self.num_rvq - self.start_stage
        nar_stage = self.rng.choices(
            [_k for _k in range(self.start_stage, self.num_rvq)],
            weights=[1.0 / num_nar_layers] * num_nar_layers,
            k=1,
        )[0]

        if self.stage_select_mode == 0:
            pass
        elif self.stage_select_mode == 1:
            raise NotImplementedError

        return nar_stage

    def forward(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        zs: torch.Tensor,
        xlens: torch.Tensor,
        ylens: torch.Tensor,
        zlens: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_attention_mask: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
          xs:
            A 3-D tensor of shape (N, S, 4).
          ys:
            A 3-D tensor of shape (N, T, num_rvq). It contains the RVQ Codec index.
          zs:
            A 3-D tensor of shape (N, T', num_rvq). It contains the acoustic prompt's RVQ Codec index.
          xlens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          ylens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          zlens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `prompt`
            before padding.
        Returns:
          Return the cross-entropy loss and Top-1/Top-10 accuracy.
        """

        device = xs.device
        stats = {}

        nar_stage = self.select_nar_stage()
        # nar_stage = torch.LongTensor(nar_stage).reshape((-1, 1)).to(device)
        nar_stage_cond = torch.zeros((xs.size(0), 1), dtype=torch.long).to(device) + nar_stage - self.start_stage
        nar_stage_cond = self.stage_embed(nar_stage_cond)
        if self.cond_mode == 0:
            nar_stage_emb = torch.zeros_like(nar_stage_cond)
        else:
            nar_stage_emb = nar_stage_cond
            nar_stage_cond = None

        if self.style_embed is not None:
            assert "stl" in kwargs
            stl = self.style_embed(kwargs["stl"]).unsqueeze(1)
        else:
            stl = None

        # text
        xmasks = make_pad_mask(xlens).to(device)
        xs_emb = self.text_embed(xs, xmasks)
        if self.backbone_type == "FunCodec":
            xs_emb = self.text_pos_enc(xs_emb)

        # code prompt
        zmasks = make_pad_mask(zlens).to(device)
        zs = zs.masked_fill_(zmasks.unsqueeze(-1), self._code_pad_id)
        zs_emb = self.code_embed_lst[0](zs[...,0].clone())
        for j in range(1, self.num_rvq):
            zs_emb = zs_emb + self.code_embed_lst[j](zs[...,j].clone())
        if stl is not None:
            zs_emb = torch.cat([stl, zs_emb], dim=1)
            zlens = zlens + 1
            zmasks = make_pad_mask(zlens).to(device)
        if self.backbone_type == "FunCodec":
            zs_emb = self.code_pos_enc(zs_emb)

        # code base
        ## pad y with EOS
        ymasks = make_pad_mask(ylens).to(device)
        ys = ys.masked_fill_(ymasks.unsqueeze(-1), self._code_pad_id)

        ys_input = ys[...,0:nar_stage]
        ys_target = ys[...,nar_stage]

        ys_input_emb = self.code_embed_lst[0](ys_input[...,0])
        for j in range(1, nar_stage):
            ys_input_emb = ys_input_emb + self.code_embed_lst[j](ys_input[...,j])
        if self.backbone_type == "FunCodec":
            ys_input_emb = self.code_pos_enc(ys_input_emb + nar_stage_emb)
        else:
            ys_input_emb = ys_input_emb + nar_stage_emb

        xzy_input_emb = torch.cat([xs_emb, zs_emb, ys_input_emb], dim=1)

        omasks = ymasks

        if self.nar_attention_direction == "bidirection":
            attn_masks = self.make_bi_attn_mask(xmasks, zmasks, ymasks)
        elif self.nar_attention_direction == "unidirection":
            attn_masks = self.make_prefix_attn_mask(xmasks, zmasks, ymasks)

        xs_max_len = xlens.max()
        zs_max_len = zlens.max()

        if self.merge_strategy is not None:
            attn_masks = self.merge_strategy.merge_frame_window(
                attention_mask=attn_masks,
                xlens=xlens + zlens,
                # xlens=xlens + zlens + self.prompt_lens,
                # window_size=int(math.ceil(self.merge_strategy.frame_window_size / 2)),
                # training=False,
            )

        if self.backbone_type == "FunCodec":
            decoder_output = self.decoder(xzy_input_emb, nar_stage_cond, None, attn_masks)
            hidden = decoder_output.last_hidden_state
        elif self.backbone_type == "transformers":
            decoder_output = self.decoder(
                inputs_embeds=xzy_input_emb,
                attention_mask=(~attn_masks.unsqueeze(1)).int(), 
                # attention_mask=~attn_masks.unsqueeze(1).bool(),
                condition=nar_stage_cond,
                output_hidden_states=True, output_attentions=output_attentions, 
                return_dict=True, return_logits=False,
            )
            # print(666, type(decoder_output), decoder_output.keys())
            hidden = decoder_output["hidden_states"][-1]
        logits = self.project_layer_lst[nar_stage - self.start_stage](hidden[:, xs_max_len + zs_max_len:, :])
        loss = self.loss_cls(logits, ys_target, masks=omasks)

        topk_acc = calc_topk_accuracy(logits.detach(), ys_target, ~omasks,
                                      topk=kwargs.get("topk_lst", (10, 1)))
        stats["top10_acc"] = topk_acc[0]
        stats["top1_acc"] = topk_acc[1]
        stats["loss"] = loss
        stats[f"{nar_stage}_top10_acc"] = topk_acc[0]
        stats[f"{nar_stage}_top1_acc"] = topk_acc[1]
        stats[f"{nar_stage}_loss"] = loss
        # stats["nar_stage"] = nar_stage

        return (loss, stats)

    # TODO support batch-inference
    def inference(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        zs: torch.Tensor,
        fast_inference: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_attention_mask: Optional[bool] = None,
        extra_mask_interval: Optional[Sequence] = None,
        **kwargs,
    ) -> torch.Tensor:

        if fast_inference:
            return self.fast_inference(
                xs=xs,
                ys=ys,
                zs=zs,
                output_attentions=output_attentions,
                **kwargs,
            )
        
        device = xs.device
        batch_size = xs.size(0)

        if self.style_embed is not None:
            assert "stl" in kwargs
            stl = self.style_embed(kwargs["stl"]).unsqueeze(1)
        else:
            stl = None

        xs_emb = self.text_embed(xs)
        if self.backbone_type == "FunCodec":
            xs_emb = self.text_pos_enc(xs_emb)
        xlens = torch.zeros(batch_size, dtype=torch.long).to(device) + xs_emb.size(1)
        x_max_len = xlens.max()

        zlens = torch.zeros(batch_size, dtype=torch.long).to(device) + zs.size(1)
        zs_emb = self.code_embed_lst[0](zs[...,0])
        for j in range(1, self.num_rvq):
            zs_emb = zs_emb + self.code_embed_lst[j](zs[...,j])
        if stl is not None:
            zs_emb = torch.cat([stl, zs_emb], dim=1)
            zlens = zlens + 1
        if self.backbone_type == "FunCodec":
            zs_emb = self.code_pos_enc(zs_emb)
        z_max_len = zlens.max()

        codes = [ys[...,i] for i in range(self.start_stage)]

        ys_emb = self.code_embed_lst[0](ys[...,0])
        for i in range(1, self.start_stage):
            ys_emb = ys_emb + self.code_embed_lst[i](ys[...,i])
        logits_list = []

        # if self.backbone_type == "transformers":
        if True:
            xmasks = make_pad_mask(xlens).to(device)
            zmasks = make_pad_mask(zlens).to(device)
            ylens = torch.zeros(batch_size, dtype=torch.long).to(device) + ys.size(1)
            ymasks = make_pad_mask(ylens).to(device)
            if self.nar_attention_direction == "bidirection":
                attn_masks = self.make_bi_attn_mask(xmasks, zmasks, ymasks)
            elif self.nar_attention_direction == "unidirection":
                attn_masks = self.make_prefix_attn_mask(xmasks, zmasks, ymasks)
                # print(666, attn_masks.shape, self.make_bi_attn_mask(xmasks, zmasks, ymasks).shape)
        if self.merge_strategy is not None:
            attn_masks = self.merge_strategy.merge_frame_window(
                attention_mask=attn_masks, 
                xlens=xlens + zlens,
                # xlens=xlens + zlens + self.prompt_lens,
                # window_size=int(math.ceil(self.merge_strategy.frame_window_size / 2)),
                training=False,
            )
            # print(f"666 window_size = {int(math.ceil(self.merge_strategy.frame_window_size / 2))}")
        
        decoder_output_list = []
        for j in range(self.start_stage, self.num_rvq):
            nar_stage_cond = torch.zeros((xs.size(0), 1), dtype=torch.long).to(device) + j - self.start_stage
            nar_stage_cond = self.stage_embed(nar_stage_cond)

            if self.cond_mode == 0:
                nar_stage_emb = torch.zeros_like(nar_stage_cond)
            else:
                nar_stage_emb = nar_stage_cond
                nar_stage_cond = None

            if self.backbone_type == "FunCodec":
                xzy_input_emb = torch.cat([xs_emb, zs_emb, self.code_pos_enc(ys_emb + nar_stage_emb)], dim=1)
            else:
                xzy_input_emb = torch.cat([xs_emb, zs_emb, ys_emb + nar_stage_emb], dim=1)

            if j == self.start_stage:
                print(f"nar: xlens = {xlens}, zlens = {zlens}, ylens = {ylens}, xzy_input_emb = {xzy_input_emb.shape}, prefix_lens = {xlens + zlens + self.prompt_lens}")

            if self.backbone_type == "FunCodec":
                decoder_output = self.decoder(xzy_input_emb, nar_stage_cond, None, None)
                hidden = decoder_output.last_hidden_state
            elif self.backbone_type == "transformers":
                decoder_output = self.decoder(
                    inputs_embeds=xzy_input_emb, 
                    attention_mask=(~attn_masks.unsqueeze(1)).int(),
                    # attention_mask=~attn_masks.unsqueeze(1).bool(),
                    condition=nar_stage_cond,
                    output_hidden_states=True, output_attentions=output_attentions, 
                    return_dict=True, return_logits=False,
                )
                hidden = decoder_output["hidden_states"][-1]
                # print("attn", attn_masks[:, -200].int().cpu().numpy().tolist())
            logits = self.project_layer_lst[j - self.start_stage](hidden[:, x_max_len + z_max_len:, :])
            logits_list.append(logits)
            samples = torch.argmax(logits[...,:-1], dim=-1) # exclude padding token
            codes.append(samples)
            decoder_output_list.append(decoder_output)

            ys_emb = ys_emb + self.code_embed_lst[j](samples)

        # return torch.stack(codes, dim=-1)
        return {
            "codes": torch.stack(codes, dim=-1),
            "logits_list": logits_list,
            "decoder_output": decoder_output_list,
            "attention_mask": attn_masks if output_attention_mask else None,
        }

    # TODO support batch-inference
    def fast_inference(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        zs: torch.Tensor,
        fast_inference: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_attention_mask: Optional[bool] = None,
        extra_mask_interval: Optional[Sequence] = None,
        **kwargs,
    ) -> torch.Tensor:

        device = xs.device
        batch_size = xs.size(0)

        if self.style_embed is not None:
            assert "stl" in kwargs
            stl = self.style_embed(kwargs["stl"]).unsqueeze(1)
        else:
            stl = None

        xs_emb = self.text_embed(xs)
        if self.backbone_type == "FunCodec":
            xs_emb = self.text_pos_enc(xs_emb)
        xlens = torch.zeros(batch_size, dtype=torch.long).to(device) + xs_emb.size(1)
        x_max_len = xlens.max()

        zlens = torch.zeros(batch_size, dtype=torch.long).to(device) + zs.size(1)
        zs_emb = self.code_embed_lst[0](zs[...,0])
        for j in range(1, self.num_rvq):
            zs_emb = zs_emb + self.code_embed_lst[j](zs[...,j])
        if stl is not None:
            zs_emb = torch.cat([stl, zs_emb], dim=1)
            zlens = zlens + 1
        if self.backbone_type == "FunCodec":
            zs_emb = self.code_pos_enc(zs_emb)
        z_max_len = zlens.max()

        codes = [ys[...,i] for i in range(self.start_stage)]

        ys_emb = self.code_embed_lst[0](ys[...,0])
        for i in range(1, self.start_stage):
            ys_emb = ys_emb + self.code_embed_lst[i](ys[...,i])
        logits_list = []

        # if self.backbone_type == "transformers":
        if True:
            xmasks = make_pad_mask(xlens).to(device)
            zmasks = make_pad_mask(zlens).to(device)
            ylens = torch.zeros(batch_size, dtype=torch.long).to(device) + ys.size(1)
            ymasks = make_pad_mask(ylens).to(device)
            if self.nar_attention_direction == "bidirection":
                attn_masks = self.make_bi_attn_mask(xmasks, zmasks, ymasks)
            elif self.nar_attention_direction == "unidirection":
                attn_masks = self.make_prefix_attn_mask(xmasks, zmasks, ymasks)
        if self.merge_strategy is not None:
            attn_masks = self.merge_strategy.merge_frame_window(
                attention_mask=attn_masks, 
                xlens=xlens + zlens,
                # xlens=xlens + zlens + self.prompt_lens,
                # window_size=int(math.ceil(self.merge_strategy.frame_window_size / 2)),
                training=False,
            )
        
        prefix_lens = (xlens + zlens).item()
        past_key_values_stages = [None for _ in range(self.num_rvq)]
        decoder_output_list = []
        # select_embeds
        for j in range(self.start_stage, self.num_rvq):
            nar_stage_cond = torch.zeros((xs.size(0), 1), dtype=torch.long).to(device) + j - self.start_stage
            nar_stage_cond = self.stage_embed(nar_stage_cond)

            if self.cond_mode == 0:
                nar_stage_emb = torch.zeros_like(nar_stage_cond)
            else:
                nar_stage_emb = nar_stage_cond
                nar_stage_cond = None

            if self.backbone_type == "FunCodec":
                xzy_input_emb = torch.cat([xs_emb, zs_emb, self.code_pos_enc(ys_emb + nar_stage_emb)], dim=1)
            else:
                xzy_input_emb = torch.cat([xs_emb, zs_emb, ys_emb + nar_stage_emb], dim=1)

            if j == self.start_stage:
                print(f"fast nar: xlens = {xlens}, zlens = {zlens}, ylens = {ylens}, xzy_input_emb = {xzy_input_emb.shape}, prefix_lens = {xlens + zlens}")

            if self.backbone_type == "FunCodec":
                decoder_output = self.decoder(xzy_input_emb, nar_stage_cond, None, None)
                hidden = decoder_output.last_hidden_state
                logits = self.project_layer_lst[j - self.start_stage](hidden[:, x_max_len + z_max_len:, :])
            elif self.backbone_type == "transformers":
                # print(f"666 xzy_input_emb = {xzy_input_emb.shape}")
                logits = None
                # for k in range(prefix_lens, xzy_input_emb.shape[1], self.prompt_lens):
                for k in range(prefix_lens, xzy_input_emb.shape[1], self.merge_strategy.frame_window_size):
                    attn_masks = attn_masks.clone()
                    end = min(k + self.merge_strategy.frame_window_size, xzy_input_emb.shape[1]) # [start, end)
                    if k == prefix_lens:
                        start = 0
                        _attn_masks = attn_masks[:, :end, :end]
                    else:
                        start = k
                        past_key_values_stages[j] = get_past_key_values_with_lens(
                            past_key_values_stages[j],
                            prefix_lens=prefix_lens,
                            cur_lens=self.merge_strategy.frame_window_size - 1,
                            # cur_position_id=prefix_lens + self.prompt_lens,
                        )
                        _attn_masks = torch.cat([attn_masks[:, start: end, :prefix_lens], attn_masks[:, start: end, start - self.merge_strategy.frame_window_size + 1: end]], dim=-1)
                    inputs_embeds = xzy_input_emb[:, start: end]
                    position_ids = torch.tensor([[m for m in range(start, end)]], dtype=torch.long, device=device)
                    # print(f"666 k = {k}, start = {start}, end = {end}, past_key_values = {None if past_key_values_stages[j] is None else past_key_values_stages[j][0][0].shape}, inputs_embeds = {inputs_embeds.shape}, _attn_masks = {_attn_masks.shape}, max_prefix = {x_max_len + z_max_len}")
                    # print(position_ids)
                    # print("attn", f"end = {end}\n", _attn_masks[:, -1].int().cpu().numpy().tolist(), "\n", attn_masks[:, end - 1].int().cpu().numpy().tolist())
                    
                    decoder_output = self.decoder(
                        inputs_embeds=inputs_embeds, 
                        attention_mask=(~_attn_masks.unsqueeze(1)).int(),
                        position_ids=position_ids,
                        condition=nar_stage_cond,
                        use_cache=True,
                        cache_type="sink",
                        past_key_values=past_key_values_stages[j],
                        output_hidden_states=True, output_attentions=output_attentions, 
                        return_dict=True, return_logits=False,
                    )
                    past_key_values_stages[j] = decoder_output.past_key_values
                    # print(f"666.666 past_key_values = {None if past_key_values_stages[j] is None else past_key_values_stages[j][0][0].shape}")
                    hidden = decoder_output["hidden_states"][-1]
                    if k == prefix_lens:
                        _logits = self.project_layer_lst[j - self.start_stage](hidden[:, prefix_lens: end, :])
                    else:
                        _logits = self.project_layer_lst[j - self.start_stage](hidden)
                    if logits is None:
                        logits = _logits
                    else:
                        logits = torch.cat([logits, _logits], dim=1)
            decoder_output_list.append(decoder_output)

            logits_list.append(logits)
            samples = torch.argmax(logits[...,:-1], dim=-1) # exclude padding token
            codes.append(samples)

            # print(f"222 stage = {j}, logits = {logits.shape}, samples = {samples.shape}, ys_emb = {ys_emb.shape}\n")
            ys_emb = ys_emb + self.code_embed_lst[j](samples)

        # return torch.stack(codes, dim=-1)
        return {
            "codes": torch.stack(codes, dim=-1),
            "logits_list": logits_list,
            "decoder_output": decoder_output_list,
            "attention_mask": attn_masks if output_attention_mask else None,
        }

    # TODO support batch-inference
    def res_decoding(
            self,
            xs: torch.Tensor,
            ys: torch.Tensor,
            zs: torch.Tensor,
            start_stage: int,
            **kwargs,
    ) -> torch.Tensor:

        assert start_stage >= self.start_stage
        device = xs.device
        batch_size = xs.size(0)

        if self.style_embed is not None:
            assert "stl" in kwargs
            stl = self.style_embed(kwargs["stl"]).unsqueeze(1)
        else:
            stl = None

        xs_emb = self.text_embed(xs)
        if self.backbone_type == "FunCodec":
            xs_emb = self.text_pos_enc(xs_emb)
        xlens = torch.zeros(batch_size, dtype=torch.long).to(device) + xs_emb.size(1)
        x_max_len = xlens.max()

        zlens = torch.zeros(batch_size, dtype=torch.long).to(device) + zs.size(1)
        zs_emb = self.code_embed_lst[0](zs[...,0])
        for j in range(1, self.num_rvq):
            zs_emb = zs_emb + self.code_embed_lst[j](zs[...,j])
        if stl is not None:
            zs_emb = torch.cat([stl, zs_emb], dim=1)
            zlens = zlens + 1
        if self.backbone_type == "FunCodec":
            zs_emb = self.code_pos_enc(zs_emb)
        z_max_len = zlens.max()

        codes = [ys[...,i] for i in range(start_stage)]
        ys_emb = self.code_embed_lst[0](ys[...,0])
        for i in range(1, start_stage):
            ys_emb = ys_emb + self.code_embed_lst[i](ys[...,i])
        for j in range(start_stage, self.num_rvq):
            nar_stage_cond = torch.zeros((xs.size(0), 1), dtype=torch.long).to(device) + j - self.start_stage
            nar_stage_cond = self.stage_embed(nar_stage_cond)

            if self.cond_mode == 0:
                nar_stage_emb = torch.zeros_like(nar_stage_cond)
            else:
                nar_stage_emb = nar_stage_cond
                nar_stage_cond = None
            if self.backbone_type == "FunCodec":
                xzy_input_emb = torch.cat([xs_emb, zs_emb, self.code_pos_enc(ys_emb + nar_stage_emb)], dim=1)
            else:
                xzy_input_emb = torch.cat([xs_emb, zs_emb, ys_emb + nar_stage_emb], dim=1)

            if self.backbone_type == "FunCodec":
                decoder_output = self.decoder(xzy_input_emb, nar_stage_cond, None, None)
                hidden = decoder_output.last_hidden_state
            elif self.backbone_type == "transformers":
                decoder_output = self.decoder(
                    inputs_embeds=xzy_input_emb, 
                    attention_mask=None,
                    condition=nar_stage_cond,
                    output_hidden_states=True, return_dict=True,
                    return_logits=False,
                )
                hidden = decoder_output["hidden_states"][-1]
            logits = self.project_layer_lst[j - self.start_stage](hidden[:, x_max_len + z_max_len:, :])
            samples = torch.argmax(logits[...,:-1], dim=-1) # exclude padding token
            codes.append(samples)

            ys_emb = ys_emb + self.code_embed_lst[j](samples)

        return torch.stack(codes, dim=-1)


class VALLE(AbsESPnetModel):
    def __init__(self,
        ling_unit_size: Dict[str, int],
        ling_unit_pad: Dict[str, int],
        d_model: int,
        nhead: int,
        num_layers: int,
        code_vocab_size: int,
        dropout_rate: float = 0.1,
        num_rvq: int = 8,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        norm_before: bool = True,
        conditioning_language_id: bool = False,
        lang_type_lst: List[str] = ["ch", "en"],
        conditioning_style_emb: bool = False,
        style_emb_size: int = 128,
        syn_random: bool = False,
        tied_output: bool = False,
        nar_cond_mode: int = 0,
        prompt_mode: int = 0,
        prompt_lens: int = 150,
        train_mode: int = 0,
        nar_stage_select_mode: int = 0,
        nar_start_stage: int = 1,
        merge_strategy: Optional[MergeStrategy] = None,
        merge_strategy_for_nar_stage: Optional[MergeStrategy] = None,
        backbone_strategy: Optional[BackboneStrategy] = None,
        text_code_embed: Optional[bool] = None,
        nar_attention_direction: Optional[str] = None,
    ):
        super(VALLE, self).__init__()

        self.train_mode = train_mode
        assert self.train_mode in [0, 1, 2]

        self.backbone_strategy = None
        if backbone_strategy is not None and isinstance(backbone_strategy, dict):
            self.backbone_strategy = BackboneStrategy(**backbone_strategy)

        self.code_vocab_size = code_vocab_size
        if self.train_mode in [0, 1]:
            self.t2c_predictor = T2CARLM(
                ling_unit_size, ling_unit_pad, d_model, nhead, num_layers,
                code_vocab_size, nar_start_stage, -1, dropout_rate, positional_dropout_rate,
                attention_dropout_rate, norm_before, conditioning_language_id,
                lang_type_lst, conditioning_style_emb, style_emb_size,
                prompt_lens=prompt_lens,
                merge_strategy=merge_strategy,
                backbone_strategy=self.backbone_strategy,
                text_code_embed=text_code_embed,
            )

        if self.train_mode in [0, 2]:
            if nar_cond_mode == 0: 
                d_nar_cond = d_model
            else:
                d_nar_cond = -1
            self.rvqc_predictor = RVQCNARPredictor(
                ling_unit_size, ling_unit_pad, d_model, nhead,
                num_layers, code_vocab_size, d_nar_cond, nar_cond_mode,
                dropout_rate, num_rvq, positional_dropout_rate,
                attention_dropout_rate, norm_before, conditioning_style_emb,
                style_emb_size, syn_random, tied_output,
                nar_stage_select_mode=nar_stage_select_mode,
                start_stage=nar_start_stage,
                merge_strategy=merge_strategy_for_nar_stage,
                # merge_strategy=merge_strategy if merge_strategy_for_nar_stage is None else merge_strategy_for_nar_stage,
                backbone_strategy=self.backbone_strategy,
                prompt_lens=prompt_lens,
                # prompt_lens=int(prompt_lens // 2),
                text_code_embed=text_code_embed,
                nar_attention_direction=nar_attention_direction,
            )

        self.prompt_mode = prompt_mode
        assert self.prompt_mode in [0, 1]
        self.prompt_lens = prompt_lens

        self.merge_strategy = merge_strategy
        # print(666, merge_strategy)
        if merge_strategy is not None and isinstance(merge_strategy, dict):
            self.merge_strategy = MergeStrategy(**merge_strategy)

    @property
    def backbone_type(self):
        if self.backbone_strategy is None:
            return "FunCodec"
        return self.backbone_strategy.backbone

    def _build_prompt(self, y, y_lens, prompt_mode, prompt_size):
        if prompt_mode == 0:
            prefix_len, prompt, prompt_lens = self._prefix_prompt(y, y_lens, prompt_size)
            ys_target = y[:, prefix_len:, :]
            ys_target_lens = y_lens - prefix_len
        elif prompt_mode == 1:
            prefix_len, prompt, prompt_lens = self._random_intra_prompt(y, y_lens, prompt_size)
            ys_target = y
            ys_target_lens = y_lens
        else:
            raise NotImplementedError

        return prompt, ys_target, prompt_lens, ys_target_lens

    def _prefix_prompt(self, y, y_lens, prompt_size):
        int_low = (0.25 * y_lens.min()).type(torch.int64).item()
        prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
        # prefix_len = min(prefix_len, prompt_size) if random.random() > 0.5 else prefix_len
        prefix_len = min(prefix_len, prompt_size)
        return prefix_len, y[:, :prefix_len, :], torch.zeros_like(y_lens) + prefix_len

    def _random_intra_prompt(self, y, y_lens, prompt_size):
        int_low = (0.25 * y_lens.min()).type(torch.int64).item()
        prefix_len = min(int_low, prompt_size)

        y_prompts_codes = []
        for b in range(y.shape[0]):
            start = self.rvqc_predictor.rng.randint(0, y_lens[b].item() - prefix_len)
            y_prompts_codes.append(
                torch.clone(y[b, start: start + prefix_len])
            )

        y_prompts_codes = torch.stack(y_prompts_codes, dim=0)

        return prefix_len, y_prompts_codes, torch.zeros_like(y_lens) + prefix_len

    def forward(
        self, **batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        # logging.info(f"666 input batch = {batch.keys()}")
        t2c_extra_inputs = {}
        dist = batch.get("dist", None)
        if dist is not None:
            batch_size, seq_len = dist.shape[0], dist.shape[1]
            dist = dist.view(batch_size, seq_len, -1, self.code_vocab_size)
            dist_lengths = batch.get("dist_lengths", None)
            t2c_extra_inputs.update(dict(dist=dist[:, :, 0, :], dist_lengths=dist_lengths))

        stats = dict()
        xs = batch["text"]
        ys = batch["codec"]
        xlens = batch["text_lengths"]
        ylens = batch["codec_lengths"]

        stats["batch_size"] = xs.size(0)
        # stats["text_tot_bins"] = xs.size(0) * xs.size(1)
        stats["text_valid_bins"] = xlens.sum()
        # stats["codec_tot_bins"] = ys.size(0) * ys.size(1)
        stats["codec_valid_bins"] = ylens.sum()

        if self.train_mode in [0, 1]:
            t2c_loss, t2c_stats = self.t2c_predictor(xs, ys[...,:self.t2c_predictor.num_group], xlens, ylens, **t2c_extra_inputs)
            for k,v in t2c_stats.items():
                stats["t2c_" + k] = v
            if self.train_mode == 1:
                rvqc_loss = torch.zeros_like(t2c_loss)

        if self.train_mode in [0, 2]:
            ys_prompt, ys_target, ys_prompt_lens, ys_target_lens = self._build_prompt(ys, ylens, self.prompt_mode, self.prompt_lens)
            rvqc_loss, rvqc_stats = self.rvqc_predictor(xs, ys_target, ys_prompt, xlens, ys_target_lens, ys_prompt_lens)
            for k,v in rvqc_stats.items():
                stats["rvqc_" + k] = v
            if self.train_mode == 2:
                t2c_loss = torch.zeros_like(rvqc_loss)

        loss = t2c_loss + rvqc_loss
        stats["loss"] = loss

        loss, stats, weight = force_gatherable((loss, stats, xs.size(0)), loss.device)

        return loss, stats, weight

    # TODO support batch-inference
    def inference(
        self,
        xs: torch.Tensor,
        xs_prefix: torch.Tensor,
        ys_prefix: torch.Tensor,
        top_k: int = -100,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_steps: int = 1000,
        topk_sampling_strategy: Optional[str] = None,
        fast_inference: Optional[bool] = None,
        # fast_inference: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        extra_mask_interval: Optional[Sequence] = None,
        **kwargs,
    ) -> torch.Tensor:

        if fast_inference is None:
            if self.backbone_type == "transformers":
                fast_inference = True
            else:
                fast_inference = False

        t2c_predictor_res = self.t2c_predictor.inference(
            xs, xs_prefix, ys_prefix[...,:self.t2c_predictor.num_group],
            top_k, top_p, temperature, max_steps, 
            topk_sampling_strategy=topk_sampling_strategy,
            fast_inference=fast_inference,
            output_attentions=output_attentions,
            # extra_mask_interval=extra_mask_interval,
            # extra_mask_interval=[-100, -50],
            # extra_mask_interval=[-175, -125],
            # extra_mask_interval=[-225, -125],
        )
        y = t2c_predictor_res["codes"]
        y = y[:, ys_prefix.size(1):]

        if self.prompt_mode == 0:
            xs = torch.cat([xs_prefix, xs], dim=1)
        elif self.prompt_mode == 1:
            pass

        rvqc_predictor_res = self.rvqc_predictor.inference(
            xs, y, ys_prefix, 
            # fast_inference=fast_inference,
            output_attentions=output_attentions,
            extra_mask_interval=extra_mask_interval,
        )
        codes = rvqc_predictor_res["codes"]

        return codes

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

