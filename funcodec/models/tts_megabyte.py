# Copyright 2023 Kai Hu
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Neural Codec Language Models."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple, List, Union
import typing as tp
import random
import numpy as np

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types

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
from funcodec.models.tts_valle import row_major_flatten, row_major_reshape, AliLinguisticEmbedding

class T2CMegaByte(nn.Module):
    """
    text-to-code (semantic / acoustic) auto-regression language model mega-byte structure.
    """

    def __init__(
            self,
            ling_unit_size: Dict[str, int],
            ling_unit_pad: Dict[str, int],
            d_model: int,
            nhead: int,
            num_layers: List[int],
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
            **kwargs,
           ):
        """
        Args:
        """
        super(T2CMegaByte, self).__init__()
        self.num_group = num_group
        self.code_vocab_size = code_vocab_size

        if conditioning_language_id:
            self.lang_embed = nn.Embedding(len(lang_type_lst), d_model)
        else:
            self.lang_embed = None

        if conditioning_style_emb:
            self.style_embed = nn.Linear(style_emb_size, d_model)
        else:
            self.style_embed = None

        self.text_embed = AliLinguisticEmbedding(ling_unit_size, ling_unit_pad, d_model)
        self.text_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate)

        self.code_global_embed = nn.Embedding(code_vocab_size * num_group + 1 + 1, d_model // num_group) # 1 --> EOS && PAD, 1 --> BOS
        self.code_global_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate)

        self.code_local_embed = nn.Embedding(code_vocab_size * num_group + 1 + 1, d_model) # 1 --> EOS && PAD, 1 --> BOS
        self.code_local_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate)

        self._code_eos_id = code_vocab_size * num_group
        self._code_bos_id = code_vocab_size * num_group + 1

        # decoder-only
        self.global_decoder = TransformerAdaptiveEncoder(
            input_size=d_model,
            cond_size=d_cond,
            output_size=d_model,
            attention_heads=nhead,
            linear_units=4 * d_model,
            num_blocks=num_layers[0],
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=None,
            normalize_before=norm_before,
        )
        self.global_to_local_layer = nn.Linear(d_model // num_group, d_model)

        self.local_decoder = TransformerAdaptiveEncoder(
            input_size=d_model,
            cond_size=d_cond,
            output_size=d_model,
            attention_heads=nhead,
            linear_units=4 * d_model,
            num_blocks=num_layers[1],
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=None,
            normalize_before=norm_before,
        )

        self.project_layer = nn.Linear(d_model, code_vocab_size * num_group + 1) # 1 --> EOS
        self.loss_cls = SequenceCrossEntropy(normalize_length=kwargs.get("loss_normalize_length", True))

    def make_prefix_attn_mask(self, x_mask, y_mask=None):
        device = x_mask.device

        x_max_len = x_mask.size(1)
        if y_mask is None:
            y_max_len = 1 # 1 --> BOS
        else:
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
            **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        device = xs.device
        stats = {}

        B, T, _ = ys.size()

        # Flattening Pattern
        ys_flat, ylens_flat = row_major_flatten(self.code_vocab_size, ys, ylens)

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

        xmasks = make_pad_mask(xlens).to(device)
        ymasks = make_pad_mask(ylens).to(device)
        attn_masks = self.make_prefix_attn_mask(xmasks, ymasks)

        ymasks_flat = make_pad_mask(ylens_flat).to(device)
        omasks = make_pad_mask(ylens + 1).to(device) # 1 --> BOS
        omasks = omasks.reshape(-1, 1).expand(-1, self.num_group)

        xs = self.text_embed(xs, xmasks)
        xs = self.text_pos_enc(xs)
        xs_max_len = xlens.max()

        # pad y with Both BOS and EOS
        ys_global = ys_flat[...,0].masked_fill_(ymasks_flat, self._code_eos_id)
        ys_global = F.pad(ys_global, (self.num_group, 0), value=self._code_bos_id)
        ys_global = F.pad(ys_global, (0, self.num_group), value=self._code_eos_id)
        ys_global = ys_global.reshape(B, 1 + T + 1, self.num_group)
        ys_global_input = ys_global[:, :-1, :]

        ys_local = ys_global[:, 1:, :].reshape(-1, self.num_group)
        ys_local = F.pad(ys_local, (1, 0), value=self._code_bos_id)
        ys_local_input = ys_local[:, :-1]
        ys_local_target = ys_local[:, 1:]

        # global decoder
        ys_global_input_emb = self.code_global_embed(ys_global_input).reshape(B, 1 + T, -1)
        if stl is not None:
            ys_global_input_emb[:, 0:1, :] = ys_global_input_emb[:, 0:1, :] + stl
        if lang is not None:
            ys_global_input_emb = ys_global_input_emb + lang
        ys_global_input_emb = self.code_global_pos_enc(ys_global_input_emb)

        xy_input_emb = torch.cat([xs, ys_global_input_emb], dim=1)
        global_hidden = self.global_decoder(xy_input_emb, None, None, attn_masks)[:, xs_max_len:]

        # local decoder
        local_cond = self.global_to_local_layer(global_hidden.reshape(B * (1 + T), self.num_group, -1))
        ys_local_input_emb = self.code_local_embed(ys_local_input)
        ys_local_input_emb = ys_local_input_emb + local_cond
        ys_local_input_emb = self.code_local_pos_enc(ys_local_input_emb)

        local_attn_masks =  torch.triu(
                torch.ones(self.num_group, self.num_group, dtype=torch.bool, device=device),
                diagonal=1,
            ).unsqueeze(0).expand(B * (1 + T), -1, -1)

        local_hidden = self.local_decoder(ys_local_input_emb, None, None, local_attn_masks)
        logits = self.project_layer(local_hidden)
        loss = self.loss_cls(logits, ys_local_target, masks=omasks)

        for i in range(self.num_group):
            topk_acc = calc_topk_accuracy(logits[:,i,:].detach(), ys_local_target[:,i], ~omasks[:,i],
                                          topk=kwargs.get("topk_lst", (10, 1)))
            stats["top10_{}st_acc".format(i)] = topk_acc[0].item()
            stats["top1_{}st_acc".format(i)] = topk_acc[1].item()

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
            # 加1024 * i_codebook偏移量
            ys_prefix_flat, _ = row_major_flatten(self.code_vocab_size, ys_prefix)
            ys_prefix = ys_prefix_flat.reshape(batch_size, -1, self.num_group)

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
        xs_pad = self.text_pos_enc(xs_pad)

        if infer_mode == "continual":
            ys_pad = F.pad(ys_pad, (0, 0, 1, 0), value=self._code_bos_id)

        else:
            ylens = torch.zeros_like(xlens)
            ys_pad = torch.zeros((batch_size, 1, self.num_group), device=device).long() + self._code_bos_id

        prompts = ys_pad

        xmasks = make_pad_mask(xlens).to(device)
        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        while True:
            steps = ys_pad.shape[1] - prompts.shape[1]
            ymasks = make_pad_mask(ylens).to(device) if infer_mode == "continual" else None
            attn_masks = self.make_prefix_attn_mask(xmasks, ymasks)

            ys_global_input_emb = self.code_global_embed(ys_pad).reshape(batch_size, ys_pad.size(1), -1)
            if stl is not None:
                ys_global_input_emb[:, 0:1, :] = ys_global_input_emb[:, 0:1, :] + stl
            if lang is not None:
                ys_global_input_emb = ys_global_input_emb + lang
            ys_global_input_emb = self.code_global_pos_enc(ys_global_input_emb)

            xy_input_emb = torch.cat([xs_pad, ys_global_input_emb], dim=1)

            global_hidden = self.global_decoder(xy_input_emb, None, None, attn_masks)[:, -1]
            local_cond = self.global_to_local_layer(global_hidden.reshape(batch_size, self.num_group, -1))

            ys_local_pred_lst = [torch.zeros((batch_size, 1), device=device).long() + self._code_bos_id]
            for i in range(self.num_group):
                ys_local_input = torch.cat(ys_local_pred_lst, dim=1)
                ys_local_input_emb = self.code_local_embed(ys_local_input) + local_cond[:, :(i+1), :]
                ys_local_input_emb = self.code_local_pos_enc(ys_local_input_emb)

                local_attn_masks = torch.triu(
                    torch.ones(i+1, i+1, dtype=torch.bool, device=device),
                    diagonal=1,
                ).unsqueeze(0).expand(batch_size, -1, -1)

                local_hidden = self.local_decoder(ys_local_input_emb, None, None, local_attn_masks)
                logits = self.project_layer(local_hidden[:, -1])
                offsets = i * self.code_vocab_size
                sub_logits = logits[:, offsets:offsets + self.code_vocab_size]
                if i == 0:
                    sub_logits = torch.cat([sub_logits, logits[:, -1:]], dim=-1)

                top1_samples = torch.argmax(sub_logits, dim=-1)[0]

                # constrained sampling to avoid stopping generation at the beginning due to the random sampling schedule.
                if i == 0 and steps <= xs.size(1):
                    sub_logits = sub_logits[...,:-1]

                samples = topk_sampling(
                    sub_logits, top_k=top_k,
                    top_p=top_p, temperature=temperature
                )

                if samples[0, 0] == self.code_vocab_size: # EOS
                    samples[0, 0] = self.num_group * self.code_vocab_size
                else:
                    samples[0, 0] = samples[0, 0] + offsets

                if i == 0 and self.is_stop(top1_samples, samples[0, 0], max_steps, xs.size(1),
                                steps):
                    print(f"Text-To-Code EOS [{prompts.shape[1]} -> {ys_pad.shape[1]}]")
                    break

                ys_local_pred_lst.append(samples)

            if len(ys_local_pred_lst) == 1:
                break

            ys_pad = torch.cat([ys_pad, torch.stack(ys_local_pred_lst[1:], dim=-1)], dim=1)
            ylens = ylens + 1

        codes = ys_pad[:, 1:].reshape(batch_size, -1, 1) # exclude BOS
        codes = row_major_reshape(self.code_vocab_size, codes, (batch_size, codes.size(1)//self.num_group, self.num_group))

        return codes

class UniAudio(AbsESPnetModel):
    def __init__(self,
            ling_unit_size: Dict[str, int],
            ling_unit_pad: Dict[str, int],
            d_model: int,
            nhead: int,
            num_layers: List[int],
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
            train_mode: int = 0):
        super(UniAudio, self).__init__()

        self.train_mode = train_mode
        assert self.train_mode in [0, 1, 2]

        if self.train_mode in [0, 1]:
            self.t2c_predictor = T2CMegaByte(ling_unit_size, ling_unit_pad, d_model, nhead, num_layers,
                                         code_vocab_size, num_rvq, -1, dropout_rate, positional_dropout_rate,
                                         attention_dropout_rate, norm_before, conditioning_language_id,
                                         lang_type_lst, conditioning_style_emb, style_emb_size)

    def forward(
        self, **batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

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
            t2c_loss, t2c_stats = self.t2c_predictor(xs, ys[...,:self.t2c_predictor.num_group], xlens, ylens)
            for k, v in t2c_stats.items():
                stats["t2c_"+k] = v

        loss = t2c_loss
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
            **kwargs,
    ) -> torch.Tensor:


        y = self.t2c_predictor.inference(xs, xs_prefix, ys_prefix[...,:self.t2c_predictor.num_group],
                                         top_k, top_p, temperature, max_steps)
        codes = y[:, ys_prefix.size(1):]

        return codes

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

