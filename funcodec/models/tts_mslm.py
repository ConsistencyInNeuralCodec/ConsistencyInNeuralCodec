# Copyright 2023 Kai Hu
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Neural Codec Language Models."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple, List, Union
import typing as tp
import random

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
from funcodec.models.tts_valle import AliLinguisticEmbedding
import logging
from funcodec.utils.hinter import hint_once

class T2CARMSLM(nn.Module):
    """
    text-to-code (semantic / acoustic) multi-scale auto-regression language model
    """

    def __init__(
            self,
            ling_unit_size: Dict[str, int],
            ling_unit_pad: Dict[str, int],
            d_model: int,
            nhead: int,
            num_layers: int,
            code_vocab_size: int,
            num_group: int = 8,
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
        super(T2CARMSLM, self).__init__()
        self.num_group = num_group

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

        self.code_embed_lst = nn.ModuleList([nn.Embedding(code_vocab_size + 1 + 1, d_model) for _ in range(num_group) ]) # 1 --> EOS && PAD, 1 --> BOS
        self.code_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate)

        self.code_history_embed_lst = nn.ModuleList([nn.Embedding(code_vocab_size + 1 + 1, d_model) for _ in range(num_group - 1) ]) # 1 --> EOS && PAD, 1 --> BOS

        self._code_eos_id = code_vocab_size
        self._code_bos_id = code_vocab_size + 1

        # decoder-only
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

        self.project_layer_lst = nn.ModuleList([nn.Linear(d_model, code_vocab_size + 1) for _ in range(num_group)]) # 1 --> EOS
        self.loss_cls = SequenceCrossEntropy(normalize_length=kwargs.get("loss_normalize_length", True))

    def make_prefix_attn_mask(self, x_mask, y_mask):
        device = x_mask.device

        x_max_len = x_mask.size(1)
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

        xy_padding_mask = torch.cat(
                    [x_mask, F.pad(y_mask, (1, 0), value=False)], dim=1
                ) # 1 --> BOS

        # merge key padding and attention masks
        xy_attn_mask = xy_attn_mask.logical_or(xy_padding_mask.unsqueeze(1))

        return xy_attn_mask

    def is_stop(self, logits, sample, max_len, x_len, y_len, multiplier=16.0):
        is_stop = (sample == self._code_eos_id)
        is_stop = is_stop or (y_len > max_len) or (y_len > x_len * multiplier)
        is_stop = is_stop or (y_len > x_len and torch.argmax(logits, dim=-1)[
            0] == self._code_eos_id)

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
        omasks = make_pad_mask(ylens + 1).to(device) # 1 --> BOS
        attn_masks = self.make_prefix_attn_mask(xmasks, ymasks)

        xs = self.text_embed(xs, xmasks)
        xs = self.text_pos_enc(xs)
        xs_max_len = xlens.max()

        # pad y with Both BOS and EOS
        ys = ys.masked_fill_(ymasks.unsqueeze(-1), self._code_eos_id)
        ys = F.pad(ys, (0, 0, 1, 0), value=self._code_bos_id)
        ys = F.pad(ys, (0, 0, 0, 1), value=self._code_eos_id)
        ys_input = ys[:, :-1]
        ys_target = ys[:, 1:]

        ys_input_emb = self.code_embed_lst[0](ys_input[...,0].clone())
        for i in range(1, self.num_group):
            ys_input_emb = ys_input_emb + self.code_embed_lst[i](ys_input[...,i].clone())
        if stl is not None:
            ys_input_emb[:, 0:1, :] = ys_input_emb[:, 0:1, :] + stl
        if lang is not None:
            ys_input_emb = ys_input_emb + lang
        ys_input_emb = self.code_pos_enc(ys_input_emb)

        xy_input_emb = torch.cat([xs, ys_input_emb], dim=1)

        hidden = self.decoder(xy_input_emb, None, None, attn_masks)
        condition = hidden[:, xs_max_len:]
        logits_lst = [self.project_layer_lst[0](condition)]
        for i in range(1, self.num_group):
            condition = condition + self.code_history_embed_lst[i-1](ys_target[...,i-1].clone())
            logits_lst.append(self.project_layer_lst[i](condition))

        loss = self.loss_cls(logits_lst[0], ys_target[...,0], masks=omasks)
        for i in range(1, self.num_group):
            loss = loss + self.loss_cls(logits_lst[i], ys_target[...,i], masks=omasks)
        loss = loss / self.num_group

        for i in range(self.num_group):
            topk_acc = calc_topk_accuracy(logits_lst[i].detach(), ys_target[...,i], ~omasks,
                                          topk=kwargs.get("topk_lst", (10, 1)))
            stats["top10_acc_{}".format(i)] = topk_acc[0].item()
            stats["top1_acc_{}".format(i)] = topk_acc[1].item()

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

        if xs_prefix is not None:
            # continual generation
            assert ys_prefix is not None
            infer_mode = "continual"
            xlens_prefix = torch.zeros_like(xlens) + xs_prefix.size(1)
            ylens_prefix = torch.zeros_like(xlens) + ys_prefix.size(1)

            xs_pad = torch.cat([xs_prefix, xs], dim=1)
            xlens = xlens + xlens_prefix
            ys_pad = ys_prefix
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
            ymasks = make_pad_mask(ylens).to(device)
            attn_masks = self.make_prefix_attn_mask(xmasks, ymasks)

            ys_input_emb = self.code_embed_lst[0](ys_pad[...,0])
            for i in range(1, self.num_group):
                ys_input_emb = ys_input_emb + self.code_embed_lst[i](ys_pad[...,i])
            if stl is not None:
                ys_input_emb[:, 0:1, :] = ys_input_emb[:, 0:1, :] + stl
            if lang is not None:
                ys_input_emb = ys_input_emb + lang
            ys_input_emb = self.code_pos_enc(ys_input_emb)

            xy_input_emb = torch.cat([xs_pad, ys_input_emb], dim=1)

            hidden = self.decoder(xy_input_emb, None, None, attn_masks)
            condition = hidden[:, -1]
            logits = self.project_layer_lst[0](condition)

            samples_lst = []
            # constrained sampling to avoid stopping generation at the beginning due to the random sampling schedule.
            samples = topk_sampling(
                logits if (prompts.size(1) + xs.size(1)) < ys_pad.size(1) else logits[..., :-1], top_k=top_k,
                top_p=top_p, temperature=temperature
            )

            if self.is_stop(logits, samples[0, 0], max_steps, xs.size(1), ys_pad.size(1) - prompts.size(1)):
                print(f"Text-To-Code EOS [{prompts.shape[1]} -> {ys_pad.shape[1]}]")
                break

            samples_lst.append(samples)
            for i in range(1, self.num_group):
                condition = condition + self.code_history_embed_lst[i-1](samples_lst[-1].reshape(-1))
                logits = self.project_layer_lst[i](condition)
                # constrained sampling to avoid stopping generation at the beginning due to the random sampling schedule.
                samples = topk_sampling(
                    logits if (prompts.size(1) + xs.size(1)) < ys_pad.size(1) else logits[..., :-1], top_k=top_k,
                    top_p=top_p, temperature=temperature
                )
                samples_lst.append(samples)

            ys_pad = torch.cat([ys_pad, torch.stack(samples_lst, dim=-1)], dim=1)
            ylens = ylens + 1

        return ys_pad[:, 1:] # exclude BOS

# Multi-Scale Voice Generator
class MSVoiceGen(AbsESPnetModel):
    def __init__(self,
            ling_unit_size: Dict[str, int],
            ling_unit_pad: Dict[str, int],
            d_model: int,
            nhead: int,
            num_layers: int,
            code_vocab_size: int,
            dropout_rate: float = 0.1,
            num_group: int = 8,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            norm_before: bool = True,
            conditioning_language_id: bool = False,
            lang_type_lst: List[str] = ["ch", "en"],
            conditioning_style_emb: bool = False,
            style_emb_size: int = 128,
            **kwargs):
        super(MSVoiceGen, self).__init__()

        self.t2c_predictor = T2CARMSLM(ling_unit_size, ling_unit_pad, d_model, nhead, num_layers,
                                         code_vocab_size, num_group, -1, dropout_rate, positional_dropout_rate,
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
        stats["text_tot_bins"] = xs.size(0) * xs.size(1)
        stats["text_valid_bins"] = xlens.sum()
        stats["codec_tot_bins"] = ys.size(0) * ys.size(1)
        stats["codec_valid_bins"] = ylens.sum()

        t2c_loss, t2c_stats = self.t2c_predictor(xs, ys, xlens, ylens)
        for k,v in t2c_stats.items():
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


        y = self.t2c_predictor.inference(xs, xs_prefix, ys_prefix,
                                         top_k, top_p, temperature, max_steps)
        codes = y[:, ys_prefix.size(1):]

        return codes

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass
