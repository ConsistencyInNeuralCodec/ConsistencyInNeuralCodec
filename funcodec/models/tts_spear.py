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

from funcodec.models.tts_valle import T2CARLM, RVQCNARPredictor

class C2CFALM(nn.Module):
    """
    code-to-code (semantic / acoustic) time-force-aligned language model
    code src --> encoder --> hidden
    hidden --> decoder --> code tgt
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_layers: List[int],
            code_src_vocab_size: int,
            code_tgt_vocab_size: int,
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
        super(C2CFALM, self).__init__()
        self.num_group = num_group
        self.code_src_vocab_size = code_src_vocab_size
        self.code_tgt_vocab_size = code_tgt_vocab_size

        if conditioning_language_id:
            self.lang_embed = nn.Embedding(len(lang_type_lst), d_model)
        else:
            self.lang_embed = None

        if conditioning_style_emb:
            self.style_embed = nn.Linear(style_emb_size, d_model)
        else:
            self.style_embed = None

        self.code_src_embed = nn.Embedding(code_src_vocab_size + 1, d_model) # 1 --> PAD
        self.code_src_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate)
        self._code_src_pad_id = code_src_vocab_size

        self.code_tgt_embed = nn.Embedding(code_tgt_vocab_size * num_group + 1 + 1, d_model) # 1 --> PAD, 1 --> BOS
        self.code_tgt_pos_enc = ScaledPositionalEncoding(d_model, positional_dropout_rate)
        self._code_tgt_pad_id = code_tgt_vocab_size * num_group
        self._code_tgt_bos_id = code_tgt_vocab_size * num_group + 1

        self.encoder = TransformerAdaptiveEncoder(
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

        self.decoder = TransformerAdaptiveEncoder(
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

        self.project_layer = nn.Linear(d_model, code_tgt_vocab_size * num_group)
        self.loss_cls = SequenceCrossEntropy(normalize_length=kwargs.get("loss_normalize_length", True), ignore_index=self._code_tgt_pad_id)

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

    def make_causal_masks(self, y_lens):
        y_max_len = y_lens.max()
        y_mask = make_pad_mask(y_lens).to(y_lens.device)
        y_padding_mask = y_mask
        y_attn_mask = torch.triu(
            torch.ones(y_max_len, y_max_len, dtype=torch.bool, device=y_lens.device),
            diagonal=1,
        )
        y_attn_mask = y_attn_mask.logical_or(y_padding_mask.unsqueeze(1))


        return y_attn_mask

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

        # encoder
        xs_max_len = xlens.max()
        xmasks = make_pad_mask(xlens).to(device)
        xs = xs[...,0].masked_fill_(xmasks, self._code_src_pad_id)
        xs_input_emb = self.code_src_embed(xs)
        xs_input_emb = self.code_src_pos_enc(xs_input_emb)
        xs_attn_mask = torch.zeros((xs_max_len, xs_max_len), dtype=torch.bool, device=device)
        xs_attn_mask = xs_attn_mask.logical_or(xmasks.unsqueeze(1))
        xs_hidden = self.encoder(xs_input_emb, None, None, xs_attn_mask)

        ymasks = make_pad_mask(ylens).to(device)
        omasks = ymasks
        ys_attn_mask = self.make_causal_masks(ylens)

        # pad y with BOS
        ys = ys[...,0].masked_fill_(ymasks, self._code_tgt_pad_id)
        ys = F.pad(ys, (1, 0), value=self._code_tgt_bos_id)
        ys_input = ys[:, :-1]
        ys_target = ys[:, 1:]

        ys_input_emb = self.code_tgt_embed(ys_input)
        if stl is not None:
            ys_input_emb[:, 0:1, :] = ys_input_emb[:, 0:1, :] + stl
        if lang is not None:
            ys_input_emb = ys_input_emb + lang
        ys_input_emb = self.code_tgt_pos_enc(ys_input_emb + xs_hidden)

        hidden = self.decoder(ys_input_emb, None, None, ys_attn_mask)
        logits = self.project_layer(hidden)
        loss = self.loss_cls(logits, ys_target, masks=omasks)

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
            xs_prefix = xs_prefix[...,0]
            xs = xs[...,0]
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
            xs_pad = xs[...,0]
            ys_pad = None
            ylens = None

        # encoder
        xs_max_len = xlens.max()
        xmasks = make_pad_mask(xlens).to(device)
        xs_input_emb = self.code_src_embed(xs_pad)
        xs_input_emb = self.code_src_pos_enc(xs_input_emb)
        xs_attn_mask = torch.zeros((xs_max_len, xs_max_len), dtype=torch.bool, device=device)
        xs_attn_mask = xs_attn_mask.logical_or(xmasks.unsqueeze(1))
        xs_hidden = self.encoder(xs_input_emb, None, None, xs_attn_mask)

        if infer_mode == "continual":
            ys_pad = F.pad(ys_pad, (1, 0), value=self._code_tgt_bos_id)

        else:
            ylens = torch.zeros_like(xlens)
            ys_pad = torch.zeros((batch_size, 1), device=device).long() + self._code_tgt_bos_id

        prompts = ys_pad

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        for _ in range(xs_hidden.size(1) - prompts.size(1) + 1):
            ys_attn_mask = self.make_causal_masks(ylens + 1)

            ys_input_emb = self.code_tgt_embed(ys_pad)
            if stl is not None:
                ys_input_emb[:, 0:1, :] = ys_input_emb[:, 0:1, :] + stl
            if lang is not None:
                ys_input_emb = ys_input_emb + lang
            ys_input_emb = self.code_tgt_pos_enc(ys_input_emb + xs_hidden[:, :ys_pad.size(1), :])

            hidden = self.decoder(ys_input_emb, None, None, ys_attn_mask)
            logits = self.project_layer(hidden[:, -1])

            samples = topk_sampling(
                logits, top_k=top_k,
                top_p=top_p, temperature=temperature
            )

            ys_pad = torch.cat([ys_pad, samples], dim=1)
            ylens = ylens + 1

        codes = ys_pad[:, 1:] # exclude BOS

        return codes


class SPEARTTS(AbsESPnetModel):
    def __init__(self,
            ling_unit_size: Dict[str, int],
            ling_unit_pad: Dict[str, int],
            d_model: int,
            nhead: int,
            num_layers: int,
            semantic_code_vocab_size: int,
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
            nar_start_stage: int = 1):
        super(SPEARTTS, self).__init__()

        self.train_mode = train_mode
        assert self.train_mode in [0, 1, 2, 3]

        if self.train_mode in [0, 1]:
            self.t2s_predictor = T2CARLM(ling_unit_size, ling_unit_pad, d_model, nhead, num_layers,
                                         semantic_code_vocab_size, 1, -1, dropout_rate, positional_dropout_rate,
                                         attention_dropout_rate, norm_before, conditioning_language_id,
                                         lang_type_lst, conditioning_style_emb, style_emb_size)

        if self.train_mode in [0, 2]:
            self.s2c_predictor = C2CFALM(d_model, nhead, [num_layers//2, num_layers//2], semantic_code_vocab_size,
                                         code_vocab_size, 1, -1,
                                         dropout_rate, positional_dropout_rate, attention_dropout_rate,
                                         norm_before, conditioning_language_id,
                                         lang_type_lst, conditioning_style_emb, style_emb_size)

        if self.train_mode in [0, 3]:
            if nar_cond_mode == 0:
                d_nar_cond = d_model
            else:
                d_nar_cond = -1
            self.rvqc_predictor = RVQCNARPredictor(ling_unit_size, ling_unit_pad, d_model, nhead,
                                                   num_layers, code_vocab_size, d_nar_cond, nar_cond_mode,
                                                   dropout_rate, num_rvq, positional_dropout_rate,
                                                   attention_dropout_rate, norm_before, conditioning_style_emb,
                                                   style_emb_size, syn_random, tied_output,
                                                   nar_stage_select_mode=nar_stage_select_mode,
                                                   start_stage=nar_start_stage)

        self.prompt_mode = prompt_mode
        assert self.prompt_mode in [0, 1]
        self.prompt_lens = prompt_lens

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

        stats = dict()
        xs = batch["text"]
        zs = batch["stk"]
        ys = batch["codec"]
        xlens = batch["text_lengths"]
        zlens = batch["stk_lengths"]
        ylens = batch["codec_lengths"]

        min_code_length = min(zs.size(1), ys.size(1))
        zs = zs[:, :min_code_length, :]
        ys = ys[:, :min_code_length, :]
        zy_lens = torch.min(zlens, ylens)

        stats["batch_size"] = xs.size(0)
        # stats["text_tot_bins"] = xs.size(0) * xs.size(1)
        stats["text_valid_bins"] = xlens.sum()
        # stats["codec_tot_bins"] = ys.size(0) * ys.size(1)
        stats["stk_valid_bins"] = zlens.sum()

        stats["codec_valid_bins"] = ylens.sum()

        if self.train_mode in [0, 1]:
            t2s_loss, t2s_stats = self.t2s_predictor(xs, zs[...,:self.t2s_predictor.num_group], xlens, zy_lens)
            for k,v in t2s_stats.items():
                stats["t2s_"+k] = v
            if self.train_mode == 1:
                s2c_loss = rvqc_loss = torch.zeros_like(t2s_loss)

        if self.train_mode in [0, 2]:
            s2c_loss, s2c_stats = self.s2c_predictor(zs, ys[...,:1], zy_lens, zy_lens)
            for k,v in s2c_stats.items():
                stats["s2c_"+k] = v
            if self.train_mode == 2:
                t2s_loss = rvqc_loss = torch.zeros_like(s2c_loss)

        if self.train_mode in [0, 3]:
            ys_prompt, ys_target, ys_prompt_lens, ys_target_lens = self._build_prompt(ys, zy_lens, self.prompt_mode, self.prompt_lens)
            rvqc_loss, rvqc_stats = self.rvqc_predictor(xs, ys_target, ys_prompt, xlens, ys_target_lens, ys_prompt_lens)
            for k,v in rvqc_stats.items():
                stats["rvqc_"+k] = v
            if self.train_mode == 3:
                t2s_loss = s2c_loss = torch.zeros_like(rvqc_loss)

        loss = t2s_loss + s2c_loss + rvqc_loss
        stats["loss"] = loss

        loss, stats, weight = force_gatherable((loss, stats, xs.size(0)), loss.device)

        return loss, stats, weight

    # TODO support batch-inference
    def inference(
            self,
            xs: torch.Tensor,
            xs_prefix: torch.Tensor,
            ys_prefix: torch.Tensor,
            zs_prefix: torch.Tensor,
            top_k: int = -100,
            top_p: float = 1.0,
            temperature: float = 1.0,
            max_steps: int = 1000,
            **kwargs,
    ) -> torch.Tensor:

        min_zy_len = min(zs_prefix.size(1), ys_prefix.size(1))
        zs_prefix = zs_prefix[:, :min_zy_len]
        ys_prefix = ys_prefix[:, :min_zy_len]

        z = self.t2s_predictor.inference(xs, xs_prefix, zs_prefix,
                                         top_k, top_p, temperature, max_steps)
        z = z[:, zs_prefix.size(1):]

        y = self.s2c_predictor.inference(z, zs_prefix, ys_prefix[...,0], top_k, top_p, temperature)
        y = y[:, ys_prefix.size(1):]

        if self.prompt_mode == 0:
            xs = torch.cat([xs_prefix, xs], dim=1)
        elif self.prompt_mode == 1:
            pass

        codes = self.rvqc_predictor.inference(xs, y.unsqueeze(-1), ys_prefix)

        return codes

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass
