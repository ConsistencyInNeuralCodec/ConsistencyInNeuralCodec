import copy
import sys, importlib
import os, time, random, math
from collections import OrderedDict
import logging, warnings
import omegaconf
from dataclasses import dataclass
import numpy as np
import torchaudio
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


class PhaseAugConfig(PretrainedConfig):
    def __init__(
        self, 
        nfft: Optional[int] = 1024, 
        hop: Optional[int] = 256, 
        use_filter: Optional[bool] = True,
        var: Optional[float] = 6.0,
        delta_max: Optional[float] = 2.0, 
        cutoff: Optional[float] = 0.05, 
        half_width: Optional[float] = 0.012, 
        kernel_size: Optional[float] = 128, 
        filter_padding: Optional[str] = "constant",
        complex_calc=None,
        freeze: Optional[bool] = True,
    ):
        super().__init__(
            nfft=nfft,
            hop=hop,
            use_filter=use_filter,
            var=var,
            delta_max=delta_max,
            cutoff=cutoff,
            half_width=half_width,
            kernel_size=kernel_size,
            filter_padding=filter_padding,
            complex_calc=complex_calc,
            freeze=freeze,
        )

    def to_dict(self):
        return {
            "nfft": self.nfft,
            "hop": self.hop,
            "use_filter": self.use_filter,
            "var": self.var,
            "delta_max": self.delta_max,
            "cutoff": self.cutoff,
            "half_width": self.half_width,
            "kernel_size": self.kernel_size,
            "filter_padding": self.filter_padding,
            "complex_calc": self.complex_calc,
            "freeze": self.freeze,
        }


class EffectAugConfig(PretrainedConfig):
    def __init__(
        self, 
        format: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            format=format,
        )

    def to_dict(self):
        return dict(
            format=self.format,
        )


class PerturbEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        input_type: Optional[str] = "wav",
        target_sr: Optional[int] = 16_000,
        perturb_methods: Optional[Sequence[str]] = ["phase_aug"],
        phase_aug_config: Optional[Union[dict, PhaseAugConfig]] = None,
        effect_aug_config: Optional[Union[dict, EffectAugConfig]] = None,
        perturb_all_speech: Optional[bool] = True,
        perturb_slice_speech: Optional[bool] = None,
        **kwargs
    ):
        if phase_aug_config is not None and isinstance(phase_aug_config, dict):
            phase_aug_config = PhaseAugConfig(**phase_aug_config)
        if effect_aug_config is not None and isinstance(effect_aug_config, dict):
            effect_aug_config = EffectAugConfig(**effect_aug_config)
        super().__init__(
            input_type=input_type,
            target_sr=target_sr,
            perturb_methods=perturb_methods,
            phase_aug_config=phase_aug_config,
            effect_aug_config=effect_aug_config,
            perturb_all_speech=perturb_all_speech,
            perturb_slice_speech=perturb_slice_speech,
        )


class PerturbEncoder(PreTrainedModel):
    config_class = PerturbEncoderConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[dict, PerturbEncoderConfig],
        **kwargs
    ):
        if isinstance(config, dict):
            config = PerturbEncoderConfig(**config)
        super().__init__(config=config)

        if config.phase_aug_config is not None:
            config_dict = config.phase_aug_config.to_dict()
            config_dict.pop("freeze", None)
            self.phase_aug = PhaseAug(**config_dict)
            # self.phase_aug = PhaseAug()
        if config.effect_aug_config is not None:
            config_dict = config.effect_aug_config.to_dict()
            self.effect_aug = torchaudio.io.AudioEffector(**config_dict)
        
    def forward_phase_aug(self, speech: torch.FloatTensor, phi: Optional[torch.FloatTensor] = None, **kwargs):
        if self.config.phase_aug_config.freeze:
            with torch.no_grad():
                return self.phase_aug(speech, phi=phi).detach()
        else:
            return self.phase_aug(speech, phi=phi)

    def forward_effect_aug(self, speech: torch.FloatTensor, **kwargs):
        device = speech.device
        speech = rearrange(speech, "b 1 t -> t b").cpu()
        # print(666, speech.shape, self.config.target_sr)
        max_num_channels = 1
        # max_num_channels = 16 # error
        speech_split_list = []
        for i in range(0, speech.shape[1], max_num_channels):
            speech_split = speech[:, i: i +  max_num_channels]
            speech_split = self.effect_aug.apply(speech_split, sample_rate=self.config.target_sr)
            speech_split_list.append(speech_split)
        speech = torch.cat(speech_split_list, dim=1)
        speech = rearrange(speech, "t b -> b 1 t").to(device)
        # print(777, speech.shape)
        return speech

    def forward(
        self,
        speech: torch.FloatTensor,
        **kwargs
    ):
        for perturb_method in self.config.perturb_methods:
            # print(666, perturb_method)
            forward_perturb_method = getattr(self, f"forward_{perturb_method}")
            speech = forward_perturb_method(speech)
        return speech
