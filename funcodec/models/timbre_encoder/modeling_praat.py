import sys, importlib
import os, time, random
import numpy as np
import parselmouth
import logging, warnings
import omegaconf
from dataclasses import dataclass
import torch
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .configuration_praat import PraatConfig


def sampler(batch_size, ratio, device=None):
    if device is None:
        device = torch.device("cpu")
    shifts = torch.rand(batch_size, device=device) * (ratio - 1.0) + 1.0
    # flip
    flip = torch.rand(batch_size) < 0.5
    shifts[flip] = shifts[flip] ** -1
    return shifts


def praat_transform(
    snd: Union[parselmouth.Sound, np.ndarray],
    sampling_rate=16_000,
    formant_shift: float = 1.0,
    pitch_shift: float = 1.0,
    pitch_range: float = 1.0,
    pitch_steps: float = 0.01,
    pitch_floor: float = 75,
    pitch_ceil: float = 600,
    duration_factor: float = 1.0,
) -> np.ndarray:
    """Augment the sound signal with praat.
    """
    if not isinstance(snd, parselmouth.Sound):
        snd = parselmouth.Sound(snd, sampling_frequency=sampling_rate)
    pitch = parselmouth.praat.call(
        snd, 'To Pitch', pitch_steps, pitch_floor, pitch_ceil
    )
    ndpit = pitch.selected_array['frequency']
    # if all unvoiced
    nonzero = ndpit > 1e-5
    if nonzero.sum() == 0:
        return snd.values[0]
    # if voiced
    median, minp = np.median(ndpit[nonzero]).item(), ndpit[nonzero].min().item()
    # scale
    updated = median * pitch_shift
    scaled = updated + (minp * pitch_shift - updated) * pitch_range
    # for preventing infinite loop of `Change gender`
    # ref:https://github.com/praat/praat/issues/1926
    if scaled < 0.:
        pitch_range = 1.
    out, = parselmouth.praat.call(
        (snd, pitch), 'Change gender',
        formant_shift,
        median * pitch_shift,
        pitch_range,
        duration_factor).values
    return out


@dataclass
class PraatTransformOutput(ModelOutput):
    snd: Union[parselmouth.Sound, np.ndarray, torch.Tensor] = None
    sampling_rate: Optional[int] = None
    formant_shift: Optional[Union[float, torch.Tensor]] = None
    pitch_shift: Optional[Union[float, torch.Tensor]] = None
    pitch_range: Optional[Union[float, torch.Tensor]] = None
    pitch_steps: Optional[Union[float, torch.Tensor]] = None
    pitch_floor: Optional[Union[float, torch.Tensor]] = None
    pitch_ceil: Optional[Union[float, torch.Tensor]] = None
    duration_factor: Optional[Union[float, torch.Tensor]] = None


class PraatTransformer(PreTrainedModel):
    config_class = PraatConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[PraatConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = PraatConfig(**config)
        super().__init__(config=config)
    
    def forward(
        self, 
        speech: Union[np.ndarray, torch.Tensor],
        formant_shift: Optional[torch.Tensor] = None,
        pitch_shift: Optional[torch.Tensor] = None,
        pitch_range: Optional[torch.Tensor] = None,
    ) -> PraatTransformOutput:
        is_tensor = isinstance(speech, torch.Tensor)
        device = None
        if is_tensor:
            device = speech.device
            speech = speech.cpu().numpy()
        batch_size = speech.shape[0]
        # sample shifts
        if formant_shift is None:
            formant_shift = sampler(batch_size, self.config.formant_shift, device="cpu").to(device)
        if pitch_shift is None:
            pitch_shift = sampler(batch_size, self.config.pitch_shift, device="cpu").to(device)
        if pitch_range is None:
            pitch_range = sampler(batch_size, self.config.pitch_range, device="cpu").to(device)
        out = np.stack(
            [
                praat_transform(
                    snd=s, sampling_rate=self.config.sampling_rate, 
                    formant_shift=fs.item(), pitch_shift=ps.item(), pitch_range=pr.item(),
                    pitch_steps=self.config.pitch_steps,
                    pitch_floor=self.config.pitch_floor,
                    pitch_ceil=self.config.pitch_ceil,
                    duration_factor=self.config.duration_factor,
                )
                for s, fs, ps, pr in zip(
                    speech,
                    formant_shift.cpu().numpy(),
                    pitch_shift.cpu().numpy(),
                    pitch_range.cpu().numpy(),
                )
            ],
            axis=0,
        )
        if is_tensor:
            out = torch.tensor(out, dtype=torch.float32, device=device)
        return PraatTransformOutput(
            snd=out,
            sampling_rate=self.config.sampling_rate,
            formant_shift=formant_shift,
            pitch_shift=pitch_shift,
            pitch_range=pitch_range,
            pitch_steps=self.config.pitch_steps,
            pitch_floor=self.config.pitch_floor,
            pitch_ceil=self.config.pitch_ceil,
            duration_factor=self.config.duration_factor,
        )
