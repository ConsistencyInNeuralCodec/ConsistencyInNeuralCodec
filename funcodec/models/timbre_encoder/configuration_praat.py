import copy
import numpy as np
import parselmouth
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class PraatConfig(PretrainedConfig):
    """
    copy from https://github.com/revsic/torch-nansy/blob/f74746db9490052355bbf6f9110d2bb3d513ce57/utils/augment/praat.py
    """
    def __init__(
        self,
        sampling_rate: Optional[int] = 16_000,
        formant_shift: Optional[float] = 1.0,
        pitch_shift: Optional[float] = 1.0,
        pitch_range: Optional[float] = 1.0,
        pitch_steps: Optional[float] = 0.01,
        pitch_floor: Optional[float] = 75,
        pitch_ceil: Optional[float] = 600,
        duration_factor: Optional[float] = 1.0,
        reconstructed_speech_from: Optional[str] = "orig_speech",
    ):
        """
        params:
            reconstructed_speech_from:
                orig_speech, perturbed_speech
        """
        super().__init__(
            sampling_rate=sampling_rate,
            formant_shift=formant_shift,
            pitch_shift=pitch_shift,
            pitch_range=pitch_range,
            pitch_steps=pitch_steps,
            pitch_floor=pitch_floor,
            pitch_ceil=pitch_ceil,
            duration_factor=duration_factor,
            reconstructed_speech_from=reconstructed_speech_from,
        )
