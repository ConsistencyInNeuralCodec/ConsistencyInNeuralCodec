from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class PhaseaugConfig(PretrainedConfig):
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
