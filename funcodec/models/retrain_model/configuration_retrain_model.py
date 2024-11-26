import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class RetrainModelConfig(PretrainedConfig):

    model_type = "retrain-model"
    is_composition = True

    def __init__(
        self,
        all_modules: Optional[Sequence[str]] = ["encoder", "quantizer", "decoder", "discriminator"],
        freeze_modules: Optional[Sequence[str]] = [],
        retrain_modules: Optional[Sequence[str]] = [],
        **kwargs
    ):
        """
        params:
        """
        if all_modules:
            if not freeze_modules and retrain_modules:
                for module in all_modules:
                    if module not in retrain_modules:
                        freeze_modules.append(module)
            elif freeze_modules and not retrain_modules:
                for module in all_modules:
                    if module not in freeze_modules:
                        retrain_modules.append(module)
        super().__init__(
            all_modules=all_modules,
            freeze_modules=freeze_modules,
            retrain_modules=retrain_modules,
            **kwargs
        )
