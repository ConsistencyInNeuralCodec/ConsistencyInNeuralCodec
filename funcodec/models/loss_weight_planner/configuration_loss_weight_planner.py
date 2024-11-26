import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class LossWeightPlannerConfig(PretrainedConfig):

    model_type = "loss-weight-planner"
    is_composition = True

    def __init__(
        self,
        planner_type: Optional[str] = None,
        # auto_weight
        initial_weight: Optional[float] = 1.0,
        # step_weight
        step_decay: Optional[float] = 1.0,
        **kwargs
    ):
        """
        params:
            planner_type: [none, auto_weight, step_weight]
        """
        super().__init__(
            planner_type=planner_type,
            initial_weight=initial_weight,
            step_decay=step_decay,
            **kwargs
        )
