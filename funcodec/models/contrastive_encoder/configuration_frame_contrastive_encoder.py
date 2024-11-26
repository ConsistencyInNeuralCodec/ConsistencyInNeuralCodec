import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class LossWeightPlannerConfig(PretrainedConfig):
    def __init__(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        linear_increase_per_step: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            start=start,
            weight=start,
            end=end,
            linear_increase_per_step=linear_increase_per_step,
        )


class FrameContrastiveEncoderConfig(PretrainedConfig):

    model_type = "frame-contrastive-encoder"
    is_composition = True

    def __init__(
        self,
        encoder_type: Optional[str] = "simple_frame_contrastive_encoder",
        features_type: Optional[str] = "encoder",
        perturbed_features_type: Optional[str] = "encoder",
        detach_features: Optional[Sequence[str]] = None,
        merge_phoneme_features: Optional[bool] = False,
        loss_type: Optional[str] = None,
        loss_weight: Optional[Union[float, Sequence[float]]] = 1.0,
        info_nce_loss_reduction: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        mse_loss_reduction: Optional[str] = "mean",
        cross_entropy_loss_reduction: Optional[str] = "mean",
        cross_entropy_label_smoothing: Optional[float] = 0.0,
        vocab_size: Optional[int] = None,
        feature_extractor_type: Optional[str] = None,
        feature_extractor_config: Optional[PretrainedConfig] = None,
        linear_dim_list: Optional[Sequence] = None,
        dropout: Optional[float] = 0.1,
        encoder_last_n_layer: Optional[int] = None,
        loss_weight_planner_config: Optional[LossWeightPlannerConfig] = None,
        **kwargs
    ):
        """
        params:
            encoder_type:
                simple_frame_contrastive_encoder, simclr_frame_contrastive_encoder, simsiam_frame_contrastive_encoder
            features_type:
                encoder, quantizer
            detach_features: 
                None
                normal_features
                pertubred_features
                [normal_features]
            info_nce_loss_reduction:
                none = mean,
                sum
            loss_weight:
                1.0
                [1.0, 1.0]
            feature_extractor_type:
                fc, cnn_lstm
        """
        if isinstance(detach_features, str):
            detach_features = [detach_features]
        if isinstance(loss_type, str):
            loss_type = [loss_type]
        if isinstance(loss_weight, float):
            loss_weight = [loss_weight]
        if loss_weight_planner_config is not None and isinstance(loss_weight_planner_config, dict):
            loss_weight_planner_config = LossWeightPlannerConfig(**loss_weight_planner_config)
        super().__init__(
            encoder_type=encoder_type,
            features_type=features_type,
            perturbed_features_type=perturbed_features_type,
            detach_features=detach_features,
            merge_phoneme_features=merge_phoneme_features,
            loss_type=loss_type,
            loss_weight=loss_weight,
            info_nce_loss_reduction=info_nce_loss_reduction,
            temperature=temperature,
            mse_loss_reduction=mse_loss_reduction,
            cross_entropy_loss_reduction=cross_entropy_loss_reduction,
            cross_entropy_label_smoothing=cross_entropy_label_smoothing,
            vocab_size=vocab_size,
            feature_extractor_type=feature_extractor_type,
            feature_extractor_config=feature_extractor_config,
            linear_dim_list=linear_dim_list,
            dropout=dropout,
            encoder_last_n_layer=encoder_last_n_layer,
            loss_weight_planner_config=loss_weight_planner_config,
            **kwargs
        )
