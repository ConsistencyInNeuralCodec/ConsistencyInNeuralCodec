import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class ContrastiveEncoderConfig(PretrainedConfig):

    model_type = "contrastive-encoder"
    is_composition = True

    def __init__(
        self,
        encoder_type: Optional[str] = None,
        positive_type: Optional[str] = None,
        negative_type: Optional[str] = None,
        predicted_type: Optional[str] = None,
        positive_project: Optional[bool] = None,
        positive_in_dim: Optional[int] = None,
        positive_out_dim: Optional[int] = None,
        negative_project: Optional[bool] = None,
        negative_in_dim: Optional[int] = None,
        negative_out_dim: Optional[int] = None,
        predicted_project: Optional[bool] = None,
        predicted_in_dim: Optional[int] = None,
        predicted_out_dim: Optional[int] = None,
        normalize: Optional[bool] = None,
        loss_type: Optional[str] = None,
        loss_weight: Optional[Union[float, Sequence[float]]] = 1.0,
        info_nce_loss_reduction: Optional[str] = "mean",
        cosine_similarity_reduction: Optional[str] = "mean",
        temperature: Optional[float] = 1.0,
        # transformers.Wav2Vec2
        num_negatives: Optional[int] = 100,
        mask_time_prob: Optional[float] = 0.075,
        mask_time_length: Optional[int] = 10,
        # Speaker-Independent Content Feature
        sample_positive_strategy: Optional[str] = None,
        sample_positive_quantity: Optional[str] = None,
        sample_negative_strategy: Optional[str] = None,
        sample_negative_quantity: Optional[str] = None,
        cat_negative_features: Optional[bool] = True,
        apply_attention_mask_for_negative_indices: Optional[bool] = None,
        num_negative_neighbors: Optional[str] = None,
        num_samples: Optional[int] = None,
        # feature extractor
        feature_extractor_type: Optional[str] = None,
        feature_extractor_config: Optional[PretrainedConfig] = None,
        **kwargs
    ):
        """
        params:
            encoder_type: [
                    transformers.Wav2Vec2,
                    SICF # Speaker-Independent Content Feature
                ]
            positive_type, negative_type, predicted_type: [encoder_output, quantized]
            positive_project, negative_project: project positive or negative hidden states
            loss_type: 
                cosine_similarity:
                contrastive_loss, info_nce_loss:
                cosine_similarity+info_nce_loss: 
            loss_weight:
                1.0
                [1.0, 1.0]
            sample_positive_strategy: [random, neighbor, neighbor_all]
            sample_positive_quantity:
                all (str): all frames belong to the same phoneme will be viewed as positive features
                0.8 (a floating number): 80% of frames belong to the same phoneme will be viewed as positive features
                80 (a integer number): 80 frames belong to the same phoneme will be viewed as positive features
            sample_negative_strategy: [random, neighbor]
                num_negative_neighbors: the number of negative neighbor features
            sample_negative_quantity:
                all (str): all frames not belong to this phoneme will be viewed as negative features
                0.8 (a floating number): 80% of frames not belong to this phoneme will be viewed as negative features
                80 (a integer number): 80 frames not belong to this phoneme will be viewed as negative features
            cat_negative_features:
                concatenate all sampled negative features
            when the `encoder_type` == transformers.Wav2Vec2:
                loss_type = contrastive_loss
            when the `encoder_type` == SICF:
                loss_type = cosine_similarity
        """
        if isinstance(loss_weight, float):
            loss_weight = [loss_weight]
        super().__init__(
            encoder_type=encoder_type,
            positive_type=positive_type,
            negative_type=negative_type,
            predicted_type=predicted_type,
            positive_project=positive_project,
            positive_in_dim=positive_in_dim,
            positive_out_dim=positive_out_dim,
            negative_project=negative_project,
            negative_in_dim=negative_in_dim,
            negative_out_dim=negative_out_dim,
            predicted_project=predicted_project,
            predicted_in_dim=predicted_in_dim,
            predicted_out_dim=predicted_out_dim,
            loss_type=loss_type,
            loss_weight=loss_weight,
            info_nce_loss_reduction=info_nce_loss_reduction,
            cosine_similarity_reduction=cosine_similarity_reduction,
            temperature=temperature,
            # transformers.Wav2Vec2
            num_negatives=num_negatives,
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
            # Speaker-Independent Content Feature
            sample_positive_strategy=sample_positive_strategy,
            sample_positive_quantity=sample_positive_quantity,
            sample_negative_strategy=sample_negative_strategy,
            sample_negative_quantity=sample_negative_quantity,
            cat_negative_features=cat_negative_features,
            apply_attention_mask_for_negative_indices=apply_attention_mask_for_negative_indices,
            num_negative_neighbors=num_negative_neighbors,
            num_samples=num_samples,
            # feature extractor
            feature_extractor_type=feature_extractor_type,
            feature_extractor_config=feature_extractor_config,
            **kwargs
        )

    @classmethod
    def build_default_wav2vec2(cls):
        return cls(
            encoder_type="transformers.Wav2Vec2",
            positive_type="encoder_output",
            negative_type="encoder_output",
            predicted_type="quantized",
            positive_project=False,
            negative_project=False,
            predicted_project=True,
            predicted_in_dim=128,
            predicted_out_dim=128,
            loss_type="contrastive_loss",
            loss_weight=1.0,
            num_negatives=100,
            mask_time_prob=0.075,
            mask_time_length=10,
        )
