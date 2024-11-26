import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .configuration_timbre_encoder import TimbreEncoderConfig
from .configuration_timbre_extractor import TimbreExtractorConfig
from .configuration_vqvc import VQVCConfig
from .configuration_qformer import QFormerConfig
from .configuration_bottleneck import BottleneckConfig
from .configuration_phaseaug import PhaseaugConfig
from .configuration_praat import PraatConfig
from ..contrastive_encoder.configuration_contrastive_encoder import ContrastiveEncoderConfig
from ..contrastive_encoder.configuration_frame_contrastive_encoder import FrameContrastiveEncoderConfig
from ..contrastive_encoder.configuration_speaker_contrastive_encoder import SpeakerContrastiveEncoderConfig
from ..contrastive_encoder.configuration_speaker_predict_encoder import SpeakerPredictEncoderConfig
from ..contrastive_encoder.configuration_phoneme_decoder import PhonemeDecoderConfig

logger = logging.get_logger(__name__)


class TimbreStrategy(PretrainedConfig):
    def __init__(
        self,
        timbre_encoder_config: Optional[Union[Dict, TimbreEncoderConfig]] = None,
        timbre_extractor_config: Optional[Union[Dict, TimbreExtractorConfig]] = None,
        vqvc_config: Optional[Union[Dict, VQVCConfig]] = None,
        qformer_config: Optional[Union[Dict, QFormerConfig]] = None,
        bottleneck_config: Optional[Union[Dict, BottleneckConfig]] = None,
        contrastive_encoder_config: Optional[Union[Dict, ContrastiveEncoderConfig]] = None,
        frame_contrastive_encoder_config: Optional[Union[Dict, FrameContrastiveEncoderConfig]] = None,
        speaker_contrastive_encoder_config: Optional[Union[Dict, SpeakerContrastiveEncoderConfig]] = None,
        speaker_predictor_with_timbre_config: Optional[Union[Dict, SpeakerPredictEncoderConfig]] = None,
        speaker_predictor_with_quant_config: Optional[Union[Dict, SpeakerPredictEncoderConfig]] = None,
        phoneme_decoder_config: Optional[Union[Dict, PhonemeDecoderConfig]] = None,
        phaseaug_config: Optional[Union[Dict, PhaseaugConfig]] = None,
        praat_config: Optional[Union[Dict, PraatConfig]] = None,
        **kwargs
    ):
        if timbre_encoder_config is not None and isinstance(timbre_encoder_config, dict):
            timbre_encoder_config = TimbreEncoderConfig(**timbre_encoder_config)
        if timbre_extractor_config is not None and isinstance(timbre_extractor_config, dict):
            timbre_extractor_config = TimbreExtractorConfig(**timbre_extractor_config)
        if vqvc_config is not None and isinstance(vqvc_config, dict):
            vqvc_config = VQVCConfig(**vqvc_config)
        if qformer_config is not None and isinstance(qformer_config, dict):
            qformer_config = QFormerConfig(**qformer_config)
        if bottleneck_config is not None and isinstance(bottleneck_config, dict):
            bottleneck_config = BottleneckConfig(**bottleneck_config)
        if contrastive_encoder_config is not None and isinstance(contrastive_encoder_config, dict):
            contrastive_encoder_config = ContrastiveEncoderConfig(**contrastive_encoder_config)
        if frame_contrastive_encoder_config is not None and isinstance(frame_contrastive_encoder_config, dict):
            frame_contrastive_encoder_config = FrameContrastiveEncoderConfig(**frame_contrastive_encoder_config)
        if speaker_contrastive_encoder_config is not None and isinstance(speaker_contrastive_encoder_config, dict):
            speaker_contrastive_encoder_config = SpeakerContrastiveEncoderConfig(**speaker_contrastive_encoder_config)
        if speaker_predictor_with_timbre_config is not None and isinstance(speaker_predictor_with_timbre_config, dict):
            speaker_predictor_with_timbre_config = SpeakerPredictEncoderConfig(**speaker_predictor_with_timbre_config)
        if speaker_predictor_with_quant_config is not None and isinstance(speaker_predictor_with_quant_config, dict):
            speaker_predictor_with_quant_config = SpeakerPredictEncoderConfig(**speaker_predictor_with_quant_config)
        if phoneme_decoder_config is not None and isinstance(phoneme_decoder_config, dict):
            phoneme_decoder_config = PhonemeDecoderConfig(**phoneme_decoder_config)
        if phaseaug_config is not None and isinstance(phaseaug_config, dict):
            phaseaug_config = PhaseaugConfig(**phaseaug_config)
        if praat_config is not None and isinstance(praat_config, dict):
            praat_config = PraatConfig(**praat_config)
        super().__init__(
            timbre_encoder_config=timbre_encoder_config,
            timbre_extractor_config=timbre_extractor_config,
            vqvc_config=vqvc_config,
            qformer_config=qformer_config,
            bottleneck_config=bottleneck_config,
            contrastive_encoder_config=contrastive_encoder_config,
            frame_contrastive_encoder_config=frame_contrastive_encoder_config,
            speaker_contrastive_encoder_config=speaker_contrastive_encoder_config,
            speaker_predictor_with_timbre_config=speaker_predictor_with_timbre_config,
            speaker_predictor_with_quant_config=speaker_predictor_with_quant_config,
            phoneme_decoder_config=phoneme_decoder_config,
            phaseaug_config=phaseaug_config,
            praat_config=praat_config,
        )
        self.merge_embed = timbre_encoder_config.merge_embed if timbre_encoder_config is not None else None
