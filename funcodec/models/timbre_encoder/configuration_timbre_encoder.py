import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.get_logger(__name__)


class TimbreEncoderConfig(PretrainedConfig):

    model_type = "timbre-encoder"
    is_composition = True

    def __init__(
        self,
        encoder_type: Optional[str] = None,
        input_type: Optional[str] = None,
        model_dir: Optional[str] = None,
        freeze: Optional[bool] = False,
        sample_rate: Optional[int] = None,
        embed_dim: Optional[int] = None,
        repeat_embed: Optional[int] = None,
        merge_embed: Optional[str] = "cross_attention",
        merge_with_decoder: Optional[str] = "normal",
        transformed_speech_for_codec_encoder: Optional[bool] = None,
        transformed_speech_for_timbre_encoder: Optional[bool] = None,
        load_from_existed_speaker_emebd: Optional[bool] = False,
        batch_size_is_one: Optional[bool] = False,
        dropout: Optional[float] = 0.1,
        # for FastspeechDecoder
        in_dim: Optional[int] = None,
        hidden_size: Optional[int] = 256,
        num_layers: Optional[int] = 4,
        kernel_size: Optional[int] = 9,
        num_heads: Optional[int] = 2,
        # for wav2vec2
        config_path: Optional[str] = None,
        **kwargs
    ):
        """
        params:
            encoder_type: [
                    pyannote.audio.Model,
                    speechbrain.inference.classifiers.EncoderClassifier,
                    fast_speech_transformer.FastSpeechDecoder,
                    wav2vec2
                ]
                the model used to extract speaker timbre embeddings
            input_type: [wav, mel, mag]
            merge_embed: [cross_attention, add, ada_in]
            load_from_existed_speaker_emebd: 
                load existed speaker embeddings from local files.
                not implemented
            batch_size_is_one:
                For pyannote.audio.Model, it doesn't support padding mask, so its inference batch size is only 1.
            embed_dim:
                dim of k, v in cross_attn
            hidden_size:
                dim of tiimbre_encoder
            when the timbre encoder is integrated into training process:
                1) the encoder type should be speechbrain.inference.classifiers.EncoderClassifier
                2) freeze depend on whether the timbre encoder is trainable
                3) load_from_existed_speaker_emebd = False
                4) batch_size_is_one = False
            when the timbre encoder is only used in data collating:
                1) the encoder type should be pyannote.audio.Model
                2) freeze = False
                3) load_from_existed_speaker_emebd = False
                4) batch_size_is_one = True
        """
        super().__init__(
            encoder_type=encoder_type,
            input_type=input_type,
            model_dir=model_dir,
            freeze=freeze,
            sample_rate=sample_rate,
            embed_dim=embed_dim,
            repeat_embed=repeat_embed,
            merge_embed=merge_embed,
            merge_with_decoder=merge_with_decoder,
            transformed_speech_for_codec_encoder=transformed_speech_for_codec_encoder,
            transformed_speech_for_timbre_encoder=transformed_speech_for_timbre_encoder,
            load_from_existed_speaker_emebd=load_from_existed_speaker_emebd,
            batch_size_is_one=batch_size_is_one,
            dropout=dropout,
            # for FastspeechDecoder
            in_dim=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            num_heads=num_heads,
            # for wav2vec2
            config_path=config_path,
            **kwargs
        )