# Copyright 2023 Zhihao Du
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-End Speech Tokenizer SoundStream."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import typing as tp
import numpy as np
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from typeguard import check_argument_types
from funcodec.train.abs_gan_espnet_model import AbsGANESPnetModel
from funcodec.torch_utils.device_funcs import force_gatherable
from librosa.filters import mel as librosa_mel_fn
from funcodec.losses.label_smoothing_loss import LabelSmoothingLoss
from funcodec.layers.mask_along_axis import MaskAlongAxisVariableMaxWidth
import logging

from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .utils import lengths_to_padding_mask, lengths_to_attention_mask
from .timbre_encoder.strategy import TimbreStrategy
from .timbre_encoder.modeling_timbre_encoder import TimbreEncoderPreTrainedModel, TimbreEncoderOutput, mean_pooling
from .timbre_encoder.modeling_timbre_extractor import BaseTimbreExtractorModel, TimbreExtractorOutput
from .timbre_encoder.modeling_qformer import QFormerModel
from .timbre_encoder import modeling_phaseaug
from .timbre_encoder.modeling_praat import PraatTransformer
from .contrastive_encoder.configuration_contrastive_encoder import ContrastiveEncoderConfig
from .contrastive_encoder.configuration_frame_contrastive_encoder import FrameContrastiveEncoderConfig
from .contrastive_encoder.modeling_contrastive_encoder import ContrastiveEncoderPreTrainedModel, ContrastiveEncoderOutput
from .contrastive_encoder.modeling_frame_contrastive_encoder import BaseFrameContrastiveEncoder, FrameContrastiveEncoderOutput
from .contrastive_encoder.modeling_speaker_contrastive_encoder import SpeakerContrastiveEncoder, SpeakerContrastiveEncoderOutput
from .contrastive_encoder.modeling_speaker_predict_encoder import SpeakerPredictEncoderBaseModel, SpeakerPredictEncoderOutput
from .contrastive_encoder.modeling_phoneme_decoder import PhonemeDecoderPreTrainedModel, PhonemeDecoderOutput
from .retrain_model.configuration_retrain_model import RetrainModelConfig
from .retrain_model.modeling_retrain_model import RetrainModelPreTrainedModel
from funcodec.modules.modeling_feature_extractor import BaseFeatureExtractor, FeatureExtractorConfig


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        device='cpu'
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audioin, return_power_spec=False):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            # torch 2.0
            return_complex=False,
        )
        power_spec = torch.sum(torch.pow(fft, 2), dim=[-1])
        mel_output = torch.matmul(self.mel_basis, power_spec)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        if return_power_spec:
            log_power_spec = torch.log10(torch.clamp(power_spec, min=1e-5))
            return log_mel_spec, log_power_spec
        return log_mel_spec


EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]


def _linear_overlap_add(frames: tp.List[torch.Tensor], stride: int):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1]
    weight = 0.5 - (t - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


def preprocess_batch(
    batch,
    hop_length: Optional[int] = 160,
    time_ds_rate: Optional[int] = 4,
):
    """
    Args:
        batch (Dict[str, Tensor]): one batch including:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            speech_lengths_2 (Tensor): Real length of the speech, tensor (B, 1).
            speech_lengths_2_lengths (Tensor)
    """
    if isinstance(batch, dict):
        batch.pop("speech_lengths_2_lengths", None)
        if "speech_lengths_2" in batch:
            batch["speech_lengths_2"] = batch["speech_lengths_2"].squeeze(-1)
            batch["speech_lengths"] = batch["speech_lengths_2"]
        if "mel2ph" in batch or "frame2ph_token_id" in batch:
            max_lens = ((batch["speech_lengths"] // hop_length + 1) / time_ds_rate).ceil().long()
            for i in range(batch["speech"].shape[0]):
                max_len = max_lens[i]
                if "mel2ph" in batch:
                    if max_len < len(batch["mel2ph"][i]):
                        batch["mel2ph"][i] = batch["mel2ph"][i][:max_len]
                if "frame2ph_token_ids" in batch:
                    if max_len < len(batch["frame2ph_token_ids"][i]):
                        batch["frame2ph_token_ids"][i] = batch["frame2ph_token_ids"][i][:max_len]
            # print(batch["speech_lengths"], max_len, [len(item) for item in batch["mel2ph"]], batch["mel2ph"])
            # batch["mel2ph"] = [item[0] for item in batch["mel2ph"]]
        if "speaker_ids" in batch:
            speaker_ids = torch.tensor(batch["speaker_ids"], dtype=torch.long, device=batch["speech"].device)
            batch["speaker_ids"] = speaker_ids
    return batch


@dataclass
class AudioTransformOutput(ModelOutput):
    pass


@dataclass
class EncodedFrameOutput(ModelOutput):
    emb: Optional[torch.Tensor] = None
    scale: Optional[Union[Tuple, torch.Tensor]] = None
    transformed_speech: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


@dataclass
class MergedTimbreOutput(ModelOutput):
    timbre_feature_attention_mask: Optional[torch.Tensor] = None
    timbre_feature_padding_mask: Optional[torch.Tensor] = None
    frame_feature_attention_mask: Optional[torch.Tensor] = None
    q_former_hidden_states: Optional[torch.Tensor] = None
    q_former_last_hidden_state: Optional[torch.Tensor] = None
    ca_output: Optional[torch.Tensor] = None
    ca_weight: Optional[torch.Tensor] = None
    timbre_feats: Optional[torch.Tensor] = None
    quant: Optional[torch.Tensor] = None
    orig_quant: Optional[torch.Tensor] = None


@dataclass
class MergedTimbreExtractorOutput(ModelOutput):
    ca_output: Optional[torch.Tensor] = None
    ca_weight: Optional[torch.Tensor] = None
    timbre_feats: Optional[torch.Tensor] = None
    quant: Optional[torch.Tensor] = None


class FreqCodec(AbsGANESPnetModel):
    """Encodec (generator + discriminator).

    This is the Encodec model
    """

    def __init__(
        self,
        input_size: int,
        odim: int = 512,
        frontend: torch.nn.Module = None,
        encoder: torch.nn.Module = None,
        quantizer: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        discriminator: Optional[torch.nn.Module] = None,
        target_sample_hz: int = 24_000,
        multi_spectral_window_powers_of_two: Union[Tuple, List] = tuple(range(5, 11)),
        multi_spectral_n_mels: int = 64,
        recon_loss_weight: float = 1.,
        multi_spectral_recon_loss_weight: float = 1.,
        adversarial_loss_weight: float = 1/9,
        feat_match_loss_weight: float = 100/9,
        enc_quant_loss_weight: float = 1.0,
        audio_normalize: bool = False,
        segment_dur: Optional[float] = 1.0,
        overlap_ratio: Optional[float] = 0.01,
        use_power_spec_loss: Optional[bool] = False,
        bypass_quantizer: bool = False,
        codec_domain: List = ("time", "time"),
        domain_conf: Optional[Dict] = {},
        phase_invariant_training: bool = False,
        # pit means phase invariant training
        pit_feat_loss_weight: float = 1,
        pit_disc_loss_weight: float = 1000,
        feat_match_layer_start: int = -1,
        timbre_strategy: Optional[Union[Dict, TimbreStrategy]] = None,
        retrain_model_config: Optional[Union[Dict, RetrainModelConfig]] = None,
        instance_norm_before_quantization: Optional[bool] = None,
        feature_extractor_before_quantization: Optional[Union[dict, FeatureExtractorConfig]] = None,
        feature_extractor_after_quantization: Optional[Union[dict, FeatureExtractorConfig]] = None,
    ):
        """Initialize SoundStream model.

        Args:
            input_size: the channel or dimension of input data
            odim: the dimension of model
            encoder: encoder
            quantizer: quantizer
            decoder: decoder
            discriminators: several discriminators, such as STFTDisc, MultiScaleDisc, MultiPeriodDisc
            discr_multi_scales: time scales of multiple discriminators
            stft_normalized: whether to normalize by magnitude after STFT, default: False.
            multi_spectral_window_powers_of_two: for multiple spectral recon loss
            multi_spectral_n_ffts: fft bins
            multi_spectral_n_mels: Mel frequency bins
            recon_loss_weight: the weight of time-domain reconstruction loss
            multi_spectral_recon_loss_weight: the weight of frequency-domain reconstruction loss
            adversarial_loss_weight: the weight of adversarial loss from discriminator
            feat_match_loss_weight: the weight of intermediate feature loss from discriminator
            cache_generator_outputs: Whether to cache generator outputs.
        """
        assert check_argument_types()
        super().__init__()

        self.odim = odim
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        # Used by task and trainer
        self.gen_model_list = [self.encoder, self.quantizer, self.decoder]
        self.discriminator = discriminator
        self.bypass_quantizer = bypass_quantizer
        self.codec_domain = codec_domain
        self.domain_conf = domain_conf
        if codec_domain[0] in ["stft", "mag_phase", "mag_angle", "mag_oracle_phase"]:
            self.enc_trans_func = torchaudio.transforms.Spectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                power=None,
            )
        elif codec_domain[0] in ["mag"]:
            self.enc_trans_func = torchaudio.transforms.Spectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                power=1,
            )
        elif codec_domain[0] == "mel":
            # self.enc_trans_func = torchaudio.transforms.MelSpectrogram(
            #     sample_rate=target_sample_hz,
            #     n_fft=domain_conf.get("n_fft", 512),
            #     hop_length=domain_conf.get("hop_length", 160),
            #     n_mels=80,
            #     power=2,
            # )
            self.enc_trans_func = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_hz,
                n_fft=domain_conf.get("n_fft", 1024),
                hop_length=domain_conf.get("hop_length", 256),
                n_mels=128,
                power=2,
            )
        if codec_domain[1] in ["stft", "mag_phase", "mag_angle", "mag_oracle_phase"]:
            self.dec_trans_func = torchaudio.transforms.InverseSpectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
            )

        # multi spectral reconstruction
        self.mel_spec_transforms = nn.ModuleList([])

        for powers in multi_spectral_window_powers_of_two:
            win_length = 2 ** powers

            melspec_transform = Audio2Mel(
                sampling_rate=target_sample_hz,
                win_length=win_length,
                hop_length=win_length // 4,
                n_mel_channels=multi_spectral_n_mels
            )

            self.mel_spec_transforms.append(melspec_transform)

        # loss weights
        self.recon_loss_weight = recon_loss_weight
        self.multi_spectral_recon_loss_weight = multi_spectral_recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feat_match_loss_weight = feat_match_loss_weight
        self.enc_quant_loss_weight = enc_quant_loss_weight
        self.register_buffer('zero', torch.tensor([0.]), persistent=False)
        self.gen_loss = 0
        self.audio_normalize = audio_normalize
        self.segment_dur = segment_dur
        self.overlap_ratio = overlap_ratio
        self.sample_rate = target_sample_hz
        self.forward_step = 0
        self.use_power_spec_loss = use_power_spec_loss

        self.feat_match_layer_start = feat_match_layer_start
        self.phaseaug = None
        self.phase_invariant_training = phase_invariant_training
        self.pit_feat_loss_weight = pit_feat_loss_weight
        self.pit_disc_loss_weight = pit_disc_loss_weight
        if phase_invariant_training:
            from phaseaug.phaseaug import PhaseAug
            self.phaseaug = PhaseAug(nfft=512, hop=160)

        # for timbre disentangle
        self.target_sample_hz = target_sample_hz
        self.init_timbre_encoder(timbre_strategy)
        self.time_ds_rate = np.prod([_time for _complex, _time in self.encoder.ratios])
        # from speech_lengths get frame_lengths:
        # frame_lengths = ((speech_lengths // self.domain_conf.get("hop_length", 160) + 1) / self.time_ds_rate).ceil().long()

        self.prepare_retrain(retrain_model_config)

        self.instance_norm_before_quantization = instance_norm_before_quantization
        if self.instance_norm_before_quantization or \
            (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization) or \
            (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_after_encoder):
            self._instance_norm_1d = nn.InstanceNorm1d(odim, affine=False)
            self.gen_model_list.append(self._instance_norm_1d)
            self.instance_norm_1d = lambda x: self._instance_norm_1d(x.transpose(1, 2)).transpose(1, 2)

        self.feature_extractor_before_quantization = None
        self.feature_extractor_after_quantization = None
        if feature_extractor_before_quantization is not None and isinstance(feature_extractor_before_quantization, dict):
            feature_extractor_before_quantization = FeatureExtractorConfig(**feature_extractor_before_quantization)
            self.feature_extractor_before_quantization = BaseFeatureExtractor.build_model(config=feature_extractor_before_quantization)
            self.gen_model_list.append(self.feature_extractor_before_quantization)
        if feature_extractor_after_quantization is not None and isinstance(feature_extractor_after_quantization, dict):
            feature_extractor_after_quantization = FeatureExtractorConfig(**feature_extractor_after_quantization)
            self.feature_extractor_after_quantization = BaseFeatureExtractor.build_model(config=feature_extractor_after_quantization)
            self.gen_model_list.append(self.feature_extractor_after_quantization)

    def get_frame_feature_lengths(self, speech_lengths: torch.LongTensor):
        return ((speech_lengths // self.domain_conf.get("hop_length", 160) + 1) / self.time_ds_rate).ceil().long()

    def init_timbre_encoder(self, timbre_strategy: Optional[TimbreStrategy] = None):
        if isinstance(timbre_strategy, dict):
            timbre_strategy = TimbreStrategy(**timbre_strategy)
        self.timbre_strategy = timbre_strategy
        if self.timbre_strategy is not None and self.timbre_strategy.timbre_encoder_config is not None:
            self.timbre_encoder = TimbreEncoderPreTrainedModel.build_timbre_encoder(
                config=self.timbre_strategy.timbre_encoder_config,
                pretrained_model_name_or_path=self.timbre_strategy.timbre_encoder_config.model_dir,
                encoder_type=self.timbre_strategy.timbre_encoder_config.encoder_type,
            )
            self.gen_model_list.append(self.timbre_encoder)
            if self.timbre_strategy.timbre_encoder_config.input_type == "mel":
                # self.mel_transform = torchaudio.transforms.MelSpectrogram(
                #     sample_rate=self.target_sample_hz,
                #     n_fft=1024,
                #     win_length=1024,
                #     hop_length=256,
                #     window_fn=torch.hann_window,
                #     n_mels=80,
                #     power=2,
                #     f_min=55, f_max=7600, # for LibriTTS
                # )
                self.mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.target_sample_hz,
                    n_fft=512,
                    hop_length=160,
                    n_mels=80,
                    power=2,
                )
        else:
            self.timbre_encoder = None
        if self.timbre_strategy is not None and self.timbre_strategy.timbre_extractor_config is not None:
            self.timbre_extractor = BaseTimbreExtractorModel.build_model(config=self.timbre_strategy.timbre_extractor_config)
            self.gen_model_list.append(self.timbre_extractor)
            if self.timbre_strategy.timbre_extractor_config.merge_with_quant_out == "cross_attention":
                self.multihead_attn_for_timbre_extractor = nn.MultiheadAttention(
                    self.odim, num_heads=max(1, self.odim // 64), dropout=0.1, batch_first=True,
                    kdim=self.timbre_strategy.timbre_extractor_config.kdim, vdim=self.timbre_strategy.timbre_extractor_config.vdim, 
                )
                self.gen_model_list.append(self.multihead_attn_for_timbre_extractor)
            else:
                self.multihead_attn_for_timbre_extractor = None
        else:
            self.timbre_extractor = None
        if self.timbre_strategy is not None:
            if self.timbre_strategy.qformer_config is not None:
                self.q_former = QFormerModel(config=self.timbre_strategy.qformer_config)
                self.gen_model_list.append(self.q_former)
            else:
                self.q_former = None
            if self.timbre_strategy.merge_embed == "cross_attention":
                self.multihead_attn = nn.MultiheadAttention(
                    self.quantizer.code_dim, num_heads=max(1, self.odim // 64), dropout=0.1, batch_first=True,
                    kdim=self.timbre_strategy.timbre_encoder_config.embed_dim, vdim=self.timbre_strategy.timbre_encoder_config.embed_dim, 
                )
                self.gen_model_list.append(self.multihead_attn)
            elif self.timbre_strategy.merge_embed in ("add", "mean_pooling"):
                if self.quantizer.code_dim != self.timbre_strategy.timbre_encoder_config.embed_dim:
                    # self.timbre_linear = None
                    # pass
                    self.timbre_linear = nn.Linear(self.timbre_strategy.timbre_encoder_config.embed_dim, self.quantizer.code_dim)
                    self.gen_model_list.append(self.timbre_linear)
                    logging.info(f"init timbre_linear!")
                else:
                    self.timbre_linear = None
            if self.timbre_strategy.bottleneck_config is not None:
                if not hasattr(self, "project"):
                    self.project = nn.ModuleDict()
                if self.timbre_strategy.bottleneck_config.quantizer_in_dim is not None:
                    self.project["quantizer_in"] = nn.Linear(self.timbre_strategy.bottleneck_config.quantizer_in_dim, self.timbre_strategy.bottleneck_config.quantizer_dim)
                if self.timbre_strategy.bottleneck_config.quantizer_out_dim is not None:
                    self.project["quantizer_out"] = nn.Linear(self.timbre_strategy.bottleneck_config.quantizer_dim, self.timbre_strategy.bottleneck_config.quantizer_out_dim)
                self.gen_model_list.append(self.project)
        if self.timbre_strategy is not None and self.timbre_strategy.phaseaug_config is not None:
            self._phaseaug = modeling_phaseaug.PhaseAug(config=self.timbre_strategy.phaseaug_config)
            self.gen_model_list.append(self._phaseaug)
        if self.timbre_strategy is not None and self.timbre_strategy.praat_config is not None:
            self.praat_transform = PraatTransformer(config=self.timbre_strategy.praat_config)
            self.gen_model_list.append(self.praat_transform)
        if self.timbre_strategy is not None and self.timbre_strategy.contrastive_encoder_config is not None:
            self.contrastive_encoder = ContrastiveEncoderPreTrainedModel.build_contrastive_encoder(config=self.timbre_strategy.contrastive_encoder_config)
            self.gen_model_list.append(self.contrastive_encoder)
        else:
            self.contrastive_encoder = None
        if self.timbre_strategy is not None and self.timbre_strategy.frame_contrastive_encoder_config is not None:
            self.frame_contrastive_encoder = BaseFrameContrastiveEncoder.build_model(config=self.timbre_strategy.frame_contrastive_encoder_config)
            self.gen_model_list.append(self.frame_contrastive_encoder)
        else:
            self.frame_contrastive_encoder = None
        if self.timbre_strategy is not None and self.timbre_strategy.speaker_contrastive_encoder_config is not None:
            self.speaker_contrastive_encoder = SpeakerContrastiveEncoder.build_model(config=self.timbre_strategy.speaker_contrastive_encoder_config)
            self.gen_model_list.append(self.speaker_contrastive_encoder)
        else:
            self.speaker_contrastive_encoder = None
        if self.timbre_strategy is not None and self.timbre_strategy.speaker_predictor_with_timbre_config is not None:
            self.speaker_predictor_with_timbre = SpeakerPredictEncoderBaseModel.build_model(config=self.timbre_strategy.speaker_predictor_with_timbre_config)
            self.gen_model_list.append(self.speaker_predictor_with_timbre)
        else:
            self.speaker_predictor_with_timbre = None
        if self.timbre_strategy is not None and self.timbre_strategy.speaker_predictor_with_quant_config is not None:
            self.speaker_predictor_with_quant = SpeakerPredictEncoderBaseModel.build_model(config=self.timbre_strategy.speaker_predictor_with_quant_config)
            self.gen_model_list.append(self.speaker_predictor_with_quant)
        else:
            self.speaker_predictor_with_quant = None
        if self.timbre_strategy is not None and self.timbre_strategy.phoneme_decoder_config is not None:
            self.phoneme_decoder = PhonemeDecoderPreTrainedModel.build_model(config=self.timbre_strategy.phoneme_decoder_config)
            self.gen_model_list.append(self.phoneme_decoder)
        else:
            self.phoneme_decoder = None

    def prepare_retrain(self, retrain_model_config: Optional[RetrainModelConfig] = None):
        self.retrain_model = None
        if retrain_model_config is not None:
            self.retrain_model = RetrainModelPreTrainedModel(config=retrain_model_config)
            self.retrain_model.prepare(model=self)
    
    def remove_encoder_weight_norm(self):
        self.encoder.remove_weight_norm()
    
    def remove_decoder_weight_norm(self):
        self.decoder.remove_weight_norm()

    @property
    def generator(self):
        return torch.nn.ModuleList(self.gen_model_list)

    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment_dur is None:
            return None
        return int(self.segment_dur * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap_ratio) * segment_length))

    def forward(
        self,
        forward_generator: bool = True,
        batch: Dict = None,
    ) -> Dict[str, Any]:
        """Forward functions of generator and discriminator.

        Args:
            forward_generator (bool): Whether to forward generator.
            batch (Dict[str, Tensor]): one batch including:
                speech (Tensor): Speech waveform tensor (B, T_wav).
                speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        batch = preprocess_batch(batch, self.domain_conf.get("hop_length", 160), self.time_ds_rate)
        if forward_generator:
            if self.training:
                self.forward_step += 1
            return self._forward_generator(
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
                mel2ph=batch.get("mel2ph", None),
                frame2ph_token_ids=batch.get("frame2ph_token_ids", None),
                speaker_ids=batch.get("speaker_ids", None),
            )
        else:
            return self._forward_discriminator(
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
                mel2ph=batch.get("mel2ph", None),
                frame2ph_token_ids=batch.get("frame2ph_token_ids", None),
                speaker_ids=batch.get("speaker_ids", None),
            )

    def forward_timbre_encoder(
        self,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        transformed_speech: Optional[torch.FloatTensor] = None,
    ) -> TimbreEncoderOutput:
        if self.timbre_strategy is not None and self.timbre_strategy.timbre_encoder_config is not None:
            if self.timbre_strategy.timbre_encoder_config.input_type == "wav":
                input_speech = speech if transformed_speech is None else transformed_speech
                timbre_encoder_output = self.timbre_encoder(
                    wavs=input_speech,
                    wav_lens=speech_lengths,
                    attention_mask=lengths_to_attention_mask(speech_lengths),
                )
            elif self.timbre_strategy.timbre_encoder_config.input_type == "mel":
                if transformed_speech is None:
                    spectrogram = self.mel_transform(speech)
                    spectrogram = rearrange(spectrogram.squeeze(1), "b f t -> b t f")
                    input_spectrogram = spectrogram
                else:
                    input_spectrogram = transformed_speech
                speech_lengths = speech_lengths // self.mel_transform.hop_length + 1
                padding_mask = lengths_to_padding_mask(speech_lengths, max_lens=input_spectrogram.shape[1])
                # print(666, "mel", spectrogram.shape, speech_lengths)
                timbre_encoder_output = self.timbre_encoder(spectrogram=input_spectrogram, padding_mask=padding_mask)
            elif self.timbre_strategy.timbre_encoder_config.input_type == "mag":
                if transformed_speech is None:
                    spectrogram, scale = self.transform_speech(speech) # batch_size, channel, freq=n_fft/2+1=257, timestep+1
                    spectrogram = rearrange(spectrogram.squeeze(1), "b f t -> b t f")
                    input_spectrogram = spectrogram
                else:
                    input_spectrogram = transformed_speech
                speech_lengths = speech_lengths // self.domain_conf.get("hop_length", 160) + 1
                # print(666, speech_lengths, input_spectrogram.shape)
                padding_mask = lengths_to_padding_mask(speech_lengths, max_lens=input_spectrogram.shape[1])
                timbre_encoder_output = self.timbre_encoder(spectrogram=input_spectrogram, padding_mask=padding_mask)
            return timbre_encoder_output

    def encoder_output_minus_timbre_features(
        self,
        timbre_encoder_output: TimbreEncoderOutput,
        quant: torch.FloatTensor,
        frame_feature_lengths: Optional[torch.Tensor] = None,
    ):
        """
        params:
            timbre_encoder_output:
                for speechbrain.inference.classifiers.EncoderClassifier: [B, 1, D]
                for fast_speech_transformer.FastSpeechDecoder: [B, T, D]
            quants: [[quant, scale]]
        """
        timbre_feats = timbre_encoder_output.last_hidden_state # [2, 1671, 256]
        timbre_feature_attention_mask = timbre_encoder_output.padding_mask != 1 if timbre_encoder_output.padding_mask is not None else None # [2, 1671]
        frame_feature_attention_mask = lengths_to_attention_mask(frame_feature_lengths) if frame_feature_lengths is not None else None # [2, 418]
        if self.timbre_strategy.merge_embed in ("add", "mean_pooling"):
            if self.timbre_strategy.timbre_encoder_config.merge_with_decoder == "normal":
                if self.timbre_linear is not None:
                    timbre_feats = self.timbre_linear(timbre_feats)
        if self.timbre_strategy.merge_embed == "mean_pooling":
            timbre_feats = mean_pooling(timbre_feats, timbre_feature_attention_mask)
            if self.timbre_strategy.timbre_encoder_config.merge_with_decoder == "normal":
                timbre_feats = timbre_feats.unsqueeze(1)
        if self.timbre_encoder.config.repeat_embed:
            timbre_feats = timbre_feats.expand(-1, quant.shape[-2], -1) # wav: [B, T, D]
            if frame_feature_attention_mask is not None:
                # print(666, timbre_feats.shape, quant.shape, frame_feature_attention_mask.shape)
                timbre_feats = timbre_feats.clone() * frame_feature_attention_mask.unsqueeze(-1)
        if self.timbre_strategy.timbre_encoder_config.merge_with_decoder == "normal":
            if self.timbre_strategy.merge_embed == "cross_attention":
                pass
            else:
                quant = quant - timbre_feats
        return quant

    def merge_timbre_features(
        self,
        timbre_encoder_output: TimbreEncoderOutput,
        quants: Sequence[List],
        frame_feature_lengths: Optional[torch.Tensor] = None,
    ):
        """
        params:
            timbre_encoder_output:
                for speechbrain.inference.classifiers.EncoderClassifier: [B, 1, D]
                for fast_speech_transformer.FastSpeechDecoder: [B, T, D]
            quants: [[quant, scale]]
        """
        assert len(quants) == 1
        quant = quants[0][0] # [batch_size, time_step, dim]
        timbre_feats = timbre_encoder_output.last_hidden_state # [2, 1671, 256]
        timbre_feature_attention_mask = timbre_encoder_output.attention_mask
        if timbre_feature_attention_mask is None:
            timbre_feature_attention_mask = timbre_encoder_output.padding_mask != 1 if timbre_encoder_output.padding_mask is not None else None # [2, 1671]
        frame_feature_attention_mask = lengths_to_attention_mask(frame_feature_lengths) if frame_feature_lengths is not None else None # [2, 418]
        merged_timbre_output = {
            "timbre_feature_attention_mask": timbre_feature_attention_mask,
            "frame_feature_attention_mask": frame_feature_attention_mask,
            "orig_quant": quant,
        }
        if self.timbre_strategy.merge_embed in ("add", "mean_pooling"):
            if self.timbre_strategy.timbre_encoder_config.merge_with_decoder == "normal":
                if self.timbre_linear is not None:
                    timbre_feats = self.timbre_linear(timbre_feats)
        if self.timbre_strategy.merge_embed == "mean_pooling":
            timbre_feats = mean_pooling(timbre_feats, timbre_feature_attention_mask)
            merged_timbre_output["timbre_feats"] = timbre_feats
            if self.timbre_strategy.timbre_encoder_config.merge_with_decoder == "normal":
                timbre_feats = timbre_feats.unsqueeze(1)
        elif self.timbre_strategy.merge_embed == "add":
            merged_timbre_output["timbre_feats"] = timbre_feats
        if self.timbre_encoder.config.repeat_embed:
            timbre_feats = timbre_feats.expand(-1, quant.shape[-2], -1) # wav: [B, T, D]
            if frame_feature_attention_mask is not None:
                # print(666, timbre_feats.shape, quant.shape, frame_feature_attention_mask.shape)
                timbre_feats = timbre_feats.clone() * frame_feature_attention_mask.unsqueeze(-1)
        if self.q_former is not None:
            timbre_feature_padding_mask = timbre_encoder_output.padding_mask
            q_former_output = self.q_former(m1=timbre_feats, m1_key_padding_mask=timbre_feature_padding_mask)
            timbre_feats = q_former_output.last_hidden_state
            merged_timbre_output["q_former_hidden_states"] = q_former_output.hidden_states
            merged_timbre_output["q_former_last_hidden_state"] = timbre_feats
            merged_timbre_output["timbre_feature_padding_mask"] = timbre_feature_padding_mask
        if self.timbre_strategy.timbre_encoder_config.merge_with_decoder == "normal":
            if self.timbre_strategy.merge_embed == "cross_attention":
                ca_output, ca_weight = self.multihead_attn(
                    quant, timbre_feats, timbre_feats,
                )
                merged_timbre_output["ca_output"] = ca_output
                merged_timbre_output["ca_weight"] = ca_weight
                quant = ca_output + quant
            else:
                # print(666, timbre_feats.shape)
                quant = timbre_feats + quant
                # print(777, quant.sum())
            # quants[0][0] = quant
            merged_timbre_output["quant"] = quant
        return MergedTimbreOutput(**merged_timbre_output)

    def forward_timbre_extractor(
        self,
        features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.BoolTensor] = None,
        feature_padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs
    ) -> TimbreExtractorOutput:
        return self.timbre_extractor(
            hidden_states=features,
            attention_mask=feature_attention_mask,
            padding_mask=feature_padding_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

    def merge_timbre_extractor(
        self,
        timbre_extractor_output: TimbreExtractorOutput,
        quants: Sequence[List],
        timbre_feature_attention_mask: Optional[torch.BoolTensor] = None,
        timbre_feature_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MergedTimbreExtractorOutput:
        quant = quants[0][0] # [batch_size, time_step, dim]
        merged_timbre_extractor_output = {}
        if self.timbre_strategy.timbre_extractor_config.merge_with_quant_out == "mean_pooling":
            timbre_feats = mean_pooling(timbre_extractor_output.last_hidden_state, timbre_feature_attention_mask)
            merged_timbre_extractor_output["timbre_feats"] = timbre_feats # [batch_size, dim]
            if self.timbre_strategy.timbre_extractor_config.merge_with_decoder == "normal":
                timbre_feats = timbre_feats[:, None, :].expand(-1, quant.shape[1], -1)
                timbre_feats = torch.where(timbre_feature_attention_mask, timbre_feats, 0.0)
                quant = quant + timbre_feats
                merged_timbre_extractor_output["quant"] = quant
        elif self.timbre_strategy.timbre_extractor_config.merge_with_quant_out == "cross_attention":
            ca_output, ca_weight = self.multihead_attn_for_timbre_extractor(
                quant, timbre_extractor_output.last_hidden_state, timbre_extractor_output.last_hidden_state,
            )
            merged_timbre_extractor_output["ca_output"] = ca_output
            merged_timbre_extractor_output["ca_weight"] = ca_weight
            merged_timbre_extractor_output["timbre_feats"] = ca_output
            if self.timbre_strategy.timbre_extractor_config.merge_with_decoder == "normal":
                quant = ca_output + quant
                merged_timbre_extractor_output["quant"] = quant
        else:
            raise NotImplementedError
        return MergedTimbreExtractorOutput(**merged_timbre_extractor_output)

    def forward_contrastive_encoder(
        self,
        encoder_features: Optional[torch.Tensor] = None,
        quantized_features: Optional[torch.Tensor] = None,
        feature_lengths: Optional[torch.Tensor] = None,
        mel2ph: Optional[Sequence] = None,
        **kwargs
    ) -> ContrastiveEncoderOutput:
        return self.contrastive_encoder(
            encoder_features=encoder_features,
            quantized_features=quantized_features,
            feature_lengths=feature_lengths,
            mel2ph=mel2ph,
            output_hidden_states=True,
            **kwargs
        )

    def forward_frame_contrastive_encoder(
        self,
        encoder_features: Optional[torch.FloatTensor] = None,
        encoder_perturbed_features: Optional[torch.FloatTensor] = None,
        quantizer_features: Optional[torch.FloatTensor] = None,
        quantizer_perturbed_features: Optional[torch.FloatTensor] = None,
        code_indices: Optional[torch.LongTensor] = None,
        perturbed_code_indices: Optional[torch.LongTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        mel2ph: Optional[Sequence[Sequence[List]]] = None,
        **kwargs
    ) -> FrameContrastiveEncoderOutput:
        features, perturbed_features = self.frame_contrastive_encoder.get_features(
            encoder_features=encoder_features,
            encoder_perturbed_features=encoder_perturbed_features,
            quantizer_features=quantizer_features,
            quantizer_perturbed_features=quantizer_perturbed_features,
        )
        return self.frame_contrastive_encoder(
            features=features,
            perturbed_features=perturbed_features,
            code_indices=code_indices,
            perturbed_code_indices=perturbed_code_indices,
            feature_lengths=feature_lengths,
            mel2ph=mel2ph,
            **kwargs
        )

    def forward_speaker_contrastive_encoder(
        self,
        speech: torch.FloatTensor,
        speech_lengths: Optional[torch.LongTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> SpeakerContrastiveEncoderOutput:
        if self.timbre_strategy.timbre_encoder_config.input_type == "wav":
            # forward_timbre_encoder = lambda half_speech: self.forward_timbre_encoder(half_speech.speech.unsqueeze(1), half_speech.speech_lengths).last_hidden_state
            def forward_timbre_encoder(half_speech):
                return self.forward_timbre_encoder(half_speech.speech.unsqueeze(1), half_speech.speech_lengths).last_hidden_state
            get_feature_lengths_from_speech_lengths = lambda x: torch.tensor([1] * speech.shape[0], dtype=torch.long, device=speech.device)
        elif self.timbre_strategy.timbre_encoder_config.input_type in ("mel", "mag"):
            forward_timbre_encoder = lambda half_speech: self.forward_timbre_encoder(half_speech.speech.unsqueeze(1), half_speech.speech_lengths).last_hidden_state
            get_feature_lengths_from_speech_lengths = lambda x: x // self.domain_conf.get("hop_length", 160) + 1
        return self.speaker_contrastive_encoder(
            forward_timbre_encoder,
            speech.squeeze(1),
            speech_lengths,
            get_feature_lengths_from_speech_lengths(speech_lengths),
            get_feature_lengths_from_speech_lengths,
            speaker_ids,
            **kwargs
        )

    def forward_speaker_predictor_with_timbre(
        self,
        timbre_encoder_output: TimbreEncoderOutput,
        speaker_ids: Optional[torch.LongTensor] = None,
    ) -> SpeakerPredictEncoderOutput:
        attention_mask = timbre_encoder_output.attention_mask
        return self.speaker_predictor_with_timbre(
            features=timbre_encoder_output.last_hidden_state,
            feature_lengths=attention_mask.sum(-1),
            attention_mask=attention_mask,
            labels=speaker_ids,
        )

    def forward_speaker_predictor_with_quant(
        self,
        features,
        feature_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
    ) -> SpeakerPredictEncoderOutput:
        if attention_mask is None:
            attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=features.shape[1])
        return self.speaker_predictor_with_quant(
            features=features,
            feature_lengths=feature_lengths,
            attention_mask=attention_mask,
            labels=speaker_ids,
        )

    def forward_phoneme_decoder(
        self,
        phoneme_embeddings: torch.FloatTensor,
        phoneme_token_ids: Optional[Union[Sequence, torch.LongTensor]],
        feature_lengths: torch.LongTensor,
        **kwargs
    ):
        if not isinstance(phoneme_token_ids, torch.Tensor):
            phoneme_token_ids = [torch.tensor(_phoneme_token_ids, dtype=torch.long, device=phoneme_embeddings.device) for _phoneme_token_ids in phoneme_token_ids]
            phoneme_token_ids = torch.nn.utils.rnn.pad_sequence(phoneme_token_ids, batch_first=True, padding_value=-100)
            if phoneme_embeddings.shape[1] > phoneme_token_ids.shape[1]:
                _phoneme_token_ids = torch.full((phoneme_embeddings.shape[0], phoneme_embeddings.shape[1]), fill_value=-100, dtype=torch.long, device=phoneme_embeddings.device)
                _phoneme_token_ids[:, :phoneme_token_ids.shape[1]] = phoneme_token_ids
                phoneme_token_ids = _phoneme_token_ids
            # phoneme_token_ids = torch.ones(phoneme_embeddings.shape[0], phoneme_embeddings.shape[1], device=phoneme_embeddings.device, dtype=torch.long)
        return self.phoneme_decoder(
            phoneme_embeddings=phoneme_embeddings,
            labels=phoneme_token_ids,
            feature_lengths=feature_lengths,
        )

    def _encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert 0 < channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        # print("length:", length, "stride:", stride)
        for offset in range(0, length, stride):
            # print("start:", offset, "end:", offset + segment_length)
            frame = x[:, :, offset: offset + segment_length]
            encoded_frame = self._encode_frame(frame)
            encoded_frames.append(encoded_frame)
        return encoded_frames

    def transform_speech(self, x: torch.Tensor):
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment_dur is None or duration <= 1e-5 + self.segment_dur
        if self.audio_normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        if self.codec_domain[0] == "stft":
            x_complex = self.enc_trans_func(x.squeeze(1))
            if self.encoder.input_size == 2:
                x = torch.stack([x_complex.real, x_complex.imag], dim=1)
            else:
                x = torch.cat([x_complex.real, x_complex.imag], dim=1)
        elif self.codec_domain[0] == "mag":
            x_mag = self.enc_trans_func(x.squeeze(1))
            if self.encoder.input_size == 1:
                x = x_mag.unsqueeze(1)
            else:
                x = x_mag
        elif self.codec_domain[0] == "mag_angle":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x_mag = torch.abs(x_complex)
            x_log_mag = torch.log(torch.clamp(x_mag, min=1e-6))
            x_angle = torch.angle(x_complex)
            if self.encoder.input_size == 2:
                x = torch.stack([x_log_mag, x_angle], dim=1)
            else:
                x = torch.cat([x_log_mag, x_angle], dim=1)
        elif self.codec_domain[0] == "mag_phase":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x_mag = torch.abs(x_complex)
            x_log_mag = torch.log(torch.clamp(x_mag, min=1e-6))
            x_phase = x_complex / torch.clamp(x_mag, min=1e-6)
            if self.encoder.input_size == 3:
                x = torch.stack([x_log_mag, x_phase.real, x_phase.imag], dim=1)
            else:
                x = torch.cat([x_log_mag, x_phase.real, x_phase.imag], dim=1)
        elif self.codec_domain[0] == "mel":
            x = self.enc_trans_func(x.squeeze(1))
            if self.encoder.input_size == 1:
                x = x.unsqueeze(1)
        elif self.codec_domain[0] == "mag_oracle_phase":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x = torch.abs(x_complex)
            if self.encoder.input_size == 1:
                x = x.unsqueeze(1)
            x_phase = torch.angle(x_complex)
            scale = (scale, x_phase)
        # x: batch_size, channel, freq=n_fft/2+1=257, timestep+1
        return x, scale

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        x, scale = self.transform_speech(x)
        # emb = self.encoder(x)
        encoder_output = self.encoder(x)
        emb = encoder_output.last_hidden_state
        if self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_after_encoder:
            emb = self.instance_norm_1d(emb)
        return EncodedFrameOutput(emb=emb, scale=scale, transformed_speech=x, hidden_states=encoder_output.hidden_states)
        # return emb, scale

    def _decode(self, encoded_frames: tp.List[EncodedFrame], **kwargs) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0], **kwargs)

        frames = []
        for frame in encoded_frames:
            frames.append(self._decode_frame(frame, **kwargs))

        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame, **kwargs) -> torch.Tensor:
        codes, scale = encoded_frame
        emb = codes
        out = self.decoder(emb, **kwargs)
        if self.codec_domain[1] == "stft":
            if len(out.shape) == 3:
                out_list = torch.split(out, out.shape[1]//2, dim=1)
            else:
                out_list = torch.split(out, 1, dim=1)
            out = torch.complex(out_list[0], out_list[1])
            out = self.dec_trans_func(out).unsqueeze(1)
        elif self.codec_domain[1] == "mag_phase":
            if len(out.shape) == 3:
                out_list = torch.split(out, out.shape[1] // 3, dim=1)
            else:
                out_list = [x.squeeze(1) for x in torch.split(out, 1, dim=1)]
            x_mag = F.softplus(out_list[0])
            x_phase = torch.complex(out_list[1], out_list[2])
            out = x_mag * x_phase
            out = self.dec_trans_func(out).unsqueeze(1)
        elif self.codec_domain[1] == "mag_angle":
            if len(out.shape) == 3:
                out_list = torch.split(out, out.shape[1] // 2, dim=1)
            else:
                out_list = [x.squeeze(1) for x in torch.split(out, 1, dim=1)]
            x_mag = F.softplus(out_list[0])
            x_angle = torch.sin(out_list[1]) * torch.pi
            x_spec = torch.complex(torch.cos(x_angle) * x_mag, torch.sin(x_angle) * x_mag)
            out = self.dec_trans_func(x_spec).unsqueeze(1)
        elif self.codec_domain[1] == "mag_oracle_phase":
            if len(out.shape) == 4:
                out = out.squeeze(1)
            (scale, x_angle), x_mag = scale, out
            x_spec = torch.complex(torch.cos(x_angle)*x_mag, torch.sin(x_angle)*x_mag)
            out = self.dec_trans_func(x_spec).unsqueeze(1)
        elif (self.codec_domain[0] in ["stft", "mag", "mag_phase", "mag_angle", "mag_oracle_phase"] and
              self.codec_domain[1] == "time"):
            hop_length = self.domain_conf.get("hop_length", 160)
            out = out[:, :, hop_length//2: -hop_length//2]

        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def _forward_generator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        mel2ph: Optional[Sequence] = None,
        frame2ph_token_ids: Optional[Sequence] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
        disturb_speech: Optional[bool] = None,
        exchange_timbre: Optional[bool] = None,
        without_timbre: Optional[bool] = None,
        without_rvq: Optional[bool] = None,
        apply_empty_timbre: Optional[bool] = None,
        num_codebooks: Optional[int] = None,
        exchange_codebooks: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = speech.size(0)
        speech = speech.unsqueeze(1)
        orig_speech = speech.clone()
        # feature_lengths = ((speech_lengths // 160 + 1) / 4).ceil().long() if speech_lengths is not None else None
        feature_lengths = ((speech_lengths // self.domain_conf.get("hop_length", 160) + 1) / self.time_ds_rate).ceil().long() if speech_lengths is not None else None
        feature_attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None
        feature_padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None

        l1Loss = torch.nn.L1Loss(reduction='mean')
        l2Loss = torch.nn.MSELoss(reduction='mean')
        commit_losses = []
        enc_quant_losses = []
        codes = []
        code_indices = [] # offset, num_codebooks, batch_size, timestep. offset is usually set to 1.
        if self.timbre_strategy is not None and self.timbre_strategy.phaseaug_config is not None:
            speech = self._phaseaug(speech)
        timbre_encoder_output = None

        # frames = self._encode(speech)
        if disturb_speech is None:
            disturb_speech = self.timbre_strategy is not None and self.timbre_strategy.praat_config is not None
            # print(f"666 disturb_speech = {disturb_speech}")
        if disturb_speech:
            praat_output = self.praat_transform(speech)
            perturbed_speech = praat_output.snd.unsqueeze(1)
            perturbed_encoded_frames = self._encode(perturbed_speech) # offset list: [offset1]
            perturbed_frames = [(frame.emb, frame.scale, frame.transformed_speech) for frame in perturbed_encoded_frames]
        else:
            praat_output = None
            perturbed_speech = None
            perturbed_encoded_frames = None
            perturbed_frames = None
        encoded_frames = self._encode(speech) # offset list: [offset1]
        frames = [(frame.emb, frame.scale, frame.transformed_speech) for frame in encoded_frames]

        if self.timbre_encoder is not None:
            if self.timbre_encoder.config.transformed_speech_for_timbre_encoder and not exchange_timbre:
                timbre_speech_lengths = speech_lengths
                transformed_speech = encoded_frames[0].transformed_speech
                transformed_speech = rearrange(transformed_speech.squeeze(1), "b f t -> b t f")
                timbre_encoder_output = self.forward_timbre_encoder(speech_lengths=timbre_speech_lengths, transformed_speech=transformed_speech)
            else:
                timbre_speech = speech
                timbre_speech_lengths = speech_lengths
                if exchange_timbre:
                    assert speech.shape[0] == 2
                    timbre_speech = speech.clone()
                    timbre_speech_lengths = speech_lengths.clone()
                    timbre_speech[[0, 1]] = timbre_speech[[1, 0]]
                    timbre_speech_lengths[[0, 1]] = timbre_speech_lengths[[1, 0]]
                    # feature_lengths[[0, 1]] = feature_lengths[[1, 0]]
                timbre_encoder_output = self.forward_timbre_encoder(speech=timbre_speech, speech_lengths=timbre_speech_lengths)
                # if exchange_timbre:
                #     timbre_encoder_output.padding_mask[[0, 1]] = timbre_encoder_output.padding_mask[[1, 0]]
        if self.speaker_predictor_with_timbre is not None:
            speaker_predictor_with_timbre_output = self.forward_speaker_predictor_with_timbre(timbre_encoder_output, speaker_ids)
        else:
            speaker_predictor_with_timbre_output = None

        contrastive_loss = 0.0 if self.contrastive_encoder is not None else None
        speaker_predictor_with_quant_loss = 0.0 if self.speaker_predictor_with_quant is not None else None
        speaker_contrastive_loss = 0.0 if self.speaker_contrastive_encoder is not None else None
        phoneme_loss = 0.0 if self.phoneme_decoder is not None else None
        contrastive_encoder_output_list = [] if self.contrastive_encoder is not None else None
        speaker_predictor_with_quant_output_list = [] if self.speaker_predictor_with_quant is not None else None
        speaker_contrastive_encoder_output_list = [] if self.speaker_contrastive_encoder is not None else None
        phoneme_decoder_output_list = [] if self.phoneme_decoder is not None else None
        quant_in_list = []
        sub_quants_list = []
        timbre_extractor_output = None
        merged_timbre_extractor_output = None
        for i, (emb, scale, transformed_speech) in enumerate(frames):
            # if self.timbre_strategy is not None and self.timbre_strategy.praat_config is not None:
                # if self.timbre_strategy.praat_config.reconstructed_speech_from == "orig_speech":
                #     pass
                # elif self.timbre_strategy.praat_config.reconstructed_speech_from == "perturbed_speech":
                #     emb = perturbed_frames[i][0]
            feature_before_quantization = None
            if self.feature_extractor_before_quantization is not None:
                feature_before_quantization = emb.clone()
                emb = self.feature_extractor_before_quantization(feature_before_quantization)
            quant_in = emb.clone()

            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_before_quantizer is not None:
                quant_in = self.project["quantizer_in"](quant_in)

            if self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.encoder_output_minus_timbre_features:
                quant = self.encoder_output_minus_timbre_features(timbre_encoder_output, quant_in, feature_lengths)
                quant_in = quant

            if self.instance_norm_before_quantization or \
                (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization):
                quant_in = self.instance_norm_1d(quant_in)

            # if self.training and self.forward_step % 1000 == 0:
            #     self.quantizer.rq.model.inited[0] = False # kmeans update
            quant_in_list.append(quant_in)
            quant_out, indices, commit_loss, sub_quants = self.quantizer(quant_in)

            if self.frame_contrastive_encoder is not None:
                # vq_idx = 0
                # perturbed_quant_out, perturbed_indices, perturbed_commit_loss, perturbed_sub_quants = self.quantizer(perturbed_frames[i][0])
                # frame_contrastive_encoder_output = self.forward_frame_contrastive_encoder(
                #     encoder_features=encoded_frames[0].emb,
                #     encoder_perturbed_features=perturbed_encoded_frames[0].emb,
                #     quantizer_features=sub_quants[vq_idx].transpose(1, 2),
                #     quantizer_perturbed_features=perturbed_sub_quants[vq_idx].transpose(1, 2),
                #     feature_lengths=feature_lengths,
                #     mel2ph=mel2ph,
                #     code_indices=indices[vq_idx],
                #     perturbed_code_indices=perturbed_indices[vq_idx],
                # )

                encoder_features = encoded_frames[0].emb
                encoder_perturbed_features = perturbed_encoded_frames[0].emb
                last_encoder_layer = self.frame_contrastive_encoder.config.encoder_last_n_layer
                if last_encoder_layer is not None and last_encoder_layer < -1:
                    encoder_features = encoded_frames[0].hidden_states[last_encoder_layer].permute(0, 2, 1)
                    encoder_perturbed_features = perturbed_encoded_frames[0].hidden_states[last_encoder_layer].permute(0, 2, 1)
                # print(666, encoder_features.shape)

                # ?
                vq_idx = 0
                quant0 = sub_quants[vq_idx].transpose(1, 2)
                # quant0, embed_ind0 = self.quantizer.rq.model.layers[vq_idx]._codebook(
                #     encoded_frames[0].emb, 
                #     [
                #         self.quantizer.rq.model.inited[vq_idx],
                #         self.quantizer.rq.model.cluster_size[vq_idx],
                #         self.quantizer.rq.model.embed[vq_idx],
                #         self.quantizer.rq.model.embed_avg[vq_idx],
                #     ],
                # )
                # quant0 = quant0.detach()
                # quant0 = self.quantizer.rq.model.layers[0].project_out(quant0)
                perturbed_quant_out, perturbed_indices, perturbed_commit_loss, perturbed_sub_quants = self.quantizer(perturbed_frames[i][0])
                perturbed_quant0 = perturbed_sub_quants[vq_idx].transpose(1, 2)
                # perturbed_quant0, perturbed_embed_ind0 = self.quantizer.rq.model.layers[vq_idx]._codebook(
                #     perturbed_frames[i][0], 
                #     [
                #         self.quantizer.rq.model.inited[vq_idx],
                #         self.quantizer.rq.model.cluster_size[vq_idx],
                #         self.quantizer.rq.model.embed[vq_idx],
                #         self.quantizer.rq.model.embed_avg[vq_idx],
                #     ],
                # )
                # perturbed_quant0 = perturbed_quant0.detach()
                # perturbed_quant0 = self.quantizer.rq.model.layers[0].project_out(perturbed_quant0)

                frame_contrastive_encoder_output = self.forward_frame_contrastive_encoder(
                    encoder_features=encoder_features,
                    encoder_perturbed_features=encoder_perturbed_features,
                    quantizer_features=quant0,
                    quantizer_perturbed_features=perturbed_quant0,
                    feature_lengths=feature_lengths,
                    mel2ph=mel2ph,
                    code_indices=indices[vq_idx],
                    perturbed_code_indices=perturbed_indices[vq_idx],
                    training_step=self.forward_step,
                )

            else:
                frame_contrastive_encoder_output = None

            if num_codebooks is not None:
                quant_out = sub_quants[:num_codebooks].sum(0).transpose(1, 2)
            if exchange_codebooks:
                quant_out0 = sub_quants[0].transpose(1, 2)
                quant_out0[[0, 1]] = quant_out0[[1, 0]]
                quant_out1 = sub_quants[1].transpose(1, 2)
                quant_out = quant_out0 + quant_out1
            sub_quants_list.append(sub_quants) # [num_rvq, batch_size, dim, seq_len]

            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_after_quantizer is not None:
                quant_out = self.project["quantizer_out"](quant_out)
            code_embs = quant_out
            # qv = self.quantizer.forward(emb, self.sample_rate, self.bandwidth)
            commit_losses.append(commit_loss)
            # enc_quant_losses.append(l2Loss(quant_out, quant_in) ** 2) # which is right ?
            enc_quant_losses.append(l2Loss(quant_out, quant_in))
            codes.append([code_embs, scale])

            if self.timbre_extractor is not None:
                features = emb
                if self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.feature_minus_quant:
                    features = emb - quant_out
                if apply_empty_timbre:
                    features = torch.full_like(quant_in, fill_value=0.0)
                    # features = torch.full_like(quant_in, fill_value=1.0)
                timbre_extractor_output = self.forward_timbre_extractor(
                    features=features,
                    feature_attention_mask=feature_attention_mask,
                    feature_padding_mask=feature_padding_mask,
                )
                if exchange_timbre:
                    timbre_extractor_output.exchange_timbre()
            
            code_indices.append(indices)
            if self.contrastive_encoder is not None:
                vq_idx = 0
                # print(666, quant_in.shape, quant_out.shape, feature_lengths)
                quant0, embed_ind0 = self.quantizer.rq.model.layers[vq_idx]._codebook(
                    quant_in, 
                    [
                        self.quantizer.rq.model.inited[vq_idx],
                        self.quantizer.rq.model.cluster_size[vq_idx],
                        self.quantizer.rq.model.embed[vq_idx],
                        self.quantizer.rq.model.embed_avg[vq_idx],
                    ],
                )
                quant0 = quant0.detach()
                quant0 = self.quantizer.rq.model.layers[0].project_out(quant0)
                # print(666, quant0.sum(), sub_quants[vq_idx].transpose(1, 2).sum())

                contrastive_encoder_output = self.forward_contrastive_encoder(
                    encoder_features=quant_in,
                    # encoder_features=quant0,
                    # quantized_features=quant_out,
                    # encoder_features=None,
                    # quantized_features=sub_quants[vq_idx].transpose(1, 2),
                    quantized_features=quant0,
                    feature_lengths=feature_lengths,
                    mel2ph=mel2ph,
                )
                contrastive_loss += contrastive_encoder_output.loss
                contrastive_encoder_output_list.append(contrastive_encoder_output)

            if self.speaker_predictor_with_quant is not None:
                input_features = None
                if self.timbre_strategy.speaker_predictor_with_quant_config.encoder_input_type == "quant_in":
                    input_features = quant_in
                elif self.timbre_strategy.speaker_predictor_with_quant_config.encoder_input_type == "quant_out":
                    input_features = quant_out
                speaker_predictor_with_quant_output = self.forward_speaker_predictor_with_quant(
                    features=input_features,
                    feature_lengths=feature_lengths,
                    speaker_ids=speaker_ids,
                )
                speaker_predictor_with_quant_loss += speaker_predictor_with_quant_output.loss
                speaker_predictor_with_quant_output_list.append(speaker_predictor_with_quant_output)
            if self.speaker_contrastive_encoder is not None:
                speaker_contrastive_encoder_output = self.forward_speaker_contrastive_encoder(
                    speech=speech,
                    speech_lengths=speech_lengths,
                    feature_lengths=feature_lengths,
                    attention_mask=feature_attention_mask,
                    speaker_ids=speaker_ids,
                )
                speaker_contrastive_loss += speaker_contrastive_encoder_output.loss
                speaker_contrastive_encoder_output_list.append(speaker_contrastive_encoder_output)
            if self.phoneme_decoder is not None:
                # print(666, quant_in.shape, speech_lengths, feature_lengths, [len(item) for item in frame2ph_token_ids])
                phoneme_decoder_output = self.forward_phoneme_decoder(
                    phoneme_embeddings=quant_in,
                    phoneme_token_ids=frame2ph_token_ids,
                    feature_lengths=feature_lengths,
                )
                phoneme_loss += phoneme_decoder_output.loss
                phoneme_decoder_output_list.append(phoneme_decoder_output)
        merged_timbre_features = None
        if not without_timbre and self.timbre_encoder is not None:
            merged_timbre_features = self.merge_timbre_features(
                timbre_encoder_output, codes, feature_lengths,
            )
            if self.timbre_encoder.config.merge_with_decoder == "normal":
                codes[0][0] = merged_timbre_features.quant
                if self.feature_extractor_after_quantization is not None:
                    codes[0][0] = self.feature_extractor_after_quantization(merged_timbre_features.quant)
                    # print(666, "feature_extractor_after_quantization")
                recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
            elif self.timbre_encoder.config.merge_with_decoder == "conditional_layer_norm":
                recon_speech = self._decode(codes, condition_embedding=merged_timbre_features.timbre_feats)[:, :, :speech.shape[-1]]
        elif not without_timbre and self.timbre_extractor is not None:
            merged_timbre_extractor_output = self.merge_timbre_extractor(
                timbre_extractor_output, codes,
                timbre_feature_attention_mask=feature_attention_mask,
                timbre_feature_lengths=feature_lengths,
            )
            if self.timbre_extractor.config.merge_with_decoder == "normal":
                codes[0][0] = merged_timbre_extractor_output.quant
                if self.feature_extractor_after_quantization is not None:
                    codes[0][0] = self.feature_extractor_after_quantization(merged_timbre_features.quant)
                recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
            elif self.timbre_extractor.config.merge_with_decoder == "conditional_layer_norm":
                recon_speech = self._decode(codes, condition_embedding=merged_timbre_extractor_output.timbre_feats)[:, :, :speech.shape[-1]]
        else:
            recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
        commit_loss = torch.stack(commit_losses).sum()
        enc_quant_loss = torch.stack(enc_quant_losses).sum()
        feature_extractor_loss = torch.tensor([0.0], device=commit_loss.device) if feature_before_quantization is None else l2Loss(feature_before_quantization, codes[0][0])

        # A: recon loss
        if orig_speech.shape[-1] != recon_speech.shape[-1]:
            min_length = min(orig_speech.shape[-1], recon_speech.shape[-1])
            orig_speech = orig_speech[:, :, :min_length]
            recon_speech = recon_speech[:, :, :min_length] # recon_speech may be shorter than orig_speech
        recon_loss = l1Loss(orig_speech, recon_speech)
        # B: multiple spectral recon loss - eq (4) and (5) in https://arxiv.org/abs/2107.03312
        multi_spectral_recon_loss = self.zero
        if self.multi_spectral_recon_loss_weight > 0:
            for mel_transform in self.mel_spec_transforms:
                # mel_transform: (..., Time) -> (..., n_mel, Frame)
                if not self.use_power_spec_loss:
                    orig_mel, recon_mel = map(mel_transform, (orig_speech, recon_speech))

                    l1_mel_loss = l1Loss(orig_mel, recon_mel)
                    l2_mel_loss = l2Loss(orig_mel, recon_mel)
                else:
                    orig_mel, orig_power = mel_transform(orig_speech, self.use_power_spec_loss)
                    recon_mel, recon_power = mel_transform(recon_speech, self.use_power_spec_loss)
                    l1_mel_loss = l1Loss(orig_mel, recon_mel) * 0.5 + l1Loss(orig_power, recon_power) * 0.5
                    l2_mel_loss = l2Loss(orig_mel, recon_mel) * 0.5 + l2Loss(orig_power, recon_power) * 0.5

                multi_spectral_recon_loss = multi_spectral_recon_loss + (l1_mel_loss + l2_mel_loss)

            multi_spectral_recon_loss = multi_spectral_recon_loss / len(self.mel_spec_transforms)
        # C-1: calculate discriminator outputs
        # disc_outputs in the format [disc1_outputs, disc2_outputs, ...]
        # disc1_outputs includes [logits, intermediates]
        # intermediates includes [layer_1_intermediate, layer_2_intermediate, ...]
        fake_disc_outputs = self.discriminator(recon_speech)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            real_disc_outputs = self.discriminator(orig_speech)

        # C-2: calculate discriminator loss including adversarial and feat matching losses
        adversarial_losses = []
        disc_feature_losses = []
        for real_output, fake_output in zip(real_disc_outputs, fake_disc_outputs):
            real_logits, real_intermediates = real_output
            fake_logits, fake_intermediates = fake_output
            adversarial_losses.append(torch.mean(F.relu(1 - fake_logits)))
            for i, (real_inter, fake_inter) in enumerate(zip(real_intermediates, fake_intermediates)):
                if i >= self.feat_match_layer_start:
                    _loss = F.l1_loss(real_inter.detach(), fake_inter)
                    disc_feature_losses.append(_loss)

        adversarial_loss = torch.stack(adversarial_losses).mean()
        feat_match_loss = torch.stack(disc_feature_losses).mean()

        # calculate losses
        gen_loss = recon_loss * self.recon_loss_weight + \
                   multi_spectral_recon_loss * self.multi_spectral_recon_loss_weight + \
                   adversarial_loss * self.adversarial_loss_weight + \
                   feat_match_loss * self.feat_match_loss_weight + \
                   feature_extractor_loss
        self.gen_loss += gen_loss.item()
        # loss = gen_loss + commit_loss + enc_quant_loss * self.enc_quant_loss_weight
        loss = gen_loss + commit_loss
        if self.contrastive_encoder is not None:
            loss += contrastive_loss
        if frame_contrastive_encoder_output is not None:
            loss += frame_contrastive_encoder_output.loss
        if self.speaker_predictor_with_timbre is not None:
            loss += speaker_predictor_with_timbre_output.loss * self.timbre_strategy.speaker_predictor_with_timbre_config.loss_weight
        if self.speaker_predictor_with_quant is not None:
            loss += speaker_predictor_with_quant_loss * self.timbre_strategy.speaker_predictor_with_quant_config.loss_weight
        if self.speaker_contrastive_encoder is not None:
            # print(f"666 loss = {loss} speaker_contrastive_loss = {speaker_contrastive_loss}")
            loss += speaker_contrastive_loss
        if self.phoneme_decoder is not None:
            loss += self.timbre_strategy.phoneme_decoder_config.loss_weight * phoneme_loss

        stats = dict(
            generator_loss=loss.item(),
            generator_recon_loss=recon_loss.item(),
            generator_multi_spectral_recon_loss=multi_spectral_recon_loss.item(),
            generator_adv_loss=adversarial_loss.item(),
            generator_feat_match_loss=feat_match_loss.item(),
            generator_commit_loss=commit_loss.item(),
            generator_enc_quant_loss=enc_quant_loss.item(),
            generator_feature_extractor_loss=feature_extractor_loss.item(),
        )
        if self.contrastive_encoder is not None:
            stats["generator_contrastive_loss"] = contrastive_loss.item()
            if contrastive_encoder_output.loss_dict:
                for loss_type, loss_value in contrastive_encoder_output.loss_dict.items():
                    stats[f"generator_contrastive_encoder_{loss_type}"] = loss_value
        if frame_contrastive_encoder_output is not None:
            stats["generator_frame_contrastive_loss"] = frame_contrastive_encoder_output.loss.item()
            if frame_contrastive_encoder_output.loss_dict:
                for loss_type, loss_value in frame_contrastive_encoder_output.loss_dict.items():
                    stats[f"generator_frame_contrastive_encoder_{loss_type}"] = loss_value
                    stats[f"generator_frame_contrastive_encoder_loss_weight"] = self.frame_contrastive_encoder.config.loss_weight[0]
        if self.speaker_predictor_with_timbre is not None:
            stats["generator_speaker_predict_with_timbre_loss"] = speaker_predictor_with_timbre_output.loss.item()
        if self.speaker_predictor_with_quant is not None:
            stats["speaker_predictor_with_quant_loss"] = speaker_predictor_with_quant_loss.item()
        if self.speaker_contrastive_encoder is not None:
            stats["generator_speaker_contrastive_loss"] = speaker_contrastive_loss.item()
            if speaker_contrastive_encoder_output.loss_dict:
                for loss_type, loss_value in speaker_contrastive_encoder_output.loss_dict.items():
                    stats[f"generator_speaker_contrastive_encoder_{loss_type}"] = loss_value
        if self.phoneme_decoder is not None:
            stats["phoneme_loss"] = phoneme_loss.item()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        output = {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # needed for trainer
            "real": orig_speech,
            "fake": recon_speech,
            "code_indices": code_indices, # code index
            "quants": [embed for (embed, scale)in codes], # [[batch_size, time_step, dim]]
            "encoder_output": frames[0][0],
            "decoder_input": codes[0][0],
            "quants_in": quant_in_list, 
            "sub_quants": sub_quants_list,
            "transformed_speech": [frame.transformed_speech for frame in encoded_frames],
            "praat_output": praat_output,
            "frame_contrastive_encoder_output": frame_contrastive_encoder_output,
            "timbre_encoder_output": timbre_encoder_output,
            "merged_timbre_features": merged_timbre_features,
            "contrastive_encoder_output": contrastive_encoder_output_list,
            "speaker_predictor_with_timbre_output": speaker_predictor_with_timbre_output,
            "speaker_predictor_with_quant_output_list": speaker_predictor_with_quant_output_list,
            "speaker_contrastive_encoder_output": speaker_contrastive_encoder_output_list,
            "phoneme_decoder_output": phoneme_decoder_output_list,
            "timbre_extractor_output": timbre_extractor_output,
            "merged_timbre_extractor_output": merged_timbre_extractor_output,
        }
        output = {param: value for param, value in output.items() if value is not None}

        # tot1 = 0
        # for module_name, param in self.timbre_encoder.named_parameters():
        #     tot1 += param.abs().sum()
        # tot2 = 0
        # for module_name, param in self.phoneme_decoder.named_parameters():
        #     tot2 += param.abs().sum()
        # tot3 = 0
        # for module_name, param in self.encoder.named_parameters():
        #     tot3 += param.abs().sum()
        # print("timbre_encoder:", tot1, "phoneme_decoder:", tot2, "encoder:", tot3)

        return output

    def _forward_discriminator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        mel2ph: Optional[Sequence] = None,
        frame2ph_token_ids: Optional[Sequence] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
    ) -> Dict[str, Any]:
        """Perform discriminator forward.

        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).
        """
        # setup
        batch_size = speech.size(0)
        speech = speech.unsqueeze(1)
        orig_speech = speech.clone()
        # feature_lengths = ((speech_lengths // 160 + 1) / 4).ceil().long() if speech_lengths is not None else None
        feature_lengths = ((speech_lengths // self.domain_conf.get("hop_length", 160) + 1) / self.time_ds_rate).ceil().long() if speech_lengths is not None else None
        feature_attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None
        feature_padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None

        codes = []
        # frames = self._encode(speech)
        encoded_frames = self._encode(speech)
        frames = [(frame.emb, frame.scale) for frame in encoded_frames]

        if self.timbre_encoder is not None:
            if self.timbre_encoder.config.transformed_speech_for_timbre_encoder:
                timbre_speech_lengths = speech_lengths
                transformed_speech = encoded_frames[0].transformed_speech
                transformed_speech = rearrange(transformed_speech.squeeze(1), "b f t -> b t f")
                timbre_encoder_output = self.forward_timbre_encoder(speech_lengths=timbre_speech_lengths, transformed_speech=transformed_speech)
            else:
                timbre_speech = speech
                timbre_speech_lengths = speech_lengths
                timbre_encoder_output = self.forward_timbre_encoder(speech=timbre_speech, speech_lengths=timbre_speech_lengths)

        for emb, scale in frames:
            feature_before_quantization = None
            if self.feature_extractor_before_quantization is not None:
                feature_before_quantization = emb.clone()
                emb = self.feature_extractor_before_quantization(feature_before_quantization)

            quant_in = emb.clone()

            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_before_quantizer is not None:
                quant_in = self.project["quantizer_in"](quant_in)

            if self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.encoder_output_minus_timbre_features:
                quant = self.encoder_output_minus_timbre_features(timbre_encoder_output, quant_in, feature_lengths)
                quant_in = quant

            if self.instance_norm_before_quantization or \
                (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization):
                quant_in = self.instance_norm_1d(quant_in)

            quant_out, indices, commit_loss, sub_quants = self.quantizer(quant_in)
            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_after_quantizer is not None:
                quant_out = self.project["quantizer_out"](quant_out)
            code_embs = quant_out
            # codes.append((code_embs, scale))
            codes.append([code_embs, scale])

            if self.timbre_extractor is not None:
                features = emb
                if self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.feature_minus_quant:
                    features = emb - quant_out

                timbre_extractor_output = self.forward_timbre_extractor(
                    features=features,
                    feature_attention_mask=feature_attention_mask,
                    feature_padding_mask=feature_padding_mask,
                )
            
        merged_timbre_features = None

        if self.timbre_encoder is not None:
            merged_timbre_features = self.merge_timbre_features(
                timbre_encoder_output, codes, feature_lengths,
            )
            if self.timbre_encoder.config.merge_with_decoder == "normal":
                codes[0][0] = merged_timbre_features.quant
                if self.feature_extractor_after_quantization is not None:
                    codes[0][0] = self.feature_extractor_after_quantization(merged_timbre_features.quant)
                recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
            elif self.timbre_encoder.config.merge_with_decoder == "conditional_layer_norm":
                recon_speech = self._decode(codes, condition_embedding=merged_timbre_features.timbre_feats)[:, :, :speech.shape[-1]]
        elif self.timbre_extractor is not None:
            merged_timbre_extractor_output = self.merge_timbre_extractor(
                timbre_extractor_output, codes,
                timbre_feature_attention_mask=feature_attention_mask,
                timbre_feature_lengths=feature_lengths,
            )
            if self.timbre_extractor.config.merge_with_decoder == "normal":
                codes[0][0] = merged_timbre_extractor_output.quant
                if self.feature_extractor_after_quantization is not None:
                    codes[0][0] = self.feature_extractor_after_quantization(merged_timbre_features.quant)
                recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
            elif self.timbre_extractor.config.merge_with_decoder == "conditional_layer_norm":
                recon_speech = self._decode(codes, condition_embedding=merged_timbre_extractor_output.timbre_feats)[:, :, :speech.shape[-1]]
        else:
            recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]

        # B: calculate discriminator outputs
        real, fake = orig_speech.clone(), recon_speech.detach()
        real_disc_outputs = self.discriminator(real)
        fake_disc_outputs = self.discriminator(fake)

        # C: calculate discriminator losses
        disc_losses = []
        for real_output, fake_output in zip(real_disc_outputs, fake_disc_outputs):
            real_logits, real_intermediates = real_output
            fake_logits, fake_intermediates = fake_output
            one_disc_loss = torch.mean(F.relu(1 - real_logits)) + torch.mean(F.relu(1+fake_logits))
            disc_losses.append(one_disc_loss)
        disc_loss = torch.stack(disc_losses).mean()

        # Optional D: phase invariant training
        pit_disc_loss = self.zero
        if self.phase_invariant_training and self.phaseaug is not None:
            real_aug = self.phaseaug.forward(speech).detach()
            aug_disc_outputs = self.discriminator(real_aug)
            pit_disc_losses = []
            for real_output, aug_output in zip(real_disc_outputs, aug_disc_outputs):
                real_logits, real_intermediates = real_output
                aug_logits, aug_intermediates = aug_output
                one_pit_disc_loss = F.l1_loss(real_logits, aug_logits)

                disc_feature_losses = []
                for i, (real_inter, aug_inter) in enumerate(zip(real_intermediates, aug_intermediates)):
                    if i >= self.feat_match_layer_start:
                        _loss = F.l1_loss(real_inter, aug_inter)
                        disc_feature_losses.append(_loss)
                disc_feature_loss = torch.stack(disc_feature_losses).mean()

                one_pit_disc_loss = (one_pit_disc_loss +
                                     disc_feature_loss * self.pit_feat_loss_weight)

                pit_disc_losses.append(one_pit_disc_loss)
            pit_disc_loss = torch.stack(pit_disc_losses).mean()

        # To avoid discriminator overpowers the generator, without this recon losses may not converge
        if self.training:
            loss_mask = (disc_loss > self.gen_loss).float()
            disc_loss = disc_loss * loss_mask
            pit_disc_loss = pit_disc_loss * loss_mask

            if disc_loss.item() > self.gen_loss:
                logging.info(f"Will update discriminator: forward_step={self.forward_step}, "
                             f"disc_loss={disc_loss.item():.4f}, gen_loss={self.gen_loss:.4f}")
        self.gen_loss = 0

        # D: whether to use gradient penalty loss
        loss = disc_loss + pit_disc_loss * self.pit_disc_loss_weight

        stats = dict(
            discriminator_total_loss=loss.item(),
            discriminator_loss=disc_loss.item(),
        )
        if self.phase_invariant_training and self.phaseaug is not None:
            stats["pit_disc_loss"] = pit_disc_loss
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
            "real": orig_speech,
            "fake": recon_speech,
        }

    def inference(
        self,
        speech: torch.Tensor,
        speech_lengths: Optional[torch.Tensor] = None,
        need_recon: bool = True,
        bit_width: int = None,
        use_scale: bool = True,
        exchange_timbre: Optional[bool] = False,
        without_timbre: Optional[bool] = False,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            speech (torch.Tensor): input speech
            need_recon (bool): whether to return recon speech
            bit_width (int): The excepted bandwidth
            use_scale (bool): whether to use scale

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (T_wav,).
                * code_indices (Tensor): quantized code indices (L)
                * code_embeddings (Tensor): quantized code embeddings (L, d).

        """
        codes = []
        code_idxs = []
        all_sub_quants = []
        feature_lengths = ((speech_lengths // self.domain_conf.get("hop_length", 160) + 1) / self.time_ds_rate).ceil().long() if speech_lengths is not None else None
        feature_attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None
        feature_padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None

        if speech.dim() == 2:
            speech = speech.unsqueeze(1)

        encoded_frames = self._encode(speech)
        frames = [(frame.emb, frame.scale) for frame in encoded_frames]

        if self.timbre_encoder is not None and need_recon and not without_timbre:
            if self.timbre_encoder.config.transformed_speech_for_timbre_encoder and not exchange_timbre:
                timbre_speech_lengths = speech_lengths
                transformed_speech = encoded_frames[0].transformed_speech
                transformed_speech = rearrange(transformed_speech.squeeze(1), "b f t -> b t f")
                timbre_encoder_output = self.forward_timbre_encoder(speech_lengths=timbre_speech_lengths, transformed_speech=transformed_speech)
            else:
                timbre_speech = speech
                timbre_speech_lengths = speech_lengths
                if exchange_timbre:
                    assert speech.shape[0] == 2
                    timbre_speech = speech.clone()
                    timbre_speech_lengths = speech_lengths.clone()
                    timbre_speech[[0, 1]] = timbre_speech[[1, 0]]
                    timbre_speech_lengths[[0, 1]] = timbre_speech_lengths[[1, 0]]
                    # feature_lengths[[0, 1]] = feature_lengths[[1, 0]]
                timbre_encoder_output = self.forward_timbre_encoder(speech=timbre_speech, speech_lengths=timbre_speech_lengths)

        for emb, scale in frames:
            bb, tt, device = emb.shape[0], emb.shape[1], emb.device

            feature_before_quantization = None
            if self.feature_extractor_before_quantization is not None:
                feature_before_quantization = emb.clone()
                emb = self.feature_extractor_before_quantization(feature_before_quantization)

            if self.bypass_quantizer:
                code_embs, indices, sub_quants = emb, torch.zeros(bb, tt, dtype=torch.long, device=device), torch.zeros_like(emb, device=device)
                if self.instance_norm_before_quantization:
                    code_embs = self.instance_norm_1d(code_embs)
            else:
                quant_in = emb.clone()

                if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_before_quantizer is not None:
                    quant_in = self.project["quantizer_in"](quant_in)

            if self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.encoder_output_minus_timbre_features:
                quant = self.encoder_output_minus_timbre_features(timbre_encoder_output, quant_in, feature_lengths)
                quant_in = quant

                if self.instance_norm_before_quantization or \
                    (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization):
                    quant_in = self.instance_norm_1d(quant_in)

            quant_out, indices, sub_quants = self.quantizer.inference(quant_in, bandwidth=bit_width)
            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_after_quantizer is not None:
                quant_out = self.project["quantizer_out"](quant_out)
            code_embs = quant_out
            # codes.append((code_embs, scale if use_scale else None))
            codes.append([code_embs, scale if use_scale else None])

            if self.timbre_extractor is not None:
                features = emb
                if self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.feature_minus_quant:
                    features = emb - quant_out
                timbre_extractor_output = self.forward_timbre_extractor(
                    features=features,
                    feature_attention_mask=feature_attention_mask,
                    feature_padding_mask=feature_padding_mask,
                )
                if exchange_timbre:
                    timbre_extractor_output.exchange_timbre()

            code_idxs.append(indices)
            all_sub_quants.append(sub_quants)
        recon_speech = None
        if need_recon:
            if not without_timbre and self.timbre_encoder is not None:
                merged_timbre_features = self.merge_timbre_features(
                    timbre_encoder_output, codes, feature_lengths,
                )
                if self.timbre_encoder.config.merge_with_decoder == "normal":
                    codes[0][0] = merged_timbre_features.quant
                    if self.feature_extractor_after_quantization is not None:
                        codes[0][0] = self.feature_extractor_after_quantization(merged_timbre_features.quant)
                    recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
                elif self.timbre_encoder.config.merge_with_decoder == "conditional_layer_norm":
                    recon_speech = self._decode(codes, condition_embedding=merged_timbre_features.timbre_feats)[:, :, :speech.shape[-1]]
            elif not without_timbre and self.timbre_extractor is not None:
                merged_timbre_extractor_output = self.merge_timbre_extractor(
                    timbre_extractor_output, codes,
                    timbre_feature_attention_mask=feature_attention_mask,
                    timbre_feature_lengths=feature_lengths,
                )
                if self.timbre_extractor.config.merge_with_decoder == "normal":
                    codes[0][0] = merged_timbre_extractor_output.quant
                    if self.feature_extractor_after_quantization is not None:
                        codes[0][0] = self.feature_extractor_after_quantization(merged_timbre_features.quant)
                    recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
                elif self.timbre_extractor.config.merge_with_decoder == "conditional_layer_norm":
                    recon_speech = self._decode(codes, condition_embedding=merged_timbre_extractor_output.timbre_feats)[:, :, :speech.shape[-1]]
            else:
                recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]

        retval = dict(
            recon_speech=recon_speech,
            code_indices=code_idxs,
            code_embeddings=codes,
            sub_quants=all_sub_quants
        )
        return retval

    def inference_encoding(
        self,
        speech: torch.Tensor,
        speech_lengths: Optional[torch.Tensor] = None,
        need_recon: bool = False,
        bit_width: int = None,
        use_scale: bool = True,
        exchange_timbre: Optional[bool] = False,
        without_timbre: Optional[bool] = False,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            speech (torch.Tensor): input speech
            need_recon (bool): whether to return recon speech
            bit_width (int): The excepted bandwidth
            use_scale (bool): whether to use scale

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (T_wav,).
                * code_indices (Tensor): quantized code indices (L)
                * code_embeddings (Tensor): quantized code embeddings (L, d).

        """
        codes = []
        code_idxs = []
        all_sub_quants = []
        feature_lengths = ((speech_lengths // self.domain_conf.get("hop_length", 160) + 1) / self.time_ds_rate).ceil().long() if speech_lengths is not None else None
        feature_attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None
        feature_padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None

        if speech.dim() == 2:
            speech = speech.unsqueeze(1)

        encoded_frames = self._encode(speech)
        frames = [(frame.emb, frame.scale) for frame in encoded_frames]

        if self.timbre_encoder is not None and need_recon and not without_timbre:
            if self.timbre_encoder.config.transformed_speech_for_timbre_encoder and not exchange_timbre:
                timbre_speech_lengths = speech_lengths
                transformed_speech = encoded_frames[0].transformed_speech
                transformed_speech = rearrange(transformed_speech.squeeze(1), "b f t -> b t f")
                timbre_encoder_output = self.forward_timbre_encoder(speech_lengths=timbre_speech_lengths, transformed_speech=transformed_speech)
            else:
                timbre_speech = speech
                timbre_speech_lengths = speech_lengths
                if exchange_timbre:
                    assert speech.shape[0] == 2
                    timbre_speech = speech.clone()
                    timbre_speech_lengths = speech_lengths.clone()
                    timbre_speech[[0, 1]] = timbre_speech[[1, 0]]
                    timbre_speech_lengths[[0, 1]] = timbre_speech_lengths[[1, 0]]
                    # feature_lengths[[0, 1]] = feature_lengths[[1, 0]]
                timbre_encoder_output = self.forward_timbre_encoder(speech=timbre_speech, speech_lengths=timbre_speech_lengths)

        for emb, scale in frames:
            feature_before_quantization = None
            if self.feature_extractor_before_quantization is not None:
                feature_before_quantization = emb.clone()
                emb = self.feature_extractor_before_quantization(feature_before_quantization)
            quant_in = emb.clone()

            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_before_quantizer is not None:
                quant_in = self.project["quantizer_in"](quant_in)

            if self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.encoder_output_minus_timbre_features:
                quant = self.encoder_output_minus_timbre_features(timbre_encoder_output, quant_in, feature_lengths)
                quant_in = quant

            if self.instance_norm_before_quantization or \
                (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization):
                quant_in = self.instance_norm_1d(quant_in)

            quant_out, indices, sub_quants = self.quantizer.inference(quant_in, bandwidth=bit_width)
            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_after_quantizer is not None:
                quant_out = self.project["quantizer_out"](quant_out)
            code_embs = quant_out
            # codes.append((code_embs, scale if use_scale else None))
            codes.append([code_embs, scale if use_scale else None])

            if self.timbre_extractor is not None:
                features = emb
                if self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.feature_minus_quant:
                    features = emb - quant_out
                timbre_extractor_output = self.forward_timbre_extractor(
                    features=features,
                    feature_attention_mask=feature_attention_mask,
                    feature_padding_mask=feature_padding_mask,
                )
                if exchange_timbre:
                    timbre_extractor_output.exchange_timbre()

            code_idxs.append(indices)
            all_sub_quants.append(sub_quants)
        recon_speech = None
        if need_recon:
            if not without_timbre and self.timbre_encoder is not None:
                merged_timbre_features = self.merge_timbre_features(
                    timbre_encoder_output, codes, feature_lengths,
                )
                if self.timbre_encoder.config.merge_with_decoder == "normal":
                    codes[0][0] = merged_timbre_features.quant
                    if self.feature_extractor_after_quantization is not None:
                        codes[0][0] = self.feature_extractor_after_quantization(merged_timbre_features.quant)
                    recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
                elif self.timbre_encoder.config.merge_with_decoder == "conditional_layer_norm":
                    recon_speech = self._decode(codes, condition_embedding=merged_timbre_features.timbre_feats)[:, :, :speech.shape[-1]]
            elif not without_timbre and self.timbre_extractor is not None:
                merged_timbre_extractor_output = self.merge_timbre_extractor(
                    timbre_extractor_output, codes,
                    timbre_feature_attention_mask=feature_attention_mask,
                    timbre_feature_lengths=feature_lengths,
                )
                if self.timbre_extractor.config.merge_with_decoder == "normal":
                    codes[0][0] = merged_timbre_extractor_output.quant
                    if self.feature_extractor_after_quantization is not None:
                        codes[0][0] = self.feature_extractor_after_quantization(merged_timbre_features.quant)
                    recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
                elif self.timbre_extractor.config.merge_with_decoder == "conditional_layer_norm":
                    recon_speech = self._decode(codes, condition_embedding=merged_timbre_extractor_output.timbre_feats)[:, :, :speech.shape[-1]]
            else:
                recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]

        retval = dict(
            recon_speech=recon_speech,
            code_indices=code_idxs,
            code_embeddings=codes,
            sub_quants=all_sub_quants
        )
        return retval
    
    def inference_encoding_withoutnorm(
            self,
            speech: torch.Tensor,
            need_recon: bool = False,
            bit_width: int = None,
            use_scale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            speech (torch.Tensor): input speech
            need_recon (bool): whether to return recon speech
            bit_width (int): The excepted bandwidth
            use_scale (bool): whether to use scale

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (T_wav,).
                * code_indices (Tensor): quantized code indices (L)
                * code_embeddings (Tensor): quantized code embeddings (L, d).

        """
        self.audio_normalize = False
        codes = []
        code_idxs = []
        all_sub_quants = []
        # feature_lengths = ((speech_lengths // 160 + 1) / 4).ceil().long() if speech_lengths is not None else None
        feature_lengths = ((speech_lengths // self.domain_conf.get("hop_length", 160) + 1) / self.time_ds_rate).ceil().long() if speech_lengths is not None else None
        if speech.dim() == 2:
            speech = speech.unsqueeze(1)
        if need_recon and self.timbre_encoder is not None:
            timbre_encoder_output = self.forward_timbre_encoder(speech=speech, speech_lengths=speech_lengths)
        # frames = self._encode(speech)
        encoded_frames = self._encode(speech)
        frames = [(frame.emb, frame.scale) for frame in encoded_frames]
        for emb, scale in frames:
            quant_in = emb.clone()
            if self.instance_norm_before_quantization:
                quant_in = self.instance_norm_1d(quant_in)
            quant_out, indices, sub_quants = self.quantizer.inference(quant_in, bandwidth=bit_width)
            code_embs = quant_out
            # codes.append((code_embs, scale if use_scale else None))
            codes.append([code_embs, scale if use_scale else None])
            code_idxs.append(indices)
            all_sub_quants.append(sub_quants)
        recon_speech = None
        if need_recon:
            if self.timbre_encoder is not None:
                merged_timbre_features = self.merge_timbre_features(timbre_encoder_output, codes, feature_lengths)
                codes = merged_timbre_features.quants
            recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
        retval = dict(
            recon_speech=recon_speech,
            code_indices=code_idxs,
            code_embeddings=codes,
            sub_quants=all_sub_quants
        )
        return retval

    def inference_decoding(
            self,
            token_idx: torch.Tensor,
            need_recon: bool = True,
            bit_width: int = None,
            use_scale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            token_idx (torch.Tensor): input token indices, B x T x n_q
            need_recon (bool): whether to return recon speech
            bit_width (int): The excepted bandwidth
            use_scale (bool): whether to use scale

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (T_wav,).
                * code_indices (Tensor): quantized code indices (L)
                * code_embeddings (Tensor): quantized code embeddings (L, d).

        """
        codes = []
        token_idx = token_idx.permute(2, 0, 1).unsqueeze(0)
        for tokens in token_idx:
            code_embs = self.quantizer.decode(tokens)
            # codes.append((code_embs.transpose(1, 2), None))
            codes.append([code_embs.transpose(1, 2), None])
        recon_speech = None
        if need_recon:
            recon_speech = self._decode(codes)
        retval = dict(
            recon_speech=recon_speech,
            code_indices=None,
            code_embeddings=codes,
            sub_quants=None
        )
        return retval

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass
