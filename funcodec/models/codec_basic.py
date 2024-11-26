# Copyright 2023 Zhihao Du
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-End Speech Tokenizer SoundStream."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple, List, Union
import typing as tp
import torch
import torchaudio
import numpy as np
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types
from funcodec.train.abs_gan_espnet_model import AbsGANESPnetModel
from funcodec.torch_utils.device_funcs import force_gatherable
from librosa.filters import mel as librosa_mel_fn
from funcodec.losses.label_smoothing_loss import LabelSmoothingLoss
from funcodec.layers.mask_along_axis import MaskAlongAxisVariableMaxWidth
import logging
from funcodec.utils.hinter import hint_once

from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .utils import lengths_to_padding_mask, lengths_to_attention_mask
from .timbre_encoder.strategy import TimbreStrategy
from .timbre_encoder.modeling_timbre_encoder import TimbreEncoderPreTrainedModel, TimbreEncoderOutput, mean_pooling
from .timbre_encoder.modeling_qformer import QFormerModel
from .timbre_encoder import modeling_phaseaug
from .timbre_encoder.modeling_praat import PraatTransformer
from .contrastive_encoder.configuration_contrastive_encoder import ContrastiveEncoderConfig
from .contrastive_encoder.modeling_frame_contrastive_encoder import BaseFrameContrastiveEncoder, FrameContrastiveEncoderOutput
from .contrastive_encoder.modeling_contrastive_encoder import ContrastiveEncoderPreTrainedModel, ContrastiveEncoderOutput
from .contrastive_encoder.modeling_frame_contrastive_encoder import FrameContrastiveEncoder, FrameContrastiveEncoderOutput
from .timbre_encoder.modeling_timbre_extractor import BaseTimbreExtractorModel, TimbreExtractorOutput
from .contrastive_encoder.modeling_speaker_contrastive_encoder import SpeakerContrastiveEncoder, SpeakerContrastiveEncoderOutput
from funcodec.models.timbre_encoder.modeling_timbre_encoder import StyleAdaptiveOutput, StyleAdaptiveLayerNorm
from .contrastive_encoder.modeling_speaker_predict_encoder import SpeakerPredictEncoderBaseModel, SpeakerPredictEncoderOutput
from .contrastive_encoder.modeling_phoneme_decoder import PhonemeDecoderPreTrainedModel, PhonemeDecoderOutput
from .retrain_model.configuration_retrain_model import RetrainModelConfig
from .retrain_model.modeling_retrain_model import RetrainModelPreTrainedModel
from funcodec.models.encoding_path.encoding_path import EncodingPathConfig, BaseEncodingPathModel

from funcodec.models.consistency.strategy import ConsistencyStrategy
from funcodec.models.consistency.perturb_encoder import(
    PhaseAugConfig,
    PerturbEncoderConfig,
    PerturbEncoder,
)
from funcodec.models.consistency.slice_encoder import (
    SliceEncoder,
    SliceInterval,
    SliceEncoderOutput,
)
from funcodec.models.semantic_distill.semantic_distill import (
    SemanticDistillConfig,
    SemanticDistillOutput,
    SemanticDistillBaseModel,
)


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
        device='cuda'
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).cuda().float()
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


@dataclass
class EncodedFrameOutput(ModelOutput):
    emb: Optional[torch.Tensor] = None
    scale: Optional[Union[Tuple, torch.Tensor]] = None
    transformed_speech: Optional[torch.Tensor] = None


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
    style_output: Optional[torch.Tensor] = None


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
    hop_length: Optional[int] = 320,
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
            max_lens = (batch["speech_lengths"] // hop_length).ceil().long()
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
class MergedTimbreExtractorOutput(ModelOutput):
    ca_output: Optional[torch.Tensor] = None
    ca_weight: Optional[torch.Tensor] = None
    timbre_feats: Optional[torch.Tensor] = None
    quant: Optional[torch.Tensor] = None


class Encodec(AbsGANESPnetModel):
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
        audio_normalize: bool = True,
        segment_dur: Optional[float] = 1.0,
        overlap_ratio: Optional[float] = 0.01,
        use_power_spec_loss: Optional[bool] = False,
        context_loss_weight: Optional[float] = 0.0,
        context_loss_conf: Optional[Dict] = None,
        bypass_quantizer: bool = False,
        codec_domain: str = "time",
        domain_conf: Optional[Dict] = {},
        timbre_strategy: Optional[Union[Dict, TimbreStrategy]] = None,
        retrain_model_config: Optional[Union[Dict, RetrainModelConfig]] = None,
        consistency_strategy: Optional[Union[Dict, ConsistencyStrategy]] = None,
        semantic_distill_config: Optional[Union[Dict, SemanticDistillConfig]] = None,
        instance_norm_before_quantization: Optional[bool] = None,
        l2_norm_before_quantization: Optional[bool] = None,
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

        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        # Used by task and trainer
        self.gen_model_list = [self.encoder, self.quantizer, self.decoder]
        self.discriminator = discriminator
        self.bypass_quantizer = bypass_quantizer
        self.codec_domain = codec_domain
        if codec_domain == "stft":
            self.stft_fun = torchaudio.transforms.Spectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                power=None,
            )
            self.inverse_fun = torchaudio.transforms.InverseSpectrogram(
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

        self.context_loss_weight = context_loss_weight
        if self.context_loss_weight > 0 and context_loss_conf is not None:
            self.use_quant_for_context = context_loss_conf.get("use_quant_for_context", False)
            self.mask_pred_weight = context_loss_conf.get("mask_pred_weight", None)
            self.context_model = self.build_context_model(
                context_loss_conf["model"],
                context_loss_conf["model_conf"]
            )
            # add context model to generator for optimizer.
            self.gen_model_list.append(self.context_model)
            self.context_masker = self.build_context_mask(context_loss_conf.get("mask_conf", None))
            self.ce_loss_weight = context_loss_conf.get("ce_loss_weight", 0.0)
            self.context_lm_weight = context_loss_conf.get("lm_loss_weight", 0.0)
            self.contrast_loss_weight = context_loss_conf.get("contrast_loss_weight", 0.0)
            self.context_ce_criterion = nn.CrossEntropyLoss(reduction="none")

        # for timbre disentangle
        self.odim = odim
        self.target_sample_hz = target_sample_hz
        self.init_timbre_encoder(timbre_strategy)
        self.time_ds_rate = np.prod(self.encoder.ratios)
        # from speech_lengths get frame_lengths:
        # frame_lengths = (speech_lengths // self.time_ds_rate).ceil().long()

        self.instance_norm_before_quantization = instance_norm_before_quantization
        if self.instance_norm_before_quantization or \
            (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization) or \
            (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_after_encoder):
            self._instance_norm_1d = nn.InstanceNorm1d(odim, affine=False)
            self.gen_model_list.append(self._instance_norm_1d)
            self.instance_norm_1d = lambda x: self._instance_norm_1d(x.transpose(1, 2)).transpose(1, 2)

        self.l2_norm_before_quantization = l2_norm_before_quantization
        
        self.init_consistency_encoder(consistency_strategy)
        self.init_semantic_distill(semantic_distill_config)

        self.prepare_retrain(retrain_model_config)
        self.eps = 1e-5

    def get_frame_feature_lengths(self, speech_lengths: torch.LongTensor):
        return (speech_lengths / self.time_ds_rate).ceil().long()

    def init_consistency_encoder(self, consistency_strategy: Optional[Union[Dict, ConsistencyStrategy]] = None):
        if isinstance(consistency_strategy, dict):
            consistency_strategy = ConsistencyStrategy(**consistency_strategy)
        self.consistency_strategy = consistency_strategy
        if consistency_strategy is not None and consistency_strategy.perturb_encoder_config is not None:
            self.perturb_encoder = PerturbEncoder(config=consistency_strategy.perturb_encoder_config)
            self.gen_model_list.append(self.perturb_encoder)
        else:
            self.perturb_encoder = None
        if consistency_strategy is not None and consistency_strategy.slice_encoder_config is not None:
            self.slice_encoder = SliceEncoder(config=consistency_strategy.slice_encoder_config)
            self.gen_model_list.append(self.slice_encoder)
        else:
            self.slice_encoder = None

    def init_semantic_distill(self, semantic_distill_config: Optional[Union[Dict, SemanticDistillConfig]] = None):
        if isinstance(semantic_distill_config, dict):
            semantic_distill_config = SemanticDistillConfig(**semantic_distill_config)
        self.semantic_distill_config = semantic_distill_config
        if semantic_distill_config is not None:
            self.semantic_encoder = SemanticDistillBaseModel.build_model(config=semantic_distill_config)
            self.gen_model_list.append(self.semantic_encoder)
        else:
            self.semantic_encoder = None

    def forward_encoder_quantizer(
        self,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        speech = speech.unsqueeze(1)
        orig_speech = speech.clone()
        codes = []
        code_indices = []
        quant_in_list = []
        sub_quants_list = []
        encoded_frames = self._encode(speech) # offset list: [offset1]
        frames = [(frame.emb, frame.scale, frame.transformed_speech) for frame in encoded_frames]
        for emb, scale, transformed_speech in frames:
            quant_in = emb.clone()
            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_before_quantizer is not None:
                quant_in = self.project["quantizer_in"](quant_in)
            if self.instance_norm_before_quantization or \
                (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization):
                quant_in = self.instance_norm_1d(quant_in)
            if self.l2_norm_before_quantization:
                quant_in = F.normalize(quant_in, dim=-1, eps=self.eps)
            quant_in_list.append(quant_in)
            # quant_out, indices, commit_loss, sub_quants = self.quantizer(quant_in)
            quantizer_output = self.quantizer(quant_in)
            quant_out, indices, commit_loss, sub_quants = quantizer_output["x"], quantizer_output["indices"], quantizer_output["commit_loss"], quantizer_output["sub_quants"]
            sub_quants_list.append(sub_quants) # [num_rvq, batch_size, dim, seq_len]
            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_after_quantizer is not None:
                quant_out = self.project["quantizer_out"](quant_out)
            code_embs = quant_out
            codes.append([code_embs, scale])
            code_indices.append(indices)
        output = {
            "real": orig_speech,
            "fake": None,
            "code_indices": code_indices, # code index
            "quants": [embed for (embed, scale)in codes], # [[batch_size, time_step, dim]]
            "encoder_output": frames[0][0],
            "decoder_input": codes[0][0],
            "quants_in": quant_in_list, 
            "sub_quants_list": sub_quants_list,
            "transformed_speech": [frame.transformed_speech for frame in encoded_frames],
        }
        output = {param: value for param, value in output.items() if value is not None}
        return output

    def forward_slice_encoder(
        self,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        quant_in: Optional[torch.FloatTensor] = None,
        quant_out: Optional[torch.FloatTensor] = None,
        sub_quants: Optional[torch.FloatTensor] = None,
        code_indices: Optional[torch.FloatTensor] = None,
        feature_lengths: Optional[torch.LongTensor] = None,
        slice_interval: Optional[SliceInterval] = None,
        codebook: Optional[torch.FloatTensor] = None,
        forward_step: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        if slice_interval is None:
            slice_interval = self.slice_encoder.sample_split_intervals(feature_lengths)
        slice_speech_input = self.slice_encoder.gather_speech(
            speech=speech,
            speech_lengths=speech_lengths,
            slice_interval=slice_interval
        )
        if self.perturb_encoder is not None and self.perturb_encoder.config.perturb_slice_speech:
            perturbed_slice_speech = self.perturb_encoder(slice_speech_input["slice_speech"].unsqueeze(1))
            perturbed_slice_speech = perturbed_slice_speech.squeeze(1)
            speech_min_length = min(speech.shape[-1], perturbed_slice_speech.shape[-1])
            # print(f"666 perturb_slice_speech: speech = {speech.shape}, perturbed_slice_speech = {perturbed_slice_speech.shape}")
            if perturbed_slice_speech.shape[-1] <= speech_min_length:
                slice_speech_input["slice_speech"][:, :perturbed_slice_speech.shape[-1]] = perturbed_slice_speech
            else:
                slice_speech_input["slice_speech"] = perturbed_slice_speech[:, :speech_min_length]
            slice_speech_input["slice_speech"] = slice_speech_input["slice_speech"].clone()

        slice_feature_input = self.forward_encoder_quantizer(
            speech=slice_speech_input["slice_speech"],
            speech_lengths=slice_speech_input["speech_lengths"],
        )
        return self.slice_encoder(
            speech=speech,
            slice_speech=slice_speech_input["slice_speech"],
            speech_lengths=speech_lengths,
            quant_in=quant_in,
            slice_quant_in=slice_feature_input["quants_in"][0],
            quant_out=quant_out,
            slice_quant_out=slice_feature_input["quants"][0],
            sub_quants=sub_quants,
            slice_sub_quants=slice_feature_input["sub_quants_list"][0].transpose(2, 3),
            code_indices=code_indices,
            slice_code_indices=slice_feature_input["code_indices"][0],
            feature_lengths=feature_lengths,
            slice_feature_lengths=slice_interval["split_interval_lengths"],
            slice_interval=slice_interval,
            codebook=codebook,
            forward_step=forward_step,
        )


    def init_timbre_encoder(self, timbre_strategy: Optional[TimbreStrategy] = None):
        self.ada_in = None
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
            elif self.timbre_strategy.merge_embed == "ada_in":
                self.ada_in = StyleAdaptiveLayerNorm(normalized_shape=self.timbre_strategy.timbre_encoder_config.embed_dim)
                self.gen_model_list.append(self.ada_in)
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

    def build_context_model(self, model_type: str, model_conf: Dict):
        if model_type == "lstm":
            from funcodec.models.encoder.rnn_encoder import RNNEncoder
            model = RNNEncoder(
                input_size=self.encoder.output_size(),
                rnn_type=model_conf.get("rnn_type", "lstm"),
                bidirectional=model_conf.get("bidirectional", True),
                use_projection=model_conf.get("use_projection", True),
                num_layers=model_conf.get("num_layers", 4),
                hidden_size=model_conf.get("hidden_size", 512),
                output_size=model_conf.get("output_size", self.encoder.output_size()),
                dropout=model_conf.get("dropout", 0.0),
                subsample=model_conf.get("subsample", [1, 1, 1, 1]),
            )
        elif model_type == "transformer":
            from funcodec.models.encoder.transformer_encoder import TransformerEncoder
            model = TransformerEncoder(
                input_size=self.encoder.output_size(),
                output_size=model_conf.get("output_size", self.encoder.output_size()),
                attention_heads=model_conf.get("attention_heads", 8),
                linear_units=model_conf.get("linear_units", 2048),
                num_blocks=model_conf.get("num_blocks", 6),
                dropout_rate=model_conf.get("dropout_rate", 0.0),
                positional_dropout_rate=model_conf.get("positional_dropout_rate", 0.0),
                attention_dropout_rate=model_conf.get("attention_dropout_rate", 0.0),
                input_layer=model_conf.get("input_layer", "linear"),
                causal_mode=model_conf.get("causal_mode", "causal"),
            )
        else:
            raise TypeError(f"Unknown model type {model_type}, only support lstm and transformer.")

        return model

    def build_context_mask(self, args):
        # input should in (Batch, Length, Freq)
        time_mask = MaskAlongAxisVariableMaxWidth(
            dim="time",
            mask_width_ratio_range=args["mask_ratio_range"],
            num_mask=args["num_mask"],
            replace_with_zero=True,
        )
        return time_mask

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

    def forward_timbre_encoder(
        self,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        transformed_speech: Optional[torch.FloatTensor] = None,
    ) -> TimbreEncoderOutput:
        if self.timbre_strategy is not None and self.timbre_strategy.timbre_encoder_config is not None:
            if self.timbre_strategy.timbre_encoder_config.input_type == "wav":
                input_speech = speech if transformed_speech is None else transformed_speech
                timbre_encoder_output = self.timbre_encoder(wavs=input_speech, wav_lens=speech_lengths)
                # timbre_padding_mask = lengths_to_padding_mask(speech_lengths)
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
            
            if self.timbre_strategy.merge_embed == "ada_in":
                timbre_feats = timbre_encoder_output.last_hidden_state
                timbre_feature_attention_mask = timbre_encoder_output.padding_mask != 1 if timbre_encoder_output.padding_mask is not None else None # [2, 1671]
                timbre_feats = mean_pooling(timbre_feats, timbre_feature_attention_mask)
                if self.timbre_strategy.timbre_encoder_config.merge_with_decoder == "normal":
                    timbre_feats = timbre_feats.unsqueeze(1)
                style_output = self.ada_in.forward_statistic(timbre_feats)
                timbre_encoder_output = TimbreEncoderOutput(
                    style_output=style_output,
                    **timbre_encoder_output
                )

            return timbre_encoder_output

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
        if self.timbre_strategy.merge_embed in ("mean_pooling", "ada_in"):
            timbre_feats = mean_pooling(timbre_feats, timbre_feature_attention_mask)
            merged_timbre_output["timbre_feats"] = timbre_feats
            if self.timbre_strategy.timbre_encoder_config.merge_with_decoder == "normal":
                timbre_feats = timbre_feats.unsqueeze(1)
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
            elif self.timbre_strategy.merge_embed in ("add", "mean_pooling"):
                # print(666, timbre_feats.sum(), quant.sum())
                # print(666, timbre_feats.shape, quant.shape)
                quant = timbre_feats + quant
            elif self.timbre_strategy.merge_embed == "ada_in":
                style_output = timbre_encoder_output.style_output
                if style_output is None:
                    style_output = self.ada_in(quant, timbre_feats)
                else:
                    # print(666, "style_output", style_output)
                    style_output = self.ada_in(quant, timbre_feats, gamma=style_output.gamma, beta=style_output.beta)
                merged_timbre_output["style_output"] = style_output
                quant = style_output.out
            # quants[0][0] = quant
            merged_timbre_output["quant"] = quant
        return MergedTimbreOutput(**merged_timbre_output)

    def forward_timbre_extractor(
        self,
        features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.BoolTensor] = None,
        feature_padding_mask: Optional[torch.BoolTensor] = None,
        half_speech_contrastive: Optional[bool] = False,
        speech_quant_out_contrastive: Optional[bool] = False,
        quant_out: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> TimbreExtractorOutput:
        return self.timbre_extractor(
            hidden_states=features,
            attention_mask=feature_attention_mask,
            padding_mask=feature_padding_mask,
            half_speech_contrastive=half_speech_contrastive,
            speech_quant_out_contrastive=speech_quant_out_contrastive,
            quant_out=quant_out,
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
        if self.timbre_strategy.timbre_extractor_config.merge_embed == "mean_pooling":
            if timbre_extractor_output.merged_output is None:
                timbre_feats = mean_pooling(timbre_extractor_output.last_hidden_state, timbre_feature_attention_mask)
            else:
                timbre_feats = timbre_extractor_output.merged_output
            merged_timbre_extractor_output["timbre_feats"] = timbre_feats # [batch_size, dim]
            if self.timbre_strategy.timbre_extractor_config.merge_with_quant_out in ("add", "linear"):
                # print(666, "forward_merge_extractor", timbre_feats.shape, quant.shape, timbre_feature_attention_mask.shape)
                timbre_feats = timbre_feats[:, None, :].expand(-1, quant.shape[1], -1)
                # timbre_feats = torch.where(timbre_feature_attention_mask, timbre_feats, 0.0)
                timbre_feats = timbre_feats * timbre_feature_attention_mask.unsqueeze(-1)
                if self.timbre_strategy.timbre_extractor_config.merge_with_quant_out == "add":
                    quant = quant + timbre_feats
                elif self.timbre_strategy.timbre_extractor_config.merge_with_quant_out == "linear":
                    # print(f"111 timbre_feats", timbre_feats.mean(1).mean(1), timbre_feature_attention_mask.sum(1))
                    # print(f"111 merge_with_quant_out = linear", quant.mean(1).mean(1))
                    quant = self.timbre_extractor.quant_proj(torch.cat([quant, timbre_feats], dim=-1))
                    # print(f"222 merge_with_quant_out = linear", quant.mean(1).mean(1))
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
            if self.timbre_strategy.timbre_encoder_config.input_type == "mel":
                get_feature_lengths_from_speech_lengths = lambda x: x // self.mel_transform.hop_length + 1
            elif self.timbre_strategy.timbre_encoder_config.input_type == "mag":
                # not supported
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
        batch = preprocess_batch(batch, self.time_ds_rate)
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

    def _encode(self, x: torch.Tensor) -> tp.List[EncodedFrameOutput]:
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

        encoded_frames = []
        # print("length:", length, "stride:", stride)
        for offset in range(0, length, stride):
            # print("start:", offset, "end:", offset + segment_length)
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrameOutput:
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

        if self.codec_domain == "stft":
            x_complex = self.stft_fun(x.squeeze(1))
            x = torch.cat([x_complex.real, x_complex.imag], dim=1)
        emb = self.encoder(x)

        # return emb, scale
        return EncodedFrameOutput(emb=emb, scale=scale, transformed_speech=x)

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
        if self.codec_domain == "stft":
            out_list = torch.split(out, out.shape[1]//2, dim=1)
            out = torch.complex(out_list[0], out_list[1])
            out = self.inverse_fun(out).unsqueeze(1)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def _context_lm_loss(self, inputs, ilens, code_emb, labels):
        # inputs BxTxD
        padded_inputs = F.pad(inputs, [0, 0, 1, 0, 0, 0])
        outs = self.context_model(padded_inputs[:, :-1, :], ilens)[0]
        dist = -(
                outs.pow(2).sum(2, keepdim=True)
                - 2 * outs @ code_emb
                + code_emb.pow(2).sum(1, keepdim=True)
        )
        # for numerically stable
        dist = dist - torch.max(dist, dim=-1, keepdim=True).values.detach()
        acc = (torch.argmax(dist, dim=-1) == labels).sum() / labels.numel()

        context_ce_loss = self.context_ce_criterion(dist.transpose(1, 2), labels)
        return context_ce_loss, acc

    def _cal_context_loss(self, enc_out, indices, sub_quants, quant_idx=0):
        bb, tt, _ = enc_out.shape
        index = indices[quant_idx]
        quant = sub_quants[quant_idx].transpose(1, 2)
        ilens = torch.ones((bb,)).to(enc_out.device).long() * tt
        code_emb = self.quantizer.rq.model.embed[quant_idx].t().unsqueeze(0)  # 1xDxN

        # Pass-Through-Estimator
        if self.use_quant_for_context:
            enc_out = enc_out + (quant - enc_out).detach()  # BxTxD

        if hasattr(self, "context_lm_weight") and self.context_lm_weight > 0:
            context_lm_loss, pred_acc = self._context_lm_loss(
                inputs=enc_out,
                ilens=ilens,
                code_emb=code_emb,
                labels=index
            )
            context_lm_loss = context_lm_loss.sum() / (bb * tt)
            return context_lm_loss * self.context_lm_weight, pred_acc

        # loss_mask: (Batch, Length, 1)
        masked_emb, _, loss_mask = self.context_masker(enc_out, return_mask=True)
        outs = self.context_model(masked_emb, ilens)[0]
        # dist: B x T x N
        dist = -(
                outs.pow(2).sum(2, keepdim=True)
                - 2 * outs @ code_emb
                + code_emb.pow(2).sum(1, keepdim=True)
        )
        # for numerically stable
        dist = dist - torch.max(dist, dim=-1, keepdim=True).values.detach()
        pred_acc = (torch.argmax(dist, dim=-1) == index).sum() / index.numel()

        # calculate  HuBert-style Masked Prediction Loss
        context_ce_loss = self.context_ce_criterion(dist.transpose(1, 2), index)
        if self.mask_pred_weight is None:
            context_ce_loss = context_ce_loss.sum() / (bb * tt)
        else:
            loss_mask = loss_mask.squeeze(2)
            masked_loss = (context_ce_loss * loss_mask).sum() / max(loss_mask.sum(), 1e-12)
            unmasked_loss = (context_ce_loss * (~loss_mask)).sum() / max((~loss_mask).sum(), 1e-12)
            context_ce_loss = masked_loss * self.mask_pred_weight + unmasked_loss * (1-self.mask_pred_weight)
        return context_ce_loss * self.ce_loss_weight, pred_acc

    def _forward_generator(
        self,
        speech: torch.FloatTensor,
        speech_lengths: torch.LongTensor,
        bandwidth: Optional[int] = None,
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
        codebook_ids: Optional[Sequence] = None,
        exchange_codebook_ids: Optional[bool] = None,
        first_layer_features: Optional[torch.Tensor] = None,
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

        # if self.perturb_encoder is None or self.perturb_encoder.config.perturb_all_speech:
        feature_lengths = self.get_frame_feature_lengths(speech_lengths)
        feature_attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None
        feature_padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None

        l1Loss = torch.nn.L1Loss(reduction='mean')
        l2Loss = torch.nn.MSELoss(reduction='mean')
        commit_losses = []
        enc_quant_losses = []
        context_loss = self.zero
        codes = []
        code_indices = [] # offset, num_codebooks, batch_size, timestep. offset is usually set to 1.

        if self.perturb_encoder is not None and self.perturb_encoder.config.perturb_all_speech:
            speech = self.perturb_encoder(speech)
            speech_min_length = min(speech.shape[-1], orig_speech.shape[-1])
            orig_speech = speech[:, :speech_min_length]
            feature_lengths = self.get_frame_feature_lengths(speech_lengths)
            feature_attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None
            feature_padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None

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
        sub_quants_list = []
        timbre_extractor_output = None
        merged_timbre_extractor_output = None

        context_pred_acc = []
        quant_in_list = []
        sub_quants_list = []
        dist_list = []
        # for emb, scale in frames:
        # for emb, scale, transformed_speech in frames:
        for i, (emb, scale, transformed_speech) in enumerate(frames):
            # if self.ada_in is not None:
            #     emb = (emb - timbre_encoder_output.style_output.beta) / (timbre_encoder_output.style_output.gamma + self.eps)
            #     emb = self.ada_in.norm(emb)

            quant_in = emb.clone()
            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_before_quantizer is not None:
                quant_in = self.project["quantizer_in"](quant_in)
            if self.instance_norm_before_quantization or \
                (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization):
                quant_in = self.instance_norm_1d(quant_in)
            if self.l2_norm_before_quantization:
                quant_in = F.normalize(quant_in, dim=-1, eps=self.eps)
                # print(666, "l2_norm_before_quantization")

            if self.semantic_encoder is not None:
                semantic_output = self.semantic_encoder(speech=speech, speech_lengths=speech_lengths, codec_seq_len=emb.shape[1])
                first_layer_features = semantic_output.last_hidden_state
            # else:
            #     first_layer_features = None

            quant_in_list.append(quant_in)
            # quant_out, indices, commit_loss, sub_quants = self.quantizer(quant_in, bandwidth=bandwidth)
            quantizer_output = self.quantizer(quant_in, bandwidth=bandwidth, first_layer_features=first_layer_features)
            quant_out, indices, commit_loss, sub_quants = quantizer_output["x"], quantizer_output["indices"], quantizer_output["commit_loss"], quantizer_output["sub_quants"]
            sub_quants_list.append(sub_quants)
            dist_list.append(quantizer_output["dist"])

            if num_codebooks:
                # print(666, sub_quants.shape)
                # sub_quants: [num_codebooks, batch_size, dim, seq_len]
                quant_out = sub_quants[:num_codebooks].sum(0).transpose(1, 2)
                print(f"only use {num_codebooks} num_codebooks")
            if exchange_codebooks:
                quant_out0 = sub_quants[0].transpose(1, 2)
                quant_out0[[0, 1]] = quant_out0[[1, 0]]
                quant_out1 = sub_quants[1].transpose(1, 2)
                quant_out = quant_out0 + quant_out1
            if exchange_codebook_ids is not None:
                _quant_out = torch.zeros_like(quant_out, device=quant_out.device)
                for codebook_idx in range(sub_quants.shape[0]):
                    if codebook_idx in exchange_codebook_ids:
                        print(f"exchange {codebook_idx} codebook")
                        _quant_out += sub_quants[codebook_idx].transpose(1, 2)[[1, 0]]
                    else:
                        _quant_out += sub_quants[codebook_idx].transpose(1, 2)
                quant_out = _quant_out
            if codebook_ids is not None:
                _quant_out = torch.zeros_like(quant_out, device=quant_out.device)
                for codebook_idx in codebook_ids:
                    _quant_out += sub_quants[codebook_idx].transpose(1, 2)
                quant_out = _quant_out
                print(f"only use {codebook_ids} codebooks")
            sub_quants_list.append(sub_quants) # [num_rvq, batch_size, dim, seq_len]

            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_after_quantizer is not None:
                quant_out = self.project["quantizer_out"](quant_out)

            code_embs = quant_out
            # qv = self.quantizer.forward(emb, self.sample_rate, self.bandwidth)
            commit_losses.append(commit_loss)
            enc_quant_losses.append(l2Loss(quant_out, quant_in) ** 2)
            # codes.append((code_embs, scale))
            codes.append([code_embs, scale])
            code_indices.append(indices)
            if self.context_loss_weight > 0:
                _loss, _pred_acc = self._cal_context_loss(emb, indices, sub_quants, quant_idx=0)
                context_loss = context_loss + _loss
                context_pred_acc.append(_pred_acc)

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
                    half_speech_contrastive=self.timbre_extractor.config.half_speech_contrastive,
                    speech_quant_out_contrastive=self.timbre_extractor.config.speech_quant_out_contrastive,
                    quant_out=quant_out,
                    # quant_out=emb + (quant_out - emb).detach(),
                )
                if exchange_timbre:
                    timbre_extractor_output.exchange_timbre()

            if self.contrastive_encoder is not None:
                vq_idx = 0
                # print(666, quant_in.shape, quant_out.shape, feature_lengths)
                contrastive_encoder_output = self.forward_contrastive_encoder(
                    encoder_features=quant_in,
                    # quantized_features=quant_out,
                    # encoder_features=None,
                    quantized_features=sub_quants[vq_idx].transpose(1, 2),
                    feature_lengths=feature_lengths,
                    mel2ph=mel2ph,
                )
                contrastive_loss += contrastive_encoder_output.loss
                contrastive_encoder_output_list.append(contrastive_encoder_output)
            if self.frame_contrastive_encoder is not None:
                # print(666, "frame_contrastive_encoder")
                vq_idx = 0
                perturbed_quantizer_output = self.quantizer(perturbed_frames[i][0])
                perturbed_quant_out, perturbed_indices, perturbed_commit_loss, perturbed_sub_quants = perturbed_quantizer_output["x"], perturbed_quantizer_output["indices"], perturbed_quantizer_output["commit_loss"], perturbed_quantizer_output["sub_quants"]
                

                encoder_features = encoded_frames[0].emb
                encoder_perturbed_features = perturbed_encoded_frames[0].emb
                # quantizer_features = sub_quants[vq_idx].transpose(1, 2)
                # quantizer_perturbed_features = perturbed_sub_quants[vq_idx].transpose(1, 2)
                quantizer_features = quant_out
                quantizer_perturbed_features = perturbed_quant_out
                quantizer_features = encoder_features + (quant_out - encoder_features).detach()
                quantizer_perturbed_features = encoder_perturbed_features + (perturbed_quant_out - encoder_perturbed_features).detach()
                # emb + (quant_out - emb).detach()

                frame_contrastive_encoder_output = self.forward_frame_contrastive_encoder(
                    encoder_features=encoder_features,
                    encoder_perturbed_features=encoder_perturbed_features,
                    quantizer_features=quantizer_features,
                    quantizer_perturbed_features=quantizer_perturbed_features,
                    feature_lengths=feature_lengths,
                    mel2ph=mel2ph,
                    code_indices=indices[vq_idx],
                    perturbed_code_indices=perturbed_indices[vq_idx],
                )
            else:
                frame_contrastive_encoder_output = None

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

        if self.slice_encoder is not None:
            slice_encoder_output = self.forward_slice_encoder(
                # speech=speech.squeeze(1),
                speech=orig_speech.squeeze(1),
                speech_lengths=speech_lengths,
                quant_in=quant_in_list[0],
                quant_out=codes[0][0],
                sub_quants=sub_quants_list[0].transpose(2, 3),
                code_indices=code_indices[0],
                feature_lengths=feature_lengths,
                codebook=self.quantizer.rq.model.embed,
                forward_step=self.forward_step,
            )
        else:
            slice_encoder_output = None

        merged_timbre_features = None
        if not without_timbre and self.timbre_encoder is not None:
            merged_timbre_features = self.merge_timbre_features(
                timbre_encoder_output, codes, feature_lengths,
            )
            if self.timbre_encoder.config.merge_with_decoder == "normal":
                # print(111, merged_timbre_features.quant.sum(), codes[0][0].sum())
                codes[0][0] = merged_timbre_features.quant
                # print(222, merged_timbre_features.quant.sum(), codes[0][0].sum())
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
                recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
                # print(f"333 merge_with_decoder = normal", codes[0][0].mean(1).mean(1))
            elif self.timbre_extractor.config.merge_with_decoder == "conditional_layer_norm":
                recon_speech = self._decode(codes, condition_embedding=merged_timbre_extractor_output.timbre_feats)[:, :, :speech.shape[-1]]
        else:
            recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]

        commit_loss = torch.stack(commit_losses).sum()
        enc_quant_loss = torch.stack(enc_quant_losses).sum()
        context_pred_acc = torch.stack(context_pred_acc).mean() if len(context_pred_acc) > 0 else self.zero

        # A: recon loss
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
            for real_inter, fake_inter in zip(real_intermediates, fake_intermediates):
                _loss = F.l1_loss(real_inter.detach(), fake_inter)
                disc_feature_losses.append(_loss)

        adversarial_loss = torch.stack(adversarial_losses).mean()
        feat_match_loss = torch.stack(disc_feature_losses).mean()

        # calculate losses
        gen_loss = recon_loss * self.recon_loss_weight + \
                   multi_spectral_recon_loss * self.multi_spectral_recon_loss_weight + \
                   adversarial_loss * self.adversarial_loss_weight + \
                   feat_match_loss * self.feat_match_loss_weight
        self.gen_loss += gen_loss.item()
        # loss = gen_loss + commit_loss + enc_quant_loss * self.enc_quant_loss_weight + context_loss * self.context_loss_weight
        loss = gen_loss + commit_loss + context_loss * self.context_loss_weight
        if self.contrastive_encoder is not None:
            loss += contrastive_loss
        if self.frame_contrastive_encoder is not None:
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
        if self.timbre_extractor is not None:
            half_speech_contrastive_output = timbre_extractor_output.get("half_speech_contrastive_output", None)
            if half_speech_contrastive_output is not None:
                loss += self.timbre_extractor.config.half_speech_contrastive_loss * half_speech_contrastive_output.loss
            speech_quant_out_contrastive_output = timbre_extractor_output.get("speech_quant_out_contrastive_output", None)
            if speech_quant_out_contrastive_output is not None:
                loss += self.timbre_extractor.config.speech_quant_out_contrastive_loss * speech_quant_out_contrastive_output.loss
        if self.slice_encoder is not None:
            loss += slice_encoder_output["loss"]

        stats = dict(
            generator_loss=loss.item(),
            generator_recon_loss=recon_loss.item(),
            generator_multi_spectral_recon_loss=multi_spectral_recon_loss.item(),
            generator_adv_loss=adversarial_loss.item(),
            generator_feat_match_loss=feat_match_loss.item(),
            generator_commit_loss=commit_loss.item(),
            generator_enc_quant_loss=enc_quant_loss.item(),
            context_loss=context_loss.item(),
            context_pred_acc=context_pred_acc.item(),
            batch_size=batch_size,
            batch_length=speech.shape[2],
        )
        if self.contrastive_encoder is not None:
            stats["generator_contrastive_loss"] = contrastive_loss.item()
            if contrastive_encoder_output.loss_dict:
                for loss_type, loss_value in contrastive_encoder_output.loss_dict.items():
                    stats[f"generator_contrastive_encoder_{loss_type}"] = loss_value
        if self.frame_contrastive_encoder is not None:
            stats["generator_frame_contrastive_loss"] = frame_contrastive_encoder_output.loss.item()
            if frame_contrastive_encoder_output.loss_dict:
                for loss_type, loss_value in frame_contrastive_encoder_output.loss_dict.items():
                    stats[f"generator_frame_contrastive_encoder_{loss_type}"] = loss_value
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
        if self.timbre_extractor is not None:
            half_speech_contrastive_output = timbre_extractor_output.get("half_speech_contrastive_output", None)
            if half_speech_contrastive_output is not None:
                stats["half_speech_contrastive_loss"] = half_speech_contrastive_output.loss.item()
            speech_quant_out_contrastive_output = timbre_extractor_output.get("speech_quant_out_contrastive_output", None)
            if speech_quant_out_contrastive_output is not None:
                stats["speech_quant_out_contrastive_loss"] = speech_quant_out_contrastive_output.loss.item()
        if self.slice_encoder is not None:
            stats["generator_slice_encoder_loss"] = slice_encoder_output["loss"].item()
            if slice_encoder_output["loss_dict"]:
                for loss_type, loss_value in slice_encoder_output["loss_dict"].items():
                    stats[f"generator_slice_encoder_{loss_type}"] = loss_value

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
            "sub_quants": sub_quants_list,
            "encoder_output": frames[0][0],
            "decoder_input": codes[0][0],
            "quants_in": quant_in_list, 
            "dist_list": dist_list,
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
            "slice_encoder_output": slice_encoder_output,
        }
        output = {param: value for param, value in output.items() if value is not None}
        return output

    def _forward_discriminator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        mel2ph: Optional[Sequence] = None,
        frame2ph_token_ids: Optional[Sequence] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
        disturb_speech: Optional[bool] = None,
        exchange_timbre: Optional[bool] = None,
        without_timbre: Optional[bool] = None,
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
        feature_lengths = self.get_frame_feature_lengths(speech_lengths)
        feature_attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None
        feature_padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None

        codes = []
        encoded_frames = self._encode(speech)
        frames = [[frame.emb, frame.scale] for frame in encoded_frames]
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
            # if self.ada_in is not None:
            #     emb = (emb - timbre_encoder_output.style_output.beta) / (timbre_encoder_output.style_output.gamma + self.eps)
            #     emb = self.ada_in.norm(emb)
                
            quant_in = emb.clone()

            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_before_quantizer is not None:
                quant_in = self.project["quantizer_in"](quant_in)
            if self.instance_norm_before_quantization or \
                (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization):
                quant_in = self.instance_norm_1d(quant_in)
            if self.l2_norm_before_quantization:
                quant_in = F.normalize(quant_in, dim=-1, eps=self.eps)

            if self.semantic_encoder is not None:
                semantic_output = self.semantic_encoder(speech=speech, speech_lengths=speech_lengths, codec_seq_len=emb.shape[1])
                first_layer_features = semantic_output.last_hidden_state
            else:
                first_layer_features = None

            # quant_out, indices, commit_loss, sub_quants = self.quantizer(quant_in)
            quantizer_output = self.quantizer(quant_in, first_layer_features=first_layer_features)
            quant_out, indices, commit_loss, sub_quants = quantizer_output["x"], quantizer_output["indices"], quantizer_output["commit_loss"], quantizer_output["sub_quants"]
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
            one_disc_loss = torch.mean(F.relu(1-real_logits)) + torch.mean(F.relu(1+fake_logits))
            disc_losses.append(one_disc_loss)
        disc_loss = torch.stack(disc_losses).mean()
        # To avoid discriminator overpowers the generator, without this recon losses may not converge
        if self.training:
            disc_loss = disc_loss * (disc_loss > self.gen_loss).float()
        if disc_loss.item() > self.gen_loss and self.training:
            logging.info(f"Will update discriminator: forward_step={self.forward_step}, "
                         f"disc_loss={disc_loss.item():.4f}, gen_loss={self.gen_loss:.4f}")
        self.gen_loss = 0

        # D: whether to use gradient penalty loss
        loss = disc_loss

        stats = dict(
            discriminator_total_loss=loss.item(),
            discriminator_loss=disc_loss.item(),
        )
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
        feature_lengths = self.get_frame_feature_lengths(speech_lengths)
        codes = []
        code_idxs = []
        all_sub_quants = []
        if speech.dim() == 2:
            speech = speech.unsqueeze(1)

        encoded_frames = self._encode(speech)
        frames = [(frame.emb, frame.scale) for frame in encoded_frames]
        dist_list = []
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
            # if self.ada_in is not None:
            #     emb = (emb - timbre_encoder_output.style_output.beta) / (timbre_encoder_output.style_output.gamma + self.eps)
            #     emb = self.ada_in.norm(emb)

            bb, tt, device = emb.shape[0], emb.shape[1], emb.device
            if self.bypass_quantizer:
                code_embs, indices, sub_quants = emb, torch.zeros(bb, tt, dtype=torch.long, device=device), torch.zeros_like(emb, device=device)
            else:
                quant_in = emb.clone()
                if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_before_quantizer is not None:
                    quant_in = self.project["quantizer_in"](quant_in)
                if self.instance_norm_before_quantization or \
                    (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization):
                    quant_in = self.instance_norm_1d(quant_in)
                if self.l2_norm_before_quantization:
                    quant_in = F.normalize(quant_in, dim=-1, eps=self.eps)

                if self.semantic_encoder is not None:
                    semantic_output = self.semantic_encoder(speech=speech, speech_lengths=speech_lengths, codec_seq_len=emb.shape[1])
                    first_layer_features = semantic_output.last_hidden_state
                else:
                    first_layer_features = None
                    
                # quant_out, indices, sub_quants = self.quantizer.inference(quant_in, bandwidth=bit_width)
                quantizer_output = self.quantizer.inference(quant_in, bandwidth=bit_width, first_layer_features=first_layer_features)
                quant_out, indices, sub_quants = quantizer_output["x"], quantizer_output["indices"], quantizer_output["sub_quants"]
                dist_list.append(quantizer_output["dist"])
                if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_after_quantizer is not None:
                    quant_out = self.project["quantizer_out"](quant_out)
                code_embs = quant_out
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
                    recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
                elif self.timbre_extractor.config.merge_with_decoder == "conditional_layer_norm":
                    recon_speech = self._decode(codes, condition_embedding=merged_timbre_extractor_output.timbre_feats)[:, :, :speech.shape[-1]]
            else:
                recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
        retval = dict(
            recon_speech=recon_speech,
            code_indices=code_idxs,
            code_embeddings=codes,
            sub_quants=all_sub_quants,
            dist_list=dist_list,
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
        feature_lengths = self.get_frame_feature_lengths(speech_lengths)
        feature_attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None
        feature_padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=feature_lengths.max()) if feature_lengths is not None else None

        codes = []
        code_idxs = []
        all_sub_quants = []
        if speech.dim() == 2:
            speech = speech.unsqueeze(1)
        # frames = self._encode(speech)
        encoded_frames = self._encode(speech)
        frames = [[frame.emb, frame.scale] for frame in encoded_frames]
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
        dist_list = []
        for emb, scale in frames:
            # if self.ada_in is not None:
            #     emb = (emb - timbre_encoder_output.style_output.beta) / (timbre_encoder_output.style_output.gamma + self.eps)
            #     emb = self.ada_in.norm(emb)

            quant_in = emb.clone()
            if self.timbre_strategy is not None and self.timbre_strategy.bottleneck_config is not None and self.timbre_strategy.bottleneck_config.linear_before_quantizer is not None:
                quant_in = self.project["quantizer_in"](quant_in)
            if self.instance_norm_before_quantization or \
                (self.timbre_strategy is not None and self.timbre_strategy.vqvc_config is not None and self.timbre_strategy.vqvc_config.instance_norm_before_quantization):
                quant_in = self.instance_norm_1d(quant_in)
            if self.l2_norm_before_quantization:
                quant_in = F.normalize(quant_in, dim=-1, eps=self.eps)

            if self.semantic_encoder is not None:
                semantic_output = self.semantic_encoder(speech=speech, speech_lengths=speech_lengths, codec_seq_len=emb.shape[1])
                first_layer_features = semantic_output.last_hidden_state
            else:
                first_layer_features = None
                
            # quant_out, indices, sub_quants = self.quantizer.inference(quant_in, bandwidth=bit_width)
            quantizer_output = self.quantizer.inference(quant_in, bandwidth=bit_width, first_layer_features=first_layer_features)
            quant_out, indices, sub_quants = quantizer_output["x"], quantizer_output["indices"], quantizer_output["sub_quants"]
            dist_list.append(quantizer_output["dist"])

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
                    recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
                elif self.timbre_extractor.config.merge_with_decoder == "conditional_layer_norm":
                    recon_speech = self._decode(codes, condition_embedding=merged_timbre_extractor_output.timbre_feats)[:, :, :speech.shape[-1]]
            else:
                recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
        retval = dict(
            recon_speech=recon_speech,
            code_indices=code_idxs,
            code_embeddings=codes,
            sub_quants=all_sub_quants,
            dist_list=dist_list,
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
        if speech.dim() == 2:
            speech = speech.unsqueeze(1)
        frames = self._encode(speech)
        dist_list = []
        for emb, scale in frames:
            quant_in = emb
            quantizer_output = self.quantizer.inference(quant_in, bandwidth=bit_width)
            quant_out, indices, sub_quants = quantizer_output["x"], quantizer_output["indices"], quantizer_output["sub_quants"]
            code_embs = quant_out
            codes.append([code_embs, scale if use_scale else None])
            code_idxs.append(indices)
            all_sub_quants.append(sub_quants)
            dist_list.append(quantizer_output["dist"])
        recon_speech = None
        if need_recon:
            recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
        retval = dict(
            recon_speech=recon_speech,
            code_indices=code_idxs,
            code_embeddings=codes,
            sub_quants=all_sub_quants,
            dist_list=dist_list,
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

    def inference_decoding_emb(
            self,
            token_idx: torch.Tensor,
            need_recon: bool = True,
            bit_width: int = None,
            use_scale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            token_idx (torch.Tensor): input code embeddings, B x T x Dim
            need_recon (bool): whether to return recon speech
            bit_width (int): The excepted bandwidth
            use_scale (bool): whether to use scale

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (T_wav,).
                * code_indices (Tensor): quantized code indices (L)
                * code_embeddings (Tensor): quantized code embeddings (L, d).

        """
        codes = [(token_idx, None)]
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
