import sys, importlib
import os, time, random
import logging, warnings
import omegaconf
from collections import OrderedDict
from dataclasses import dataclass
import torch
from torch import nn
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

from speechbrain.inference.classifiers import EncoderClassifier as SpeechbrainEncoderClassifier
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .configuration_timbre_encoder import TimbreEncoderConfig
from .fast_speech_transformer import FastSpeechDecoder
from .fa_transformer import TransformerEncoder as SALNTransformerEncoder
from .fa_transformer import TransformerEncoderConfig as SALNTransformerEncoderConfig
from ..utils import lengths_to_padding_mask, lengths_to_attention_mask

logger = logging.getLogger(__name__)


def mean_pooling(x: torch.Tensor, attention_mask: torch.Tensor):
    """
    x: [bsz, length, dim]
    attention_mask: [bsz, length]
        0: mask
        1: not mask
    """
    attention_mask = attention_mask.unsqueeze(-1) # [bsz, length, 1]
    x = x * attention_mask
    mean_x = x.sum(1) / attention_mask.sum(1).clamp(min=1)
    return mean_x


@dataclass
class StyleAdaptiveOutput(BaseModelOutput):
    gamma: torch.FloatTensor = None
    beta: torch.FloatTensor = None
    out: torch.FloatTensor = None


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.in_dim = normalized_shape
        self.norm = nn.LayerNorm(self.in_dim, eps=eps, elementwise_affine=False)
        self.style = nn.Linear(self.in_dim, self.in_dim * 2)
        self.style.bias.data[: self.in_dim] = 1
        self.style.bias.data[self.in_dim :] = 0

    def forward(self, x, condition, gamma=None, beta=None):
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)
        # x: (B, T, d); condition: (B, 1, d)
        if gamma is None and beta is None:
            style = self.style(condition)
            gamma, beta = style.chunk(2, -1)
        # out = self.norm(x)
        out = x
        out = gamma * out + beta
        return StyleAdaptiveOutput(
            gamma=gamma,
            beta=beta,
            out=out,
        )

    def forward_statistic(self, condition):
        style = self.style(condition)
        gamma, beta = style.chunk(2, -1)
        return StyleAdaptiveOutput(
            gamma=gamma,
            beta=beta,
        )


@dataclass
class TimbreEncoderOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    transformed_speech: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.BoolTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
    output_feature_lengths: Optional[torch.LongTensor] = None
    style_output: Optional[torch.Tensor] = None


class TimbreEncoderPreTrainedModel(PreTrainedModel):
    config_class = TimbreEncoderConfig
    supports_gradient_checkpointing = True

    @staticmethod
    def build_timbre_encoder(
        config: Optional[Union[TimbreEncoderConfig, Dict]] = None,
        pretrained_model_name_or_path: Optional[str] = None,
        encoder_type: Optional[str] = None,
        **kwargs
    ) -> "TimbreEncoderPreTrainedModel":
        if config is not None:
            if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
                config = TimbreEncoderConfig(**config)
            if config.encoder_type == "speechbrain.inference.classifiers.EncoderClassifier":
                return SpeakerEncoderFromSpeechBrain(config=config)
            elif config.encoder_type == "speechbrain.Xvector":
                return SpeakerEncoderFromXvector.from_pretrained(config=config)
            elif config.encoder_type == "fast_speech_transformer.FastSpeechDecoder":
                return SpeakerEncoderFromFastSpeech(config=config)
            elif config.encoder_type == "wav2vec2":
                return SpeakerEncoderFromWav2vec2(config=config)
            elif config.encoder_type == "fa_timbre_encoder":
                return SpeakerEncoderFromFACodec(config=config)
            else:
                raise NotImplementedError
        if pretrained_model_name_or_path is not None and encoder_type is not None:
            if encoder_type == "speechbrain.inference.classifiers.EncoderClassifier":
                return SpeakerEncoderFromSpeechBrain.from_pretrained(pretrained_model_name_or_path, **kwargs)
        raise NotImplementedError


class SpeakerEncoderFromXvector(TimbreEncoderPreTrainedModel):
    def __init__(
        self,
        config: Union[TimbreEncoderConfig, Dict],
        mods: Optional[nn.Module] = None,
        xvector_config: Optional[Dict] = None,
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = TimbreEncoderConfig(**config)
        super().__init__(config=config)
        self.mods = mods
        self.xvector_config = xvector_config
        if self.config.freeze:
            self.freeze()
        else:
            self.unfreeze()

    @classmethod
    def from_pretrained(
        cls,
        config: Union[TimbreEncoderConfig, Dict],
        model_dir: Optional[str] = None,
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = TimbreEncoderConfig(**config)
        if model_dir is None:
            model_dir = config.model_dir
        import speechbrain
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.Xvector import Xvector
        xvector_config_path = os.path.join(model_dir, "config.yaml")
        xvector_config = omegaconf.OmegaConf.load(xvector_config_path)
        xvector_config = omegaconf.OmegaConf.to_container(xvector_config, resolve=True)

        compute_features = speechbrain.lobes.features.Fbank(**xvector_config["compute_features"])
        mean_var_norm = speechbrain.processing.features.InputNormalization(**xvector_config["mean_var_norm"])
        embedding_model = speechbrain.lobes.models.Xvector.Xvector(**xvector_config["embedding_model"])
        mean_var_norm_emb = speechbrain.processing.features.InputNormalization(**xvector_config["mean_var_norm_emb"]) # False

        mods = torch.nn.ModuleDict(
            OrderedDict(
                compute_features=compute_features,
                mean_var_norm=mean_var_norm,
                embedding_model=embedding_model,
                mean_var_norm_emb=mean_var_norm_emb,
            )
        )
        for module in xvector_config["load_modules"]:
            # print(666, module)
            state_dict = torch.load(os.path.join(xvector_config["pretrained_model_dir"], f"{module}.ckpt"))
            mods[module].load_state_dict(state_dict, strict=False)

        return SpeakerEncoderFromXvector(
            config=config,
            mods=mods,
            xvector_config=xvector_config,
        )

    def freeze(self):
        # self.mods.eval()
        self.requires_grad = False
        self.mods.requires_grad = False
        for name, param in self.mods.named_parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        # self.mods.train()
        self.requires_grad = True
        self.mods.requires_grad = True
        for name, param in self.mods.named_parameters():
            param.requires_grad = True

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)
        if wav_lens is None:
            wav_lens = torch.tensor([wavs.shape[1]] * wavs.shape[0], dtype=torch.long, device=wavs.device)
        wavs = wavs.float()
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.mods.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=wavs.device)
            )
        return embeddings

    def forward(
        self,
        wavs,
        wav_lens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: Optional[bool] = False
    ):
        if wavs.dim() == 3:
            if wavs.shape[1] == 1:
                # batch_size, channel, timestep
                wavs = wavs.squeeze(1)
                # batch_size, timestep
            else:
                raise NotImplementedError(f"wavs shape = {wavs.shape}")
        if wav_lens is None:
            if padding_mask is not None:
                wav_lens = padding_mask.shape[1] - padding_mask.sum(-1)
            elif attention_mask is not None:
                wav_lens = attention_mask.sum(-1)
        # if attention_mask is None:
        #     attention_mask = torch.ones(wavs.shape[0], wavs.shape[1], dtype=torch.bool, device=wavs.device)
        # print(f"666 wav_lens = {wav_lens}")
        if self.config.freeze:
            with torch.no_grad():
                embeddings = self.encode_batch(wavs, wav_lens, normalize)
        else:
            embeddings = self.encode_batch(wavs, wav_lens, normalize)
        return TimbreEncoderOutput(last_hidden_state=embeddings, attention_mask=attention_mask)


class SpeakerEncoderFromSpeechBrain(TimbreEncoderPreTrainedModel):
    def __init__(
        self,
        config: Union[TimbreEncoderConfig, Dict],
        model: Optional[SpeechbrainEncoderClassifier] = None,
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = TimbreEncoderConfig(**config)
        # print("SpeakerEncoderFromSpeechBrain", config)
        if model is None:
            model = SpeechbrainEncoderClassifier.from_hparams(
                source=config.model_dir,
                savedir=os.path.join(config.model_dir, "cache", time.strftime("%m-%d_%H:%M:%S", time.localtime()), f"{random.randint(0, 1_000_000_000_000)}"),
                # savedir=os.path.join(config.model_dir, "cache"),
                # savedir="/cpfs01/shared/Group-m6-intern/user.user/checkpoint/speechbrain"
                # overwrite=True,
            )
        super().__init__(config=config)
        self.model = model
        if self.config.freeze:
            self.freeze()
        else:
            self.unfreeze()
    
    def freeze(self):
        # self.model.eval()
        self.requires_grad = False
        self.model.requires_grad = False
        for name, param in self.model.named_parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        # self.model.train()
        self.requires_grad = True
        self.model.requires_grad = True
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path: str, **kwargs):
        model = SpeechbrainEncoderClassifier.from_hparams(source=pretrained_model_name_or_path)
        config_kwargs = {
            "encoder_type": "speechbrain.inference.classifiers.EncoderClassifier",
            "model_dir": pretrained_model_name_or_path,
            "freeze": kwargs.pop("freeze", False),
            "sample_rate": kwargs.pop("sample_rate", 16_000),
            "load_from_existed_speaker_emebd": False,
            "batch_size_is_one": False,
        }
        config = TimbreEncoderConfig(**config_kwargs)
        return SpeakerEncoderFromSpeechBrain(config=config, model=model)

    def _forward(self, wavs, wav_lens=None, normalize=False):
        """Encodes the input audio into a single vector embedding.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = <this>.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0])

        # Storing waveform in the specified device
        # wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.model.mods.compute_features(wavs)
        feats = self.model.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.model.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.model.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0])
            )
        attention_mask = torch.ones(wavs.shape[0], 1, dtype=torch.bool, device=wavs.device)
        return TimbreEncoderOutput(last_hidden_state=embeddings, attention_mask=attention_mask)

    def forward(
        self,
        wavs,
        wav_lens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: Optional[bool] = False
    ):
        if wavs.dim() == 3:
            if wavs.shape[1] == 1:
                # batch_size, channel, timestep
                wavs = wavs.squeeze(1)
            else:
                raise NotImplementedError(f"wavs shape = {wavs.shape}")
        if wav_lens is None:
            if padding_mask is not None:
                wav_lens = padding_mask.shape[1] - padding_mask.sum(-1)
            elif attention_mask is not None:
                wav_lens = attention_mask.sum(-1)
        if self.config.freeze:
            with torch.no_grad():
                return self._forward(wavs, wav_lens, normalize)
        else:
            return self._forward(wavs, wav_lens, normalize)


class SpeakerEncoderFromFastSpeech(TimbreEncoderPreTrainedModel):
    def __init__(
        self,
        config: Union[TimbreEncoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = TimbreEncoderConfig(**config)
        super().__init__(config=config)
        fast_speech_decoder_kwargs = {
            "dropout": getattr(config, "dropout", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_layers": getattr(config, "num_layers", None),
            "kernel_size": getattr(config, "kernel_size", None),
            "num_heads": getattr(config, "num_heads", None),
        }
        fast_speech_decoder_kwargs = {param: value for param, value in fast_speech_decoder_kwargs.items() if value}
        self.model = FastSpeechDecoder(**fast_speech_decoder_kwargs)
        if config.in_dim is not None:
            self.input2hidden = nn.Linear(config.in_dim, config.hidden_size)
        else:
            self.input2hidden = nn.Identity()

    def forward(
        self, spectrogram, 
        padding_mask=None, attention_mask=None, 
        attn_mask=None, return_hiddens=False,
    ):
        x = spectrogram
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        x = self.input2hidden(x)
        if padding_mask is None and attention_mask is not None:
            padding_mask = attention_mask != 1
        output = self.model.forward(
            x,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
            return_hiddens=return_hiddens,
        )
        attention_mask = ~padding_mask
        return TimbreEncoderOutput(
            last_hidden_state=output,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            transformed_speech=spectrogram,
        )


class SpeakerEncoderFromWav2vec2(TimbreEncoderPreTrainedModel):
    def __init__(
        self,
        config: Union[TimbreEncoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = TimbreEncoderConfig(**config)
        super().__init__(config=config)
        config = Wav2Vec2Config.from_pretrained(config.config_path)
        self.model = Wav2Vec2Model(config)

    def forward(
        self,
        wavs: torch.FloatTensor,
        wav_lens: torch.LongTensor,
        **kwargs
    ):
        if wavs.dim() == 3:
            if wavs.shape[1] == 1:
                # batch_size, channel, timestep
                wavs = wavs.squeeze(1)
            else:
                raise NotImplementedError(f"wavs shape = {wavs.shape}")
        attention_mask = lengths_to_attention_mask(wav_lens, max_lens=wavs.shape[1])
        output = self.model(input_values=wavs, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        feature_lengths = self.model._get_feat_extract_output_lengths(wav_lens)
        attention_mask = lengths_to_attention_mask(feature_lengths, max_lens=feature_lengths.max().item())
        padding_mask = lengths_to_padding_mask(feature_lengths, max_lens=feature_lengths.max().item())
        return TimbreEncoderOutput(
            last_hidden_state=output["last_hidden_state"],
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            output_feature_lengths=feature_lengths,
        )
    

class SpeakerEncoderFromFACodec(TimbreEncoderPreTrainedModel):
    def __init__(
        self,
        config: Union[TimbreEncoderConfig, Dict],
        **kwargs
    ):
        if isinstance(config, (dict, omegaconf.dictconfig.DictConfig)):
            config = TimbreEncoderConfig(**config)
        super().__init__(config=config)
        self.model_config = SALNTransformerEncoderConfig(
            enc_emb_tokens=None,
            encoder_layer=config.num_layers,
            encoder_hidden=config.hidden_size,
            encoder_head=config.num_heads,
            conv_filter_size=config.hidden_size * 4,
            conv_kernel_size=config.kernel_size,
            encoder_dropout=config.dropout,
            use_cln=False,
            cfg=None,
        )
        if config.in_dim is not None:
            self.input2hidden = nn.Linear(config.in_dim, config.hidden_size)
        else:
            self.input2hidden = nn.Identity()
        self.model = SALNTransformerEncoder(**self.model_config.to_dict())

    def forward(
        self, spectrogram, 
        padding_mask=None, attention_mask=None, 
        **kwargs,
    ):
        x = spectrogram
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        x = self.input2hidden(x)
        if padding_mask is None and attention_mask is not None:
            padding_mask = ~attention_mask
        elif padding_mask is not None and attention_mask is None:
            attention_mask = ~padding_mask
        output = self.model(
            x,
            key_padding_mask=attention_mask, # key_padding_mask: (B, T), mask is 0; condition: (B, T, d)
        )
        return TimbreEncoderOutput(
            last_hidden_state=output,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            transformed_speech=spectrogram,
        )