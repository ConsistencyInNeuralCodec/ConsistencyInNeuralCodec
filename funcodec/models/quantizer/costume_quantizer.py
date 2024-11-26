import torch
import torch.nn.functional as F
from funcodec.modules.quantization.vq import ResidualVectorQuantizer, ResidualVectorQuantizerRaw
import typing as tp
from collections import OrderedDict

from funcodec.modules.modeling_feature_extractor import BaseFeatureExtractor, FeatureExtractorConfig
from funcodec.models.encoding_path.encoding_path import EncodingPathConfig, BaseEncodingPathModel
from funcodec.models.semantic_distill.semantic_distill import (
    SemanticDistillConfig,
    SemanticDistillOutput,
    SemanticDistillBaseModel,
)


class CostumeQuantizer(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        codebook_size: int = 1024,
        num_quantizers: int = 8,
        ema_decay: float = 0.95,
        kmeans_init: bool = False,
        sampling_rate: int = 24_000,
        quantize_dropout: bool = False,
        rand_num_quant: tp.Optional[tp.List] = None,
        encoder_hop_length: int = 320,
        use_ddp: bool = True,
        q0_ds_ratio: int = 1,
        codec_dim: int = None,
        codec_range: float = None,
        commitment_weight: float = 0.1,
        encoding_path_config: tp.Optional[EncodingPathConfig] = None,
        semantic_distill_config: tp.Optional[SemanticDistillConfig] = None,
        new_embedding_for_codebooks: tp.Optional[tp.Sequence] = None,
        l2_norm_on_embedding: bool = None,
        l2_norm_on_codebook: bool = None,
    ):
        super().__init__()
        if codec_dim is None:
            codec_dim = input_size

        self.input_proj, self.output_proj = None, None
        if codec_dim != input_size:
        # if codec_dim is not None:
            self.input_proj = torch.nn.Linear(input_size, codec_dim)
            self.output_proj = torch.nn.Linear(codec_dim, input_size)

        self.input_act, self.codec_range = None, None
        if codec_range is not None:
            self.input_act = torch.nn.Tanh()
            self.codec_range = codec_range

        if isinstance(encoding_path_config, dict):
            encoding_path_config = EncodingPathConfig(**encoding_path_config)
        if isinstance(semantic_distill_config, dict):
            semantic_distill_config = SemanticDistillConfig(**semantic_distill_config)

        self.rq = ResidualVectorQuantizer(
            dimension=codec_dim,
            n_q=num_quantizers,
            bins=codebook_size,
            decay=ema_decay,
            kmeans_init=kmeans_init,
            quantize_dropout=quantize_dropout,
            rand_num_quant=rand_num_quant,
            encoder_hop_length=encoder_hop_length,
            use_ddp=use_ddp,
            q0_ds_ratio=q0_ds_ratio,
            commitment_weight=commitment_weight,
            encoding_path_config=encoding_path_config,
            # semantic_distill_config=semantic_distill_config,
            new_embedding_for_codebooks=new_embedding_for_codebooks,
            l2_norm_on_embedding=l2_norm_on_embedding,
            l2_norm_on_codebook=l2_norm_on_codebook,
        )
        self.code_dim = input_size
        self.sampling_rate = sampling_rate
        self.bandwidth: tp.Optional[float] = None
        self.encoder_hop_length = encoder_hop_length
        self.codebook_size = codebook_size

        self.encoding_path_config = encoding_path_config
        self.semantic_distill_config = semantic_distill_config

        self.l2_norm_on_embedding = l2_norm_on_embedding
        self.l2_norm_on_codebook = l2_norm_on_codebook

    def forward(
            self,
            x,
            bandwidth: int = None,
            first_layer_features: tp.Optional[torch.Tensor] = None,
            **kwargs
    ):
        # x: input tensor in the shape of (B, T, C)
        # rq requires inputs in (B, C, T)

        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.input_act is not None:
            x = self.input_act(x) * self.codec_range
        qv = self.rq(
            x.permute(0, 2, 1), self.sampling_rate, bandwidth,
            first_layer_features=first_layer_features.permute(0, 2, 1) if first_layer_features is not None else None,
        )
        x, indices, commit_loss, sub_quants = qv.quantized, qv.codes, qv.penalty, qv.sub_quants
        dist = qv.dist

        x = x.permute(0, 2, 1)
        if self.output_proj is not None:
            x = self.output_proj(x)

        # return x, indices, commit_loss, sub_quants
        return OrderedDict(
            x=x, indices=indices,
            commit_loss=commit_loss,
            sub_quants=sub_quants,
            dist=dist,
        )

    def inference(
            self,
            x,
            bandwidth: int = None,
            first_layer_features: tp.Optional[torch.Tensor] = None,
            **kwargs
    ):
        # x: input tensor in the shape of (B, T, C)
        # rq requires inputs in (B, C, T)
        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.input_act is not None:
            x = self.input_act(x) * self.codec_range

        qv = self.rq(
            x.permute(0, 2, 1), self.sampling_rate, bandwidth,
            first_layer_features=first_layer_features.permute(0, 2, 1) if first_layer_features is not None else None,
        )
        x, indices, sub_quants = qv.quantized, qv.codes, qv.sub_quants
        dist = qv.dist

        x = x.permute(0, 2, 1)
        if self.output_proj is not None:
            x = self.output_proj(x)

        # return x, indices, sub_quants
        return OrderedDict(
            x=x, indices=indices,
            sub_quants=sub_quants,
            dist=dist,
        )

    def encode(
        self,
        x,
        bandwidth: int = None,
        first_layer_features: tp.Optional[torch.Tensor] = None,
        **kwargs
    ):
        # x: input tensor in the shape of (B, T, C)
        # rq requires inputs in (B, C, T)
        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.input_act is not None:
            x = self.input_act(x) * self.codec_range

        indices, dist = self.rq.encode(
            x.permute(0, 2, 1), self.sampling_rate, bandwidth,
            first_layer_features=first_layer_features.permute(0, 2, 1) if first_layer_features is not None else None,
        )
        # return value in n_q x B x T
        return indices, dist

    def decode(self, indices):
        quantized_out = self.rq.decode(indices)
        # quantized_out in B x D x T
        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out.transpose(1, 2)).transpose(1, 2)
        return quantized_out

    def output_size(self):
        return self.code_dim

class CostumeQuantizerRaw(torch.nn.Module):
    def __init__(
            self,
            input_size: int = 512,
            codebook_size: int = 1024,
            num_quantizers: int = 8,
            ema_decay: float = 0.95,
            kmeans_init: bool = False,
            sampling_rate: int = 24_000,
            quantize_dropout: bool = False,
            rand_num_quant: tp.Optional[tp.List] = None,
            encoder_hop_length: int = 320,
            use_ddp: bool = True,
            q0_ds_ratio: int = 1,
            codec_dim: int = None,
            codec_range: float = None,
            commitment_weight: float = 0.1,
            requires_projection: bool = None,
            l2_norm_on_embedding: bool = None,
            l2_norm_on_codebook: bool = None,
            feature_extractor_config: tp.Optional[FeatureExtractorConfig] = None,
            feature_extractor_only_for_quant0: tp.Optional[bool] = None,
    ):
        super().__init__()
        if codec_dim is None:
            codec_dim = input_size

        self.input_proj, self.output_proj = None, None
        # if codec_dim != input_size:
        if codec_dim is not None:
            self.input_proj = torch.nn.Linear(input_size, codec_dim)
            self.output_proj = torch.nn.Linear(codec_dim, input_size)

        self.input_act, self.codec_range = None, None
        if codec_range is not None:
            self.input_act = torch.nn.Tanh()
            self.codec_range = codec_range

        self.rq = ResidualVectorQuantizerRaw(
            dimension=codec_dim,
            n_q=num_quantizers,
            bins=codebook_size,
            decay=ema_decay,
            kmeans_init=kmeans_init,
            quantize_dropout=quantize_dropout,
            rand_num_quant=rand_num_quant,
            encoder_hop_length=encoder_hop_length,
            use_ddp=use_ddp,
            q0_ds_ratio=q0_ds_ratio,
            commitment_weight=commitment_weight,
            requires_projection=requires_projection,
            l2_norm_on_embedding=l2_norm_on_embedding,
            l2_norm_on_codebook=l2_norm_on_codebook,
            feature_extractor_config=feature_extractor_config,
            feature_extractor_only_for_quant0=feature_extractor_only_for_quant0,
        )
        self.code_dim = input_size
        self.sampling_rate = sampling_rate
        self.bandwidth: tp.Optional[float] = None
        self.encoder_hop_length = encoder_hop_length
        self.codebook_size = codebook_size

        self.l2_norm_on_embedding = l2_norm_on_embedding
        self.l2_norm_on_codebook = l2_norm_on_codebook

    def forward(
            self,
            x,
            bandwidth: int = None,
    ):
        # x: input tensor in the shape of (B, T, C)
        # rq requires inputs in (B, C, T)

        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.input_act is not None:
            x = self.input_act(x) * self.codec_range
        # linear之后, 欧氏距离之前, 归一化
        # if self.l2_norm_on_embedding:
        #     x = F.normalize(x, dim=1)
        # if self.l2_norm_on_codebook:
        #     pass

        qv = self.rq(x.permute(0, 2, 1), self.sampling_rate, bandwidth)
        x, indices, commit_loss, sub_quants = qv.quantized, qv.codes, qv.penalty, qv.sub_quants

        x = x.permute(0, 2, 1)
        if self.output_proj is not None:
            x = self.output_proj(x)

        return x, indices, commit_loss, sub_quants

    def inference(
            self,
            x,
            bandwidth: int = None,
    ):
        # x: input tensor in the shape of (B, T, C)
        # rq requires inputs in (B, C, T)
        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.input_act is not None:
            x = self.input_act(x) * self.codec_range

        qv = self.rq(x.permute(0, 2, 1), self.sampling_rate, bandwidth)
        x, indices, sub_quants = qv.quantized, qv.codes, qv.sub_quants

        x = x.permute(0, 2, 1)
        if self.output_proj is not None:
            x = self.output_proj(x)

        return x, indices, sub_quants

    def encode(
            self,
            x,
            bandwidth: int = None,
    ):
        # x: input tensor in the shape of (B, T, C)
        # rq requires inputs in (B, C, T)
        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.input_act is not None:
            x = self.input_act(x) * self.codec_range

        indices = self.rq.encode(x.permute(0, 2, 1), self.sampling_rate, bandwidth)
        # return value in n_q x B x T
        return indices

    def decode(self, indices):
        quantized_out = self.rq.decode(indices)
        # quantized_out in B x D x T
        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out.transpose(1, 2)).transpose(1, 2)
        return quantized_out

    def output_size(self):
        return self.code_dim