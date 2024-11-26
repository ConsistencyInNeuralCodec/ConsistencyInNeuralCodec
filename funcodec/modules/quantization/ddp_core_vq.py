# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""
import random
import typing as tp
from random import randrange
from einops import rearrange, repeat
from math import ceil
import torch
from torch import nn
import torch.nn.functional as F

from funcodec.modules.quantization import distrib
from funcodec.modules.modeling_feature_extractor import BaseFeatureExtractor, FeatureExtractorConfig
from funcodec.models.encoding_path.encoding_path import EncodingPathConfig, BaseEncodingPathModel

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


def preprocess(x):
    x = rearrange(x, "... d -> (...) d")
    return x


def postprocess_emb(embed_ind, shape):
    return embed_ind.view(*shape[:-1])


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        l2_norm_on_embedding: bool = None,
        l2_norm_on_codebook: bool = None,
        codebook_idx: tp.Optional[int] = None,
        encoding_path_config: tp.Optional[EncodingPathConfig] = None,
    ):
        super().__init__()
        self.decay = decay
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.inited = None
        self.cluster_size = None
        self.embed = None
        self.embed_avg = None
        self.training = True

        self.l2_norm_on_embedding = l2_norm_on_embedding
        self.l2_norm_on_codebook = l2_norm_on_codebook
        self.codebook_idx = codebook_idx
        self.encoding_path_config = encoding_path_config
        if self.encoding_path_config is not None:
            self.encoding_path_model = BaseEncodingPathModel.build_model(config=self.encoding_path_config, codebook_idx=codebook_idx)

    def init_embed_(self, data):
        if self.inited:
            return
        # If data is (2048, 128), kmeans return the self.codebook_size cluster center embeddings and count the bins for each cluster center
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        distrib.broadcast_tensors([self.embed, self.embed_avg, self.cluster_size, self.inited])
        # print(f"666 update codebook with kmeans")

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        # 某个聚类中心的样本点太少了，因此Expire了，需要做更新, 从data里面随机拿表征来做替换
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        # sync buffers outside for efficiency
        # distrib.broadcast_tensors(self.buffers())

    def quantize(self, x):
        if self.l2_norm_on_embedding:
            # print(f"666 x = {x.shape} EuclideanCodebook.l2_norm_on_embedding = {self.l2_norm_on_embedding}")
            x = F.normalize(x, dim=1)

        embed = self.embed.t()

        if self.l2_norm_on_codebook:
            # print(f"666 embed = {embed.shape} EuclideanCodebook.l2_norm_on_codebook = {self.l2_norm_on_codebook}")
            embed = F.normalize(embed, dim=0)

        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        # print(f"666 EuclideanCodebook x = {x.shape}, codebook_idx = {self.codebook_idx}, dist = {dist.shape}, codebook = {self.embed.shape}") 
        # dist: [batch_size * seq_len, codebook_size]

        return embed_ind, dist

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x, buffers):
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers

        shape = x.shape
        batch_size, seq_len, dim = shape
        # pre-process
        hidden_state = x.clone()
        x = preprocess(x)
        # quantize
        embed_ind, dist = self.quantize(x)
        # post-process
        embed_ind = postprocess_emb(embed_ind, shape)

        dist = dist.view(batch_size, seq_len, -1)
        if self.encoding_path_config is not None:
            # if not hasattr(self, "l2_distance"):
            #     codebook = self.embed.clone()
            #     cb_expanded = codebook.unsqueeze(0)  # Shape becomes [1, 1024, 128]
            #     cb_tiled = cb_expanded.repeat(self.codebook_size, 1, 1)  # Repeat the codebook 1024 times
            #     diff = cb_tiled - cb_expanded.transpose(0, 1)  # Shape becomes [1024, 1024, 128]
            #     l2_distance = torch.sqrt(torch.pow(diff, 2).sum(2))  # Shape becomes [1024, 1024]
            #     self.register_buffer("l2_distance", l2_distance)
            #     print(f"l2_distance updated")
            if self.codebook_idx in self.encoding_path_config.apply_for_codebooks:
                embed_ind = self.encoding_path_model.encode(hidden_state=hidden_state, dist=dist, embed_ind=embed_ind)

        return embed_ind, dist

    def decode(self, embed_ind, buffers):
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers

        quantize = self.dequantize(embed_ind)
        return quantize

    def find_encoding_path(self):
        if self.encoding_path_config is None:
            return False
        if self.training:
            return "train" in self.encoding_path_config.run_time
        return "eval" in self.encoding_path_config.run_time

    def forward(self, x, buffers):
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers

        shape, dtype = x.shape, x.dtype
        batch_size, seq_len, dim = shape
        hidden_state = x.clone()
        x = preprocess(x)

        self.init_embed_(x)

        embed_ind, dist = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = postprocess_emb(embed_ind, shape)
        # quantize = self.dequantize(embed_ind)

        dist = dist.view(batch_size, seq_len, -1)
        if self.find_encoding_path():
            # if not hasattr(self, "l2_distance"):
            #     codebook = self.embed.clone()
            #     cb_expanded = codebook.unsqueeze(0)  # Shape becomes [1, 1024, 128]
            #     cb_tiled = cb_expanded.repeat(self.codebook_size, 1, 1)  # Repeat the codebook 1024 times
            #     diff = cb_tiled - cb_expanded.transpose(0, 1)  # Shape becomes [1024, 1024, 128]
            #     l2_distance = torch.sqrt(torch.pow(diff, 2).sum(2))  # Shape becomes [1024, 1024]
            #     self.register_buffer("l2_distance", l2_distance)
            #     print(f"l2_distance updated")
            if self.codebook_idx in self.encoding_path_config.apply_for_codebooks:
                # print(f"666 codebook_idx = {self.codebook_idx}")
                embed_ind = self.encoding_path_model.encode(hidden_state=hidden_state, dist=dist, embed_ind=embed_ind)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            # self.cluster_size 累加的时候用ema累加
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            # moving average 的值拿来做新的embed
            self.embed.data.copy_(embed_normalized)
            # Note: after ema update, there is a very small difference between codebooks on GPUs.
            # The impact can be very small, ignore it.

        return quantize, embed_ind, dist


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently, supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
            self,
            dim: int,
            codebook_size: int,
            codebook_dim: tp.Optional[int] = None,
            decay: float = 0.99,
            epsilon: float = 1e-5,
            kmeans_init: bool = True,
            kmeans_iters: int = 50,
            threshold_ema_dead_code: int = 2,
            commitment_weight: float = 1.,
            requires_projection: bool = None,
            l2_norm_on_embedding: bool = None,
            l2_norm_on_codebook: bool = None,
            feature_extractor_config: tp.Optional[FeatureExtractorConfig] = None,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        if requires_projection is None:
            requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim)) if requires_projection else (nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim)) if requires_projection else (nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim, codebook_size=codebook_size,
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
            decay=decay, epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
            l2_norm_on_embedding=l2_norm_on_embedding,
            l2_norm_on_codebook=l2_norm_on_codebook,
        )
        self.codebook_size = codebook_size
        self.training = True

        if feature_extractor_config is not None:
            feature_extractor_config = FeatureExtractorConfig(**feature_extractor_config)
            self.feature_extractor = BaseFeatureExtractor.build_model(config=feature_extractor_config)
            self.project_out = self.feature_extractor
            # self.project_in = self.feature_extractor
        else:
            self.feature_extractor = None

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x, buffers):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x, buffers)
        return embed_in

    def decode(self, embed_ind, buffers):
        quantize = self._codebook.decode(embed_ind, buffers)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x, buffers, n_q=None):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x, buffers)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class VectorQuantizationM(nn.Module):
    """Vector quantization implementation.
    Currently, supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 0.2,
        codebook_idx: tp.Optional[int] = None,
        encoding_path_config: tp.Optional[EncodingPathConfig] = None,
        new_embedding_for_codeword: tp.Optional[bool] = False,
        l2_norm_on_embedding: bool = None,
        l2_norm_on_codebook: bool = None,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)
        # if new_embedding_for_codeword:
            # _codebook_dim = 2
            # print(666, dim, _codebook_dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim)) if requires_projection else (nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim)) if requires_projection else (nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim, codebook_size=codebook_size,
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
            decay=decay, epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
            codebook_idx=codebook_idx,
            encoding_path_config=encoding_path_config,
            l2_norm_on_embedding=l2_norm_on_embedding,
            l2_norm_on_codebook=l2_norm_on_codebook,
        )
        self.codebook_size = codebook_size
        self.training = True

        self.codebook_idx = codebook_idx
        self.encoding_path_config = encoding_path_config
        if new_embedding_for_codeword:
            self.new_embedding_for_codeword = nn.Embedding(num_embeddings=codebook_size, embedding_dim=_codebook_dim, padding_idx=None)
        else:
            self.new_embedding_for_codeword = None


    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x, buffers):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in, dist = self._codebook.encode(x, buffers)
        return embed_in, dist

    def decode(self, embed_ind, buffers):
        quantize = self._codebook.decode(embed_ind, buffers)
        if self.new_embedding_for_codeword is not None:
            quantize = self.new_embedding_for_codeword(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x, buffers, n_q=None):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        # print(f"666 VectorQuantizationM codebook_idx = {self.codebook_idx}, x = {x.shape}")

        quantize, embed_ind, dist = self._codebook(x, buffers)

        if self.training:
            assert n_q is not None, 'We should use n_q to normalize the gradient for encoders'
            quantize = x/n_q + (quantize - x/n_q).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        if self.new_embedding_for_codeword is not None:
            quantize = self.new_embedding_for_codeword(embed_ind)
            # print(f"666 embed_ind = {embed_ind.shape}, quantize = {quantize.shape}")

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")

        return quantize, embed_ind, loss, dist


class DistributedResidualVectorQuantizationM(nn.Module):
    """Efficient distributed residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(
        self, *,
        num_quantizers,
        quantize_dropout: bool = False,
        rand_num_quant: tp.Optional[tp.List] = None,
        **kwargs
    ):
        super().__init__()
        """
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        """
        codebook_size, codebook_dim = kwargs["codebook_size"], kwargs["dim"]
        kmeans_init = kwargs["kmeans_init"]
        if not kwargs["kmeans_init"]:
            embed = uniform_init(num_quantizers, codebook_size, codebook_dim)
        else:
            embed = torch.zeros(num_quantizers, codebook_size, codebook_dim)

        self.register_buffer("inited", torch.Tensor([[not kmeans_init] for _ in range(num_quantizers)]))
        self.register_buffer("cluster_size", torch.zeros(num_quantizers, codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

        self.q0_ds_ratio = 1
        if "q0_ds_ratio" in kwargs:
            self.q0_ds_ratio = kwargs.pop("q0_ds_ratio")

        self.layers = nn.ModuleList()
        for i in range(num_quantizers):
            vq_args = dict(**kwargs)
            # DEBUG: Jin: Modify to VectorQuantizationM

            vq_args["codebook_idx"] = i
            encoding_path_config = kwargs.get("encoding_path_config", None)
            if encoding_path_config is not None and i not in encoding_path_config.apply_for_codebooks:
                vq_args.pop("encoding_path_config", None)
            if kwargs.get("new_embedding_for_codebooks", None) is not None and i in kwargs["new_embedding_for_codebooks"]:
                vq_args["new_embedding_for_codeword"] = True
            if kwargs.get("new_embedding_for_codebooks", None) is not None:
                vq_args.pop("new_embedding_for_codeword")
            else:
                vq_args.pop("new_embedding_for_codebooks", None)

            vq = VectorQuantizationM(**vq_args)
            self.layers.append(vq)

        self.quantize_dropout = quantize_dropout
        self.rand_num_quant = rand_num_quant

        self.encoding_path_config = kwargs.get("encoding_path_config", None)

    def forward(
        self,
        x,
        n_q: tp.Optional[int] = None,
        first_layer_features: tp.Optional[torch.Tensor] = None,
        **kwargs
    ):
        quantized_out = torch.zeros_like(x)
        residual = x
        bb, cc, tt = x.shape
        device = x.device

        all_losses = []
        all_indices = []
        all_sub_quants = []
        all_dists = []
        n_q = n_q or len(self.layers)

        should_quantize_dropout = self.training and self.quantize_dropout and self.rand_num_quant is not None
        if should_quantize_dropout:
            rand_quantize_dropout_index = random.choice(self.rand_num_quant)

            null_indices_shape = (x.shape[0], x.shape[2])
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)
            null_loss = torch.full((1,), 0., device=device, dtype=x.dtype)
            null_sub_quant = torch.full(x.shape, -1, device=device, dtype=x.dtype)

        # print(f"666 DistributedResidualVectorQuantizationM")
        for quantizer_index, layer in enumerate(self.layers[:n_q]):
            # dropout except the first quantizer
            if should_quantize_dropout and quantizer_index >= rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                all_sub_quants.append(null_sub_quant)
                continue

            if quantizer_index == 0 and first_layer_features is not None:
                residual = first_layer_features
                # print(666, first_layer_features.shape)
            elif quantizer_index == 1 and first_layer_features is not None:
                residual = x - quantized_out.detach()
            
            quant_in = residual
            if self.q0_ds_ratio > 1 and quantizer_index == 0:
                quant_in = F.interpolate(quant_in, size=[tt//2])
            quantized, indices, loss, dist = layer(quant_in, [
                self.inited[quantizer_index],
                self.cluster_size[quantizer_index],
                self.embed[quantizer_index],
                self.embed_avg[quantizer_index]
            ],
                n_q=rand_quantize_dropout_index if should_quantize_dropout else n_q
            )
            if self.q0_ds_ratio > 1 and quantizer_index == 0:
                quantized = F.interpolate(quantized, size=[tt])
                indices = F.interpolate(indices.unsqueeze(1).float(), size=[tt]).squeeze(1).long()
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
            all_sub_quants.append(quantized)
            all_dists.append(dist)

        # sync buffers after one forward step
        distrib.broadcast_tensors(self.buffers())
        out_losses, out_indices, out_sub_quants = map(torch.stack, (all_losses, all_indices, all_sub_quants))
        out_dists = torch.stack(all_dists)

        return quantized_out, out_indices, out_losses, out_sub_quants, out_dists

    def encode(
        self,
        x: torch.Tensor,
        n_q: tp.Optional[int] = None,
        first_layer_features: tp.Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # residual = x
        residual = x.clone()
        all_indices = []
        all_dists = []
        n_q = n_q or len(self.layers)
        for i, layer in enumerate(self.layers[:n_q]):
            if i == 0 and first_layer_features is not None:
                residual = first_layer_features
            elif i == 1 and first_layer_features is not None:
                residual = x - quantized

            indices, dist = layer.encode(residual, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            quantized = layer.decode(indices, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            residual = residual - quantized
            all_indices.append(indices)
            all_dists.append(dist)
        out_indices = torch.stack(all_indices)
        out_dists = torch.stack(all_dists)
        return out_indices, out_dists

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            quantized_out = quantized_out + quantized
        return quantized_out

class DistributedResidualVectorQuantization(nn.Module):
    """Efficient distributed residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *,
                 num_quantizers,
                 quantize_dropout: bool = False,
                 rand_num_quant: tp.Optional[tp.List] = None,
                 **kwargs):
        super().__init__()
        """
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        """
        codebook_size, codebook_dim = kwargs["codebook_size"], kwargs["dim"]
        kmeans_init = kwargs["kmeans_init"]
        if not kwargs["kmeans_init"]:
            embed = uniform_init(num_quantizers, codebook_size, codebook_dim)
        else:
            embed = torch.zeros(num_quantizers, codebook_size, codebook_dim)

        self.register_buffer("inited", torch.Tensor([[not kmeans_init] for _ in range(num_quantizers)]))
        self.register_buffer("cluster_size", torch.zeros(num_quantizers, codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

        self.q0_ds_ratio = 1
        if "q0_ds_ratio" in kwargs:
            self.q0_ds_ratio = kwargs.pop("q0_ds_ratio")

        self.layers = nn.ModuleList()
        for i in range(num_quantizers):
            vq_args = dict(**kwargs)
            feature_extractor_only_for_quant0 = vq_args.pop("feature_extractor_only_for_quant0", None)
            if i > 0 and feature_extractor_only_for_quant0:
                vq_args.pop("feature_extractor_config", None)
            vq = VectorQuantization(**vq_args)
            self.layers.append(vq)

        self.quantize_dropout = quantize_dropout
        self.rand_num_quant = rand_num_quant

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = torch.zeros_like(x)
        residual = x
        bb, cc, tt = x.shape
        device = x.device

        all_losses = []
        all_indices = []
        all_sub_quants = []
        n_q = n_q or len(self.layers)

        should_quantize_dropout = self.training and self.quantize_dropout and self.rand_num_quant is not None
        if should_quantize_dropout:
            rand_quantize_dropout_index = random.choice(self.rand_num_quant)

            null_indices_shape = (x.shape[0], x.shape[2])
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)
            null_loss = torch.full((1,), 0., device=device, dtype=x.dtype)
            null_sub_quant = torch.full(x.shape, -1, device=device, dtype=x.dtype)

        for quantizer_index, layer in enumerate(self.layers[:n_q]):
            # dropout except the first quantizer
            # print(666, quantizer_index)
            if should_quantize_dropout and quantizer_index >= rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                all_sub_quants.append(null_sub_quant)
                # print(666, quantizer_index, rand_quantize_dropout_index, self.rand_num_quant)
                continue

            quant_in = residual
            if self.q0_ds_ratio > 1 and quantizer_index == 0:
                quant_in = F.interpolate(quant_in, size=[tt//2])
            quantized, indices, loss = layer(quant_in, [
                self.inited[quantizer_index],
                self.cluster_size[quantizer_index],
                self.embed[quantizer_index],
                self.embed_avg[quantizer_index],
            ])
            if self.q0_ds_ratio > 1 and quantizer_index == 0:
                quantized = F.interpolate(quantized, size=[tt])
                indices = F.interpolate(indices.unsqueeze(1).float(), size=[tt]).squeeze(1).long()
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
            all_sub_quants.append(quantized)

        # sync buffers after one forward step
        distrib.broadcast_tensors(self.buffers())
        out_losses, out_indices, out_sub_quants = map(torch.stack, (all_losses, all_indices, all_sub_quants))

        return quantized_out, out_indices, out_losses, out_sub_quants

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for i, layer in enumerate(self.layers[:n_q]):
            indices = layer.encode(residual, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            quantized = layer.decode(indices, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            quantized_out = quantized_out + quantized
        return quantized_out