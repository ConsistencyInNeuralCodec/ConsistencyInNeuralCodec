import numpy as np
import torch
import torch.nn.functional as F
import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .beam_search import beam_search


class EncodingPathConfig(PretrainedConfig):
    def __init__(
        self,
        strategy: Optional[str] = "greedy",
        run_time: Optional[Sequence] = ["eval"],
        codebook_idx: Optional[int] = None,
        apply_for_codebooks: Optional[Dict] = {},
        # beam search
        num_beams: Optional[int] = None,
        bos_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        # kmeans search
        l2_distance_threshold: Optional[float] = None,
        **kwargs
    ):
        """
        strategy:
            greedy, beam_search, kmeans_search
        apply_for_codebooks:
            {0: 7.0, 1: 6.0, 2: 5.0}
        run_time: [train, eval, test]
        """
        super().__init__(
            strategy=strategy,
            run_time=run_time,
            codebook_idx=codebook_idx,
            apply_for_codebooks=apply_for_codebooks,
            num_beams=num_beams,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            l2_distance_threshold=l2_distance_threshold,
        )


def get_logits_from_dist(
    dist: Optional[torch.FloatTensor] = None,
    # attention_mask: Optional[torch.BoolTensor] = None,
):
    logits = dist - torch.max(dist, dim=-1, keepdim=True).values.detach() # [batch_size, seq_len, codebook_size]
    # p = F.log_softmax(logits, dim=-1)
    return logits


class Interval:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.length = end - begin
        
    def __repr__(self):
        return f"begin: {self.begin}, end = {self.end}, length = {self.length}"


class BaseEncodingPathModel:
    def __init__(self, config: EncodingPathConfig, **kwargs):
        self.config = config
    
    @staticmethod
    def build_model(config: EncodingPathConfig, **kwargs):
        config = copy.deepcopy(config)
        config.codebook_idx = kwargs.pop("codebook_idx", None)
        if isinstance(config, dict):
            config = EncodingPathConfig(**config)
        if config.strategy == "greedy":
            return GreedyEncodingPathModel(config=config)
        elif config.strategy == "beam_search":
            return BeamSearchEncodingPathModel(config=config)
        elif config.strategy == "kmeans_search":
            return KmeansSearchEncodingPathModel(config=config)
        raise NotImplementedError

    def encode(
        self,
        dist: torch.FloatTensor,
        **kwargs
    ):
        """
        dist: [batch_size, seq_len, codebook_size]
        """
        raise NotImplementedError


class GreedyEncodingPathModel(BaseEncodingPathModel):
    def encode(
        self,
        dist: torch.FloatTensor,
        **kwargs
    ):
        """
        dist: [batch_size, seq_len, codebook_size]
        """
        # print(f"greedy encoding path")
        return dist.max(dim=-1).indices


class BeamSearchEncodingPathModel(BaseEncodingPathModel):
    def encode(
        self,
        dist: torch.FloatTensor,
        **kwargs
    ):
        """
        dist: [batch_size, seq_len, codebook_size]
        """
        # print(f"beam search encoding path")
        batch_size, seq_len, codebook_size = dist.shape
        logits = get_logits_from_dist(dist)
        decoded = beam_search(
            logits=logits,
            num_beams=self.config.num_beams,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            max_length=seq_len + 1,
            **kwargs
        )
        return decoded


class KmeansSearchEncodingPathModel(BaseEncodingPathModel):
    def encode_for_single_sample(
        self,
        dist: torch.FloatTensor,
        distance: torch.FloatTensor,
        embed_ind: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """
        distance: [seq_len, seq_len]
        """
        # distance = distance.cpu()
        # device = distance.device
        distance = distance.detach().clone().cpu().numpy()
        l2_distance_threshold = None
        if self.config.codebook_idx is not None and self.config.codebook_idx in self.config.apply_for_codebooks:
            l2_distance_threshold = self.config.apply_for_codebooks[self.config.codebook_idx]
        if self.config.l2_distance_threshold is not None:
            l2_distance_threshold = self.config.l2_distance_threshold
        cmp = distance <= l2_distance_threshold
        # 假设 cmp 是一个已经给定的形状为 [seq_len, seq_len] 的布尔矩阵
        seq_len = cmp.shape[0]
        # dp = torch.zeros((seq_len, seq_len), dtype=torch.long, device=device)
        dp = np.zeros((seq_len, seq_len), dtype=int)
        max_submatrices = []

        # 构建动态规划表，并且找到最大子方阵
        for i in range(seq_len):
            for j in range(seq_len):
                if cmp[i][j]:
                    # 初始方阵大小为1
                    dp[i][j] = 1
                    if i > 0 and j > 0:
                        # 如果可能，增大子方阵
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    # 检查其他元素并更新最大子方阵列表
                    curr_size = dp[i][j]
                    if curr_size > 1:
                        submatrix = cmp[i-curr_size + 1: i + 1, j - curr_size + 1: j + 1]
                        if submatrix.all():
                            max_submatrices.append((i - curr_size + 1, j - curr_size + 1, i, j))

        # 输出所有找到的最大子方阵区间
        _merged_max_submatrices = []
        for idx, (top_i, left_j, bottom_i, right_j) in enumerate(max_submatrices):
            if top_i != left_j or bottom_i != right_j:
                continue
            _merged_max_submatrices.append([top_i, left_j, bottom_i, right_j])
            
        merged_max_submatrices = []
        for idx, (top_i, left_j, bottom_i, right_j) in enumerate(_merged_max_submatrices):
            if idx + 1 < len(_merged_max_submatrices):
                # print(max_submatrices[idx + 1][:2], [top_i, left_j])
                if _merged_max_submatrices[idx + 1][:2] == [top_i, left_j]:
                    continue
            if len(merged_max_submatrices) > 0 and merged_max_submatrices[-1][-1] >= top_i:
                continue
            merged_max_submatrices.append([top_i, left_j, bottom_i, right_j])
            # print(f"Top-left: (row={top_i}, col={left_j}), Bottom-right: (row={bottom_i}, col={right_j})")
        interval_list = []
        prev_time_step = 0
        for top_i, left_j, bottom_i, right_j in merged_max_submatrices:
            cur_time_step = top_i
            if prev_time_step < cur_time_step:
                for time_step in range(prev_time_step, cur_time_step):
                    interval_list.append(Interval(time_step, time_step + 1))
            interval_list.append(Interval(top_i, bottom_i + 1))
            prev_time_step = bottom_i + 1
        for time_step in range(interval_list[-1].end, seq_len):
            interval_list.append(Interval(time_step, time_step + 1))
        codewords = []
        if embed_ind is None:
            embed_ind = dist.argmax(-1)
        for interval in interval_list:
            codeword = embed_ind[interval.begin]
            for time_step in range(interval.begin, interval.end):
                codewords.append(codeword)
        return codewords

    def encode(
        self,
        hidden_state: torch.FloatTensor,
        dist: torch.FloatTensor,
        embed_ind: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """
        dist: [batch_size, seq_len, codebook_size]
        hidden_state: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, codebook_size = dist.shape
        batch_size, seq_len, dim = hidden_state.shape
        device = hidden_state.device

        expanded_hidden_state_a = hidden_state.unsqueeze(2).expand(batch_size, seq_len, seq_len, dim)
        expanded_hidden_state_b = hidden_state.unsqueeze(1).expand(batch_size, seq_len, seq_len, dim)
        diff = expanded_hidden_state_a - expanded_hidden_state_b
        distance = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        codewords_batch = []
        for i in range(batch_size):
            codewords = self.encode_for_single_sample(dist=dist[i], distance=distance[i], embed_ind=embed_ind[i])
            # print(self.config.codebook_idx, i, len(codewords))
            codewords_batch.append(codewords)
        codewords_batch = torch.tensor(codewords_batch, dtype=torch.long, device=device)
        # print(self.config.codebook_idx, codewords_batch.shape)
        return codewords_batch
    