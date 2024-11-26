import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def _topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0, died_range=None):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # constrained sampling
    if died_range is not None:
        assert isinstance(died_range, tuple) and len(died_range) == 2
        if died_range[1] > died_range[0] and died_range[0] >= 0:
            logits[...,died_range[0]:died_range[1]] = -float("Inf")
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


def topk_sampling(
    logits, top_k=10, top_p=1.0, temperature=1.0, died_range=None,
    topk_sampling_strategy: Optional[Dict] = dict(multinomial=True),
):
    """
        temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        top_k: (`optional`) int
            The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
        top_p: (`optional`) float
            The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.
        topk_sampling_strategy:
            top2_threshold
            multinomial
            without_multinomial
            exclude_top1
    """
    # topk_sampling_strategy = dict(top2_threshold=0.35, multinomial=True)

    top2_threshold = 0.0
    # print(666, logits.shape)
    if topk_sampling_strategy is not None:
        top2_threshold = topk_sampling_strategy.get("top2_threshold", 0.0)
    # top2_threshold = 0.35
    if top2_threshold > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_logits = F.softmax(sorted_logits, dim=-1)
        if sorted_logits[0, 1] >= top2_threshold:
            # print(f"logits = {logits.shape} top_2 = {sorted_logits[0, 1]} >= {top2_threshold}")
            return sorted_indices[:1, 1:2]
            # top_k, top_p = -100, 1.0
            # return _topk_sampling(logits, top_k, top_p, temperature, died_range)

    if topk_sampling_strategy is not None and topk_sampling_strategy.get("exclude_top1", False):
        _, max_indices = torch.max(logits, dim=-1, keepdim=True)
        logits.scatter_(-1, max_indices, float('-inf'))

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # constrained sampling
    if died_range is not None:
        assert isinstance(died_range, tuple) and len(died_range) == 2
        if died_range[1] > died_range[0] and died_range[0] >= 0:
            logits[...,died_range[0]:died_range[1]] = -float("Inf")
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    if topk_sampling_strategy is None or topk_sampling_strategy.get("multinomial", False):
        # Sample
        token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    elif topk_sampling_strategy.get("without_multinomial", True):
        token = F.softmax(logits, dim=-1).argmax(-1, keepdim=True)
    # print(f"output_token = {token.shape}")
    return token