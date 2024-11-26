import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def lengths_to_padding_mask(
    lens: torch.Tensor, 
    padding_side: Optional[str] = "right",
    max_lens: Optional[int] = None,
):
    # bsz, max_lens = lens.size(0), torch.max(lens).item()
    bsz = lens.size(0)
    if max_lens is None:
        max_lens = torch.max(lens).item()
    if padding_side == "left":
        mask = torch.arange(start=max_lens-1, end=-1, step=-1).to(lens.device).view(1, max_lens)
        mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
        return mask
    if padding_side == "right":
        mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
        mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
        return mask
    raise ValueError(f"padding_side = {padding_side} is wrong.")


def lengths_to_attention_mask(
    lens: torch.Tensor,
    padding_side: Optional[str] = "right",
    max_lens: Optional[int] = None,
):
    return ~lengths_to_padding_mask(lens, padding_side, max_lens)


def get_feature_lengths(speech_lengths: torch.Tensor, hop_length: int) -> torch.Tensor:
    feature_lengths = speech_lengths // hop_length + 1
    return feature_lengths
