import torch
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass


def list_mle(
    y_pred, y_true, 
    first_k: Optional[int] = None,
    attention_mask: Optional[torch.BoolTensor] = None, lengths: Optional[torch.LongTensor] = None
):
    # y_pred : batch_size, seq_len, num_logits
    batch_size, seq_len, num_logits = y_true.shape
    if first_k is not None:
        num_logits = first_k
    _, indices = y_true.sort(descending=False, dim=-1)
    # _, indices = y_true.topk(first_k, dim=-1)
    # indices = indices.flip(-1)
    # if first_k is not None:
    #     indices = indices[:, :, -first_k:]
    pred_sorted_by_true = y_pred.gather(dim=-1, index=indices)
    cumsums = pred_sorted_by_true.exp().cumsum(dim=-1) # 按照概率, 从小到大排序
    if first_k is not None:
        cumsums = cumsums[:, :, -first_k:]
        pred_sorted_by_true = pred_sorted_by_true[:, :, -first_k:]
    list_mle_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true
    if attention_mask is not None:
        # list_mle_loss = torch.where(attention_mask, list_mle_loss, 0)
        list_mle_loss = list_mle_loss * attention_mask.unsqueeze(-1)
    if lengths is not None:
        list_mle_loss = list_mle_loss.sum() / lengths.sum() / num_logits
    else:
        list_mle_loss = list_mle_loss.mean()
    return list_mle_loss


def apply_attention_mask_for_hidden(hidden, attn_masks):
    """
    将 attn_masks 中为 True 的元素对应的 hidden 中的值设为 0
    参数:
    hidden: [batch_size, seq_len, dim]
    attn_masks: [batch_size, seq_len]. True 表示 masked, False 表示 unmasked
    
    返回:
    masked_hidden: 应用了 mask 后的 hidden 张量
    """
    attn_masks = attn_masks.bool()
    
    # 扩展 attn_masks 的维度以匹配 hidden 的形状
    mask = attn_masks.unsqueeze(-1) # [batch_size, seq_len, 1]
    
    # 将 mask 扩展到与 hidden 相同的维度
    # 新形状: [batch_size, seq_len, dim]
    mask = mask.expand_as(hidden)
    
    # 创建一个与 hidden 相同形状的全零张量
    zeros = torch.zeros_like(hidden)
    
    # 使用 torch.where 来根据 mask 选择值
    # 如果 mask 为 True，选择 zeros；否则选择 hidden 中的原始值
    masked_hidden = torch.where(mask, zeros, hidden)
    
    return masked_hidden


def modify_attn_masks_with_static_frame_window_size(attn_masks, xlens, window_size, center_window: Optional[bool] = None):
    batch_size, seq_len, _ = attn_masks.shape
    device = attn_masks.device

    # 创建时间步索引
    t_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) 
    # [[0, 1, ..., seq_len - 1], ..., [0, 1, ..., seq_len - 1]], batch_size x seq_len
    xlens_expanded = xlens.unsqueeze(1).expand(-1, seq_len)
    # [[3, 3, 3, 3, 3], ..., [5, 5, 5, 5, 5]], batch_size x seq_len

    # 创建需要被遮蔽的区域
    start = xlens_expanded.unsqueeze(2) # [[[3], [3], [3], [3], [3]], ..., [[5], [5], [5], [5], [5]]], batch_size x seq_len x 1
    end = (t_indices - window_size).clamp(min=0).unsqueeze(2) 
    # [[[0], [0], [0], [3], [4]], ..., [[0], [0], [0], [0], [0]]], batch_size x seq_len x 1
    mask_range = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
    # batch_size x seq_len x seq_len
    
    if not center_window:
        mask = (mask_range >= start) & (mask_range <= end)
    else:
        mask = (mask_range >= start) & ((mask_range <= end) | (mask_range >= end + window_size + window_size))

    attn_masks = attn_masks.clone()
    attn_masks[mask] = True
    # print(attn_masks.shape, xlens, attn_masks[0, -1, :].sum(), (~(attn_masks[0, -1, :].bool())).sum())
    # logging.info(f"{attn_masks.shape} {xlens} {attn_masks[0, -1, :].sum()}")

    return attn_masks


def select_embeds(
    inputs_embeds, position_ids, 
    attention_mask=None, padding_value=0,
    max_num: Optional[int] = None, num_elements_type: Optional[str] = None,
):
    """
    inputs_embeds: [batch_size, seq_len, dim]
    num_elements_type: static, dynamic
    """
    if num_elements_type is None or num_elements_type == "static":
        return select_embeds_from_static_num_elements(
            inputs_embeds=inputs_embeds, position_ids=position_ids, attention_mask=attention_mask, padding_value=padding_value,
        )
    elif num_elements_type == "dynamic":
        return select_embeds_from_dynamic_num_elements(
            inputs_embeds=inputs_embeds, position_ids=position_ids, max_num=max_num, padding_value=padding_value,
        )


def select_embeds_from_static_num_elements(inputs_embeds, position_ids, attention_mask=None, padding_value=0):
    """
    inputs_embeds: [batch_size, seq_len, dim]
    position_ids: [batch_size, seq_len]
    """
    batch_size, seq_len, dim = inputs_embeds.shape
    selected_embeds = torch.gather(inputs_embeds, dim=1, index=position_ids[:, :, None].expand(-1, -1, dim))
    if attention_mask is not None:
        selected_embeds.masked_fill_(~attention_mask[:, :, None].expand(-1, -1, dim).bool(), value=padding_value)
    return selected_embeds


def select_embeds_from_dynamic_num_elements(inputs_embeds, position_ids, max_num: Optional[int] = None, padding_value=0):
    """
    inputs_embeds: [batch_size, seq_len, dim]
    position_ids: [num_elements, 2]
    """
    batch_size, seq_len, dim = inputs_embeds.shape
    n = position_ids.shape[0]
    batch_indices, seq_indices = position_ids[:, 0], position_ids[:, 1]
    selected_vector = inputs_embeds[batch_indices, seq_indices]
    
    # 构建输出张量，形状为 [batch_size, max_num, dim]
    # max_num 为最大选择数量
    if max_num is None:
        unique_elements, counts = torch.unique(batch_indices, return_counts=True)
        max_num = torch.max(counts).item()
    selected_embeds = torch.full(size=(batch_size, max_num, dim), fill_value=padding_value, device=inputs_embeds.device)
    current_index = 0
    for i in range(batch_size):
        mask = batch_indices == i
        num_selected = mask.sum().item()
        # print(num_selected, mask)
        selected_embeds[i, :num_selected] = selected_vector[current_index: current_index + num_selected]
        current_index += num_selected
    return selected_embeds


def get_past_key_values_with_lens(past_key_values, prefix_lens: int, cur_lens: Optional[int] = None, cur_position_id: Optional[int] = -1):
    """
    past_key_values: [layer_0_past_key_value, ..., layer_11_past_key_value]
    layer_0_past_key_value: [key_state, value_state]
    key_state: [batch_size, num_heads, seq_len, dim // num_heads]
    """
    batch_size = past_key_values[0][0].shape[0]
    past_key_values = list(past_key_values)
    if isinstance(prefix_lens, int):
        pass
    prefix_len = prefix_lens
    cur_len = cur_lens
    num_layers = len(past_key_values)
    for layer_id in range(num_layers):
        key_state, value_state = past_key_values[layer_id]
        if cur_len is not None:
            key_state = torch.cat([key_state[:, :, :prefix_len, :], key_state[:, :, cur_position_id - cur_len + 1:, :]], dim=2)
            value_state = torch.cat([value_state[:, :, :prefix_len, :], value_state[:, :, cur_position_id - cur_len + 1:, :]], dim=2)
        else:
            key_state = key_state[:, :, :prefix_len, :]
            value_state = value_state[:, :, :prefix_len, :]
        past_key_values[layer_id] = (key_state, value_state)
    return tuple(past_key_values)


def modify_with_extra_mask_interval(attn_masks, extra_mask_interval, prefix_len: Optional[int] = None, fill_value: Optional[int] = 1):
    if isinstance(extra_mask_interval, list) or isinstance(extra_mask_interval, tuple):
        batch_size, _, seq_len = attn_masks.shape
        begin, end = extra_mask_interval
        if begin <= 0 and prefix_len is not None:
            begin = prefix_len + begin
        if end < 0 and prefix_len is not None:
            end = prefix_len + end
        if end >= seq_len:
            end = seq_len - 2
        if begin < end:
            # attn_masks[:, -1, begin: end] = fill_value
            attn_masks[:, :, begin: end] = fill_value
            # print(f"attn_masks = {attn_masks.shape}, begin = {begin}, end = {end}")
    return attn_masks
    

@dataclass
class MergeStrategy:

    apply_frame_window: Optional[bool] = False
    frame_window_size: Optional[int] = None # number of frames
    frame_window_type: Optional[str] = None # static, static_center, dynamic
    only_frame_window: Optional[bool] = False
    frame_window_dropout: Optional[float] = 1.0 # only for training. None for no dropout. x means x for dropout, and 1 - x for reserve
    frame_window_dropout_inference: Optional[float] = 1.0

    apply_unit_window: Optional[bool] = False
    unit_window_size: Optional[int] = None # number of units
    unit_window_type: Optional[str] = None # static, dynamic
    only_unit_window: Optional[bool] = False

    windows_order: Optional[Sequence] = None # [unit_window, frame_window]

    def __post_init__(self):
        if self.windows_order is None:
            self.apply_frame_window = None
            self.apply_unit_window = None
        if self.frame_window_dropout_inference is None:
            frame_window_dropout_inference = self.frame_window_dropout
        for window_name in self.windows_order:
            setattr(self, f"apply_{window_name}", True)

    @torch.no_grad()
    def merge_frame_window_for_inference(
        self, attention_mask: torch.BoolTensor, prefix_len: int, window_size: Optional[int] = None,
        dropout: Optional[float] = None, training: Optional[None] = True,
    ) -> torch.BoolTensor:
        """
        not support batch
        attention_mask: 0 for unmasked, 1 for masked
        prefix (phoneme + prompt) is unmasked
        """
        if window_size is None:
            window_size = self.frame_window_size
        modified_attention_mask = modify_attn_masks_with_static_frame_window_size(
            attention_mask.clone(), torch.tensor([prefix_len], dtype=torch.long, device=attention_mask.device), window_size,
        )
        return self.apply_frame_window_dropout(attention_mask, modified_attention_mask, dropout=dropout, training=training)
        
        
        cur_len = attention_mask.shape[1]

        if prefix_len + window_size >= cur_len:
            return attention_mask
        
        attn_masks = attention_mask.clone()
        # attn_masks_clone = attn_masks[0, prefix_len: cur_len - window_size, :].clone()
        attn_masks[0, prefix_len: cur_len - window_size + 1, :] = True
        # attn_masks_slice = attn_masks[0, prefix_len: cur_len - window_size, :]           
        # print(f"mask, start = {cur_len - window_size}, end = {cur_len}, attn_masks_clone = {attn_masks_clone.sum()}, attn_masks_slice = {attn_masks_slice.sum()}")
        # attn_masks_clone = attn_masks[0, :, prefix_len: cur_len - window_size].clone()
        attn_masks[0, :, prefix_len: cur_len - window_size + 1] = True
        # attn_masks_slice = attn_masks[0, :, prefix_len: cur_len - window_size]
        # print(f"attn_masks_clone = {attn_masks_clone.sum()}, attn_masks_slice = {attn_masks_slice.sum()}")
        # return attn_masks
        return self.apply_frame_window_dropout(attention_mask, attn_masks, dropout=dropout, training=training)


    @torch.no_grad()
    def merge_frame_window(
        self, attention_mask: torch.BoolTensor, xlens: torch.LongTensor, window_size: Optional[int] = None, 
        dropout: Optional[float] = None, training: Optional[None] = True,
    ) -> torch.BoolTensor:
        """
        attention_mask: 0 for unmasked, 1 for masked
        xlens (phoneme + BOS) is unmasked
        """
        if window_size is None:
            window_size = self.frame_window_size
        center_window = self.frame_window_type == "static_center"
        modified_attention_mask = modify_attn_masks_with_static_frame_window_size(attention_mask.clone(), xlens, window_size, center_window=center_window)
        return self.apply_frame_window_dropout(attention_mask, modified_attention_mask, dropout=dropout, training=training)

    @torch.no_grad()
    def apply_frame_window_dropout(
        self, original_attention_mask: torch.BoolTensor, modified_attention_mask: torch.BoolTensor, 
        dropout: Optional[float] = None, training: Optional[None] = True,
    ):
        if dropout is None:
            if training:
                dropout = self.frame_window_dropout
            else:
                dropout = self.frame_window_dropout_inference
        if dropout is None or dropout <= 0.0:
            return original_attention_mask
        batch_size, seq_length, seq_length = original_attention_mask.shape
        device = original_attention_mask.device
        prob = torch.rand(batch_size, device=device)
        mask = prob <= dropout
        # print(666, training, prob, dropout, mask)
        new_attention_mask = torch.where(
            mask.view(batch_size, 1, 1),
            modified_attention_mask,
            original_attention_mask,
        )
        return new_attention_mask
