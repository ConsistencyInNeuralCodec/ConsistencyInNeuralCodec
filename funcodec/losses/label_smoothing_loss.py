#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module."""

from typing import List

import torch
from torch import nn
from funcodec.modules.nets_utils import make_pad_mask


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
    ):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


class SequenceBinaryCrossEntropy(nn.Module):
    def __init__(
            self,
            normalize_length=False,
            criterion=nn.BCEWithLogitsLoss(reduction="none")
    ):
        super().__init__()
        self.normalize_length = normalize_length
        self.criterion = criterion

    def forward(self, pred, label, lengths):
        pad_mask = make_pad_mask(lengths, maxlen=pred.shape[1]).to(pred.device)
        loss = self.criterion(pred, label)
        denom = (~pad_mask).sum() if self.normalize_length else pred.shape[0]
        return loss.masked_fill(pad_mask.unsqueeze(-1), 0).sum() / denom


class SequenceCrossEntropy(nn.Module):
    def __init__(
            self,
            normalize_length=False,
            ignore_index=-100,
    ):
        super().__init__()
        self.normalize_length = normalize_length
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, pred, label, lengths=None, masks=None):
        if masks is None:
            assert lengths is not None
            masks = make_pad_mask(lengths).to(pred.device)
        loss = self.criterion(pred.reshape(-1, pred.size(-1)), label.reshape(-1))
        masks = masks.reshape(-1)
        denom = (~masks).sum() if self.normalize_length else pred.shape[0]
        return (loss * (~masks)).sum() / denom

@torch.no_grad()
def calc_topk_accuracy(output: torch.Tensor, target: torch.Tensor, masks: torch.Tensor = None, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """

    output = output.reshape(-1, output.size(-1))
    target = target.reshape(-1)
    if masks is not None:
        masks = masks.reshape(-1)

    # ---- get the topk most likely labels according to your model
    # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
    maxk = max(topk)  # max number labels we will consider in the right choices for out model
    batch_size = target.size(0)

    # get top maxk indicies that correspond to the most likely probability scores
    # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
    _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
    y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

    # - get the credit for each example if the models predictions is in maxk values (main crux of code)
    # for any example, the model will get credit if it's prediction matches the ground truth
    # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
    # if the k'th top answer of the model matches the truth we get 1.
    # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
    target_reshaped = target.reshape(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
    # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
    correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
    # original: correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    # -- get topk accuracy
    list_topk_accs = []  # idx is topk1, topk2, ... etc
    for k in topk:
        # get tensor of which topk answer was right
        ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
        # flatten it to help compute if we got it correct for each example in batch

        flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.float().sum(dim=0)  # [k, B] -> [B]
        if masks is not None:
            flattened_indicator_which_topk_matched_truth = flattened_indicator_which_topk_matched_truth * masks

        # get if we got it right for any of our top k prediction for each example in batch
        tot_correct_topk = flattened_indicator_which_topk_matched_truth.sum(dim=0, keepdim=True)  # [B] -> [1]
        # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
        if masks is not None:
            topk_acc = tot_correct_topk / masks.sum()  # topk accuracy for entire batch

        else:
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
        list_topk_accs.append(topk_acc)
    return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]
