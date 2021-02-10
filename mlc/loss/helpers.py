from typing import Union, List, Dict

import torch
from torch import nn
from mlc.utils import initialize_loss
from mlc import loss

def aggregate_loss(loss_fs: List[Dict], preds: torch.Tensor, targets: torch.Tensor, prefix: str):
    losses = dict()
    prefix = str(prefix)

    # looping through all loss functions
    for loss_f in loss_fs:
        loss_weight = loss_f['weight'].type_as(targets)
        loss_f = loss_f["module"]
        loss_name = loss_f.__class__.__name__
        loss_val = loss_weight * loss_f(preds, targets)
        losses[f"{prefix}_{loss_name}"] = loss_val
    
    # sum up the all loss
    losses[f'{prefix}_total_loss'] = sum([losses[loss_name] for loss_name in losses.keys()])
    return losses

def generate_loss(loss_cfg: Union[List, Dict, str]) -> dict:
    loss_fs = list()

    if isinstance(loss_cfg, str):
        # set weight to 1 if w is not pre-defined
        loss_f = dict(module = initialize_loss(loss, loss_cfg), weight=torch.tensor([1], dtype=torch.float32))
        loss_fs.append(loss_f)

    elif isinstance(loss_cfg, list):
        for single_loss in loss_cfg:
            if isinstance(single_loss, str):
                # set weight to 1 if w is not pre-defined
                loss_f = dict(module = initialize_loss(loss, single_loss), weight=torch.tensor([1], dtype=torch.float32))
                loss_fs.append(loss_f)
            elif isinstance(single_loss, dict):
                loss_w = single_loss["weight"]
                loss_w = torch.tensor([loss_w], dtype=torch.float32)
                loss_f = dict(module = initialize_loss(loss, single_loss["module"]), weight=loss_w)
                loss_fs.append(loss_f)

    elif isinstance(loss_cfg, dict):
        loss_w = loss_cfg["weight"]
        loss_w = torch.tensor([loss_w], dtype=torch.float32)
        loss_f = dict(module = initialize_loss(loss, loss_cfg["module"]), weight=loss_w)
        loss_fs.append(loss_f)
    
    return loss_fs