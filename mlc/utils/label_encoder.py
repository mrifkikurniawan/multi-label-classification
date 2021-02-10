import torch
from torch import nn
from typing import List

def encode_multihot(idx: List[int], num_class: int):
    labels = torch.LongTensor(idx)
    one_hot = nn.functional.one_hot(labels, num_classes=num_class)
    multi_hot = one_hot.sum(dim=0).float() 
    return multi_hot