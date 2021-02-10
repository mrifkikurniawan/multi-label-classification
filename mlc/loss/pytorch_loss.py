from torch import nn

__all__ = ['cross_entropy', 'binary_cross_entropy']

def cross_entropy(**args):
    return nn.CrossEntropyLoss(**args)

def binary_cross_entropy(**args):
    return nn.BCELoss(**args)