from mlc.utils import *
from mlc import loss
from mlc import datasets

from torch import nn

def test_initialize_loss():
    module = loss
    losses_f = ['cross_entropy', 'binary_cross_entropy']

    for loss_f in losses_f:
        loss_f = initialize_loss(module, loss_f)
        assert callable(loss_f)
        assert isinstance(loss_f, nn.Module)

def test_initialize_dataset():
    NotImplemented    

def test_encode_multihot():
    index = [0, 2, 4]
    num_class = 10
    multihot = encode_multihot(index, num_class)
    
    assert isinstance(multihot, torch.Tensor)
    assert multihot.dtype is torch.float32
    assert list(multihot.shape) == [num_class]