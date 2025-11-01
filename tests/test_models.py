import torch
from genome_anno.dl.models import CNN1D

def test_forward_shape():
    m = CNN1D()
    x = torch.randn(8, 4, 256)
    y = m(x)
    assert y.shape == (8, 2)
