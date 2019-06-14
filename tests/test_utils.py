import sys; sys.path.append('.')
import torch

from src.utils import orthogonalize

def test_orthogonalization():
    v1 = torch.rand(10 ** 6)
    v2 = torch.rand(10 ** 6)

    assert torch.dot(v1, orthogonalize(v1, v2)).abs() <= 1e-2
    assert (v1.norm() - orthogonalize(v1, v2, adjust_len_to_v1=True).norm()) <= 1e-2
    assert (v2.norm() - orthogonalize(v1, v2, adjust_len_to_v2=True).norm()) <= 1e-2


def test_orthogonalization_several_times():
    for _ in range(10):
        test_orthogonalization()
