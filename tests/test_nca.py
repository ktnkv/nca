import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nca import NCA, NEIGHBORHOODS

NAMES = list(NEIGHBORHOODS)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("invariant", [False, True])
@pytest.mark.parametrize("channels", [1, 4, 7])
def test_in_features(name, invariant, channels):
    nca = NCA(name, channels=channels, invariant=invariant)
    out = nca.perceive(torch.rand(2, channels, 9, 9))
    assert out.shape == (2, 9, 9, nca.in_features)


def test_moore_invariant_dim():
    assert NCA("moore", invariant=True).in_features == 36


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("padding", ["zeros", "circular"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotation_invariance(name, padding, k):
    nca = NCA(name, invariant=True, padding=padding)
    x = torch.rand(2, 4, 10, 10)
    a = nca.perceive(x)
    b = nca.perceive(torch.rot90(x, k, dims=(2, 3)))
    assert torch.allclose(torch.rot90(a, k, dims=(1, 2)), b, atol=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_raw_matches_unfold(name):
    nca = NCA(name)
    r, s = nca.radius, 2 * nca.radius + 1
    x = torch.rand(2, 3, 8, 8)
    nca = NCA(name, channels=3)
    ref = F.unfold(x, s, padding=r).view(2, 3, s * s, 8, 8)
    idx = [(dy + r) * s + dx + r for dy, dx in nca.offsets]
    ref = ref[:, :, idx].permute(0, 3, 4, 1, 2).reshape(2, 8, 8, -1)
    assert torch.allclose(nca.perceive(x), ref)


def test_non_closed_neighborhood_raises():
    with pytest.raises(ValueError):
        NCA(((0, 0), (0, 1)), invariant=True)


def test_residual():
    nca = NCA("moore", residual=True)
    x = torch.rand(2, 4, 8, 8)
    assert torch.equal(nca.process(lambda f: torch.zeros(*f.shape[:-1], 4), x), x)


@pytest.mark.parametrize("invariant", [False, True])
def test_process(invariant):
    nca = NCA("moore", invariant=invariant)
    net = nn.Linear(nca.in_features, nca.channels)
    x = torch.rand(2, 4, 8, 8, requires_grad=True)
    out = nca.process(net, x)
    assert out.shape == x.shape
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert nca.process(net, x[0]).shape == (4, 8, 8)
