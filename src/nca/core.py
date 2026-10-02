import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .neighborhoods import NEIGHBORHOODS, Offsets

_PAD_MODES = {"zeros": "constant", "circular": "circular", "reflect": "reflect", "replicate": "replicate"}
_N_STATS = 4  # min, max, mean, std
_EPS = 1e-12  # keeps std differentiable on constant neighborhoods


def _rot90(o: tuple[int, int]) -> tuple[int, int]:
    return -o[1], o[0]


def _resolve_offsets(neighborhood: str | Offsets) -> Offsets:
    """Look up a named neighborhood or validate a custom one."""
    if isinstance(neighborhood, str):
        if neighborhood not in NEIGHBORHOODS:
            raise ValueError(f"unknown neighborhood {neighborhood!r}, available: {list(NEIGHBORHOODS)}")
        return NEIGHBORHOODS[neighborhood]
    try:
        offsets = tuple((int(dy), int(dx)) for dy, dx in neighborhood)
    except (TypeError, ValueError) as e:
        raise ValueError("neighborhood must be a name or a sequence of (dy, dx) int pairs") from e
    if not offsets or offsets[0] != (0, 0):
        raise ValueError("(0, 0) must be the first offset")
    if len(set(offsets)) != len(offsets):
        raise ValueError("neighborhood contains duplicate offsets")
    return offsets


def _orbits(offsets: Offsets) -> list[list[int]]:
    """Orbits of the non-center offsets under 90-degree rotation (indices into offsets)."""
    index = {o: i for i, o in enumerate(offsets)}
    orbits, seen = [], set()
    for o in offsets[1:]:
        if o in seen:
            continue
        orbit = [o]
        for _ in range(3):
            orbit.append(_rot90(orbit[-1]))
        if any(p not in index for p in orbit):
            raise ValueError(f"neighborhood is not closed under 90-degree rotation: {o}")
        seen.update(orbit)
        orbits.append([index[p] for p in orbit])
    return orbits


class NCA(nn.Module):
    """Neural cellular automaton: gathers neighborhoods, applies an external net per cell.

    Args:
        neighborhood: name from ``NEIGHBORHOODS`` or a custom tuple of (dy, dx) offsets
            with (0, 0) first.
        channels: state size of a single cell (e.g. 4 for RGBA).
        invariant: replace raw neighbor values with rotation-invariant statistics
            (min, max, mean, std) of each rotation orbit, per channel.
        padding: border handling: "zeros", "circular", "reflect" or "replicate".
        residual: if True, the net predicts a delta and ``process`` returns ``img + delta``.

    Attributes:
        in_features: size of the net's input vector for a single cell.
    """

    def __init__(
        self,
        neighborhood: str | Offsets = "moore",
        channels: int = 4,
        invariant: bool = False,
        padding: str = "zeros",
        residual: bool = False,
    ):
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError(f"channels must be a positive int, got {channels!r}")
        if padding not in _PAD_MODES:
            raise ValueError(f"padding must be one of {list(_PAD_MODES)}, got {padding!r}")
        for name, flag in (("invariant", invariant), ("residual", residual)):
            if not isinstance(flag, bool):
                raise TypeError(f"{name} must be a bool, got {flag!r}")
        offsets = _resolve_offsets(neighborhood)
        if invariant and len(offsets) == 1:
            raise ValueError("invariant=True needs at least one non-center cell")

        self.residual = residual
        self.offsets, self.channels, self.invariant = offsets, channels, invariant
        self.radius = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
        self.padding = _PAD_MODES[padding]

        if invariant:
            orbits = torch.tensor(_orbits(offsets))  # (G, 4)
            self.register_buffer("orbits", orbits, persistent=False)
            self.in_features = channels * (1 + len(orbits) * _N_STATS)
        else:
            self.in_features = channels * len(offsets)

    def perceive(self, img: Tensor) -> Tensor:
        """(B, C, H, W) -> (B, H, W, in_features)."""
        B, C, H, W = img.shape
        r = self.radius
        x = F.pad(img, (r, r, r, r), mode=self.padding)
        # Cell axis first so gathers/reductions run over contiguous blocks.
        nbr = torch.stack([x[..., r + dy : r + dy + H, r + dx : r + dx + W] for dy, dx in self.offsets])
        # nbr: (K, B, C, H, W)
        if self.invariant:
            g = nbr[self.orbits]  # (G, 4, B, C, H, W)
            mean = g.mean(1)
            std = ((g - mean.unsqueeze(1)).square().mean(1) + _EPS).sqrt()  # much faster than g.var
            nbr = torch.cat([nbr[:1], g.amin(1), g.amax(1), mean, std])
        return nbr.permute(1, 3, 4, 2, 0).reshape(B, H, W, self.in_features)

    def process(self, net: nn.Module, img: Tensor) -> Tensor:
        """One CA step. ``net`` maps (..., in_features) -> (..., channels) per cell
        (the new state, or a delta if ``residual``).

        img: (B, C, H, W) or (C, H, W); returns the same shape.
        """
        batched = img.ndim == 4
        if not batched:
            img = img.unsqueeze(0)
        out = net(self.perceive(img)).permute(0, 3, 1, 2)
        if self.residual:
            out = img + out
        return out if batched else out.squeeze(0)
