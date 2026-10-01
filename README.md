# nca

Neural cellular automata on PyTorch, with a pluggable neighborhood and optional
rotation-invariant inputs.

`NCA` handles the cellular-automaton side: it gathers each cell's neighborhood
(vectorized, no per-pixel loops) and feeds it to **your** network. The network is
an external dependency and knows nothing about images or geometry.

## Installation

```bash
pip install .          # or: uv add .
```

Development setup:

```bash
uv sync
uv run pytest
uv build               # sdist + wheel in dist/
```

Requires Python 3.12+ and PyTorch.

## Quick start

```python
import torch
from torch import nn
from nca import NCA

nca = NCA("moore", channels=4, invariant=True)

net = nn.Sequential(
    nn.Linear(nca.in_features, 64),
    nn.ReLU(),
    nn.Linear(64, nca.channels),
)

img = torch.rand(1, 4, 64, 64)   # (B, C, H, W)
for _ in range(10):
    img = nca.process(net, img)  # one CA step, same shape
```

## How it works

For every cell, `NCA.perceive` builds a feature vector from its neighborhood;
`net` maps it to the cell's next state:

```
(B, C, H, W) --perceive--> (B, H, W, in_features) --net--> (B, H, W, C) --> (B, C, H, W)
```

The network contract is pointwise: `(..., in_features) -> (..., channels)`.
Any `nn.Module` (or callable) with that signature works: `nn.Linear`, an MLP, etc.

## Options

| Argument       | Default   | Meaning                                                                              |
| -------------- | --------- | ------------------------------------------------------------------------------------ |
| `neighborhood` | `"moore"` | Name from `NEIGHBORHOODS`, or a custom tuple of `(dy, dx)` offsets, `(0, 0)` first.  |
| `channels`     | `4`       | State size of one cell (4 = RGBA, 1 = grayscale, any number of hidden channels).     |
| `invariant`    | `False`   | Use rotation-invariant statistics instead of raw neighbor values (see below).        |
| `padding`      | `"zeros"` | Border handling: `"zeros"`, `"circular"`, `"reflect"`, `"replicate"`.                |
| `residual`     | `False`   | The net predicts a delta; `process` returns `img + delta` instead of the new state.  |

Attributes: `channels` (state size per cell) and `in_features` (net input size per cell).

## Neighborhoods

Defined in [`src/nca/neighborhoods.py`](src/nca/neighborhoods.py):

| Name             | Cells | Shape                                                          |
| ---------------- | ----- | -------------------------------------------------------------- |
| `von_neumann`    | 5     | Radius-1 diamond                                               |
| `moore`          | 9     | 3x3 square                                                     |
| `von_neumann_r2` | 13    | Radius-2 diamond                                               |
| `moore_r2`       | 25    | 5x5 square                                                     |
| `ring_r2`        | 17    | Center plus the hollow ring at distance 2 (skips direct neighbors) |
| `cross_r3`       | 13    | Axes only, arm length 3 (no diagonals)                         |

Custom neighborhoods:

```python
NCA(((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)))
```

## Rotation-invariant inputs

With `invariant=True`, raw neighbor values are replaced by per-channel statistics
(`min`, `max`, `mean`, `std`) over groups of cells that map onto each other under
90/180/270 degree rotations. The groups are derived from the neighborhood itself,
nothing is hardcoded. For Moore these are the 4 edge cells and the 4 diagonal cells.
The center cell is passed as is.

`in_features` is `channels * K` in raw mode (K = cells in the neighborhood) and
`channels * (1 + 4 * G)` with `invariant=True` (G = number of groups). Every group has
4 cells and there are 4 statistics, so the two are always equal (e.g. 36 for Moore
with 4 channels); always use `nca.in_features` rather than computing it yourself.

The neighborhood must be closed under 90 degree rotation, otherwise `ValueError` is raised.

## Performance

`perceive` pads the image once and stacks shifted views, so everything is a few
large tensor operations. Rough numbers on CPU (batch 8, 256x256, `moore_r2`):
about 120 ms per step raw and about 280 ms with `invariant=True`.
Run `uv run python benchmarks/bench.py` to measure on your machine.

## License

[MIT](LICENSE)
