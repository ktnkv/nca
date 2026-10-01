"""Neighborhood definitions.

A neighborhood is a tuple of ``(dy, dx)`` offsets. The center ``(0, 0)`` is
always first. All built-in neighborhoods are closed under 90-degree rotation.
"""

from itertools import product

Offsets = tuple[tuple[int, int], ...]


def _build(cells) -> Offsets:
    rest = sorted(c for c in cells if c != (0, 0))
    return ((0, 0), *rest)


def _grid(r: int):
    return product(range(-r, r + 1), repeat=2)


def von_neumann(r: int = 1) -> Offsets:
    """Diamond: |dy| + |dx| <= r."""
    return _build(c for c in _grid(r) if abs(c[0]) + abs(c[1]) <= r)


def moore(r: int = 1) -> Offsets:
    """Square: max(|dy|, |dx|) <= r."""
    return _build(_grid(r))


def ring(r: int = 2) -> Offsets:
    """Center plus the hollow square ring at Chebyshev distance exactly r.

    Skips the immediate neighbors, so the rule only sees distant cells.
    """
    return _build(c for c in _grid(r) if max(abs(c[0]), abs(c[1])) == r or c == (0, 0))


def cross(r: int = 3) -> Offsets:
    """Axis-aligned plus sign of arm length r (no diagonals at all)."""
    return _build([(d, 0) for d in range(-r, r + 1)] + [(0, d) for d in range(-r, r + 1)])


NEIGHBORHOODS: dict[str, Offsets] = {
    "von_neumann": von_neumann(1),  # 5 cells
    "moore": moore(1),  # 9 cells
    "von_neumann_r2": von_neumann(2),  # 13 cells
    "moore_r2": moore(2),  # 25 cells
    "ring_r2": ring(2),  # 17 cells, hollow
    "cross_r3": cross(3),  # 13 cells, axes only
}
