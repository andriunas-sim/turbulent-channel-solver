"""Wall-normal grids for a half-channel, in wall units.

The domain runs from the wall at y+ = 0 to the channel centreline at
y+ = Re_tau. Cells are finite volumes: `faces` has N+1 entries, `centres`
has N. Clustering toward the wall uses a one-sided hyperbolic tangent
stretch, and the stretching parameter can be solved for a target first
cell-centre distance y1+, which is the control the near-wall study needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

__all__ = ["Grid", "uniform_grid", "tanh_grid", "grid_for_y1plus"]


@dataclass(frozen=True)
class Grid:
    """Cell-centred finite-volume grid on [0, Re_tau]."""

    faces: np.ndarray
    Re_tau: float

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.faces[:-1] + self.faces[1:])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.faces)

    @property
    def n_cells(self) -> int:
        return self.faces.size - 1

    @property
    def y1_plus(self) -> float:
        """First cell-centre distance from the wall, in wall units."""
        return float(self.centres[0])

    def check(self) -> None:
        if self.faces[0] != 0.0:
            raise ValueError("first face must sit on the wall")
        if not np.isclose(self.faces[-1], self.Re_tau):
            raise ValueError("last face must sit on the centreline")
        if np.any(np.diff(self.faces) <= 0.0):
            raise ValueError("faces must be strictly increasing")


def uniform_grid(Re_tau: float, n_cells: int) -> Grid:
    g = Grid(np.linspace(0.0, Re_tau, n_cells + 1), Re_tau)
    g.check()
    return g


def tanh_grid(Re_tau: float, n_cells: int, gamma: float) -> Grid:
    """One-sided tanh clustering toward the wall.

    gamma -> 0 recovers a uniform grid; larger gamma clusters harder.
    """
    if gamma <= 0.0:
        return uniform_grid(Re_tau, n_cells)
    eta = np.linspace(0.0, 1.0, n_cells + 1)
    faces = Re_tau * (1.0 - np.tanh(gamma * (1.0 - eta)) / np.tanh(gamma))
    faces[0] = 0.0
    faces[-1] = Re_tau
    g = Grid(faces, Re_tau)
    g.check()
    return g


def grid_for_y1plus(Re_tau: float, n_cells: int, y1_plus: float) -> Grid:
    """Grid whose first cell centre sits at the requested y1+.

    Solves for the tanh stretching parameter by bisection. Raises if the
    target is unreachable with this cell count, which is itself useful
    information: it tells you the mesh cannot be that fine at the wall
    without an absurd expansion ratio.
    """
    if y1_plus <= 0.0:
        raise ValueError("y1_plus must be positive")
    uniform_y1 = 0.5 * Re_tau / n_cells
    if y1_plus > uniform_y1:
        raise ValueError(
            f"y1+={y1_plus:g} is coarser than the uniform grid value "
            f"{uniform_y1:g}; reduce n_cells"
        )

    def residual(gamma: float) -> float:
        return tanh_grid(Re_tau, n_cells, gamma).y1_plus - y1_plus

    lo, hi = 1e-6, 1.0
    while residual(hi) > 0.0:
        hi *= 2.0
        if hi > 1e3:
            raise ValueError(f"cannot reach y1+={y1_plus:g} with {n_cells} cells")
    gamma = brentq(residual, lo, hi, xtol=1e-12, rtol=1e-14)
    return tanh_grid(Re_tau, n_cells, gamma)
