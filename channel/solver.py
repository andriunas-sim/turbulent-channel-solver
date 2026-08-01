"""Fully developed turbulent channel flow, wall units, one wall-normal
direction only.

Momentum reduces to an exact total-shear balance,

    (1 + nu_t+) dU+/dy+ = tau+(y+) = 1 - y+/Re_tau,

with U+(0) = 0 and dU+/dy+ = 0 on the centreline.

Two solvers are provided deliberately:

* `solve_quadrature` inverts the closure analytically and integrates. For
  this equation it is essentially exact, so it serves as the verification
  reference.
* `solve_fvm` discretises the second-order form on a finite-volume grid and
  Picard-iterates the eddy viscosity. It is slower and less accurate, and it
  is the one that extends to the energy equation and to two-equation models.

Agreement between the two is code verification: am I solving the equations
correctly. Agreement with DNS is validation: am I solving the correct
equations. They are separate questions and the package keeps them separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import fixed_quad
from scipy.linalg import solve_banded

from .grid import Grid
from .turbulence import MixingLength

__all__ = ["Solution", "solve_quadrature", "solve_fvm", "total_shear"]


def total_shear(y_plus: np.ndarray, Re_tau: float) -> np.ndarray:
    """Exact total shear stress profile for fully developed channel flow."""
    return 1.0 - np.asarray(y_plus, dtype=float) / Re_tau


def _segment_integrate(f, a: float, b: float, n_seg: int = 24, order: int = 40) -> float:
    """Composite Gauss-Legendre quadrature on a geometrically graded partition.

    The integrand is analytic but has a thin near-wall layer, so a uniform
    partition wastes points. Grading the partition toward `a` and using a high
    Gauss order per segment converges to round-off without adaptive machinery,
    which keeps the reference solution deterministic.
    """
    if b <= a:
        return 0.0
    t = np.linspace(0.0, 1.0, n_seg + 1) ** 2.0
    edges = a + (b - a) * t
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            total += fixed_quad(f, lo, hi, n=order)[0]
    return float(total)


@dataclass
class Solution:
    """Velocity field and derived integral quantities, all in wall units."""

    y_plus: np.ndarray
    U_plus: np.ndarray
    Re_tau: float
    U_bulk_plus: float
    nu_t_plus: np.ndarray | None = None
    dUdy_plus: np.ndarray | None = None
    y_faces_plus: np.ndarray | None = None  # where nu_t and dUdy live, if not co-located
    iterations: int = 0
    residual: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def Re_bulk(self) -> float:
        """Bulk Reynolds number on full channel height 2*delta."""
        return 2.0 * self.U_bulk_plus * self.Re_tau

    @property
    def Re_centreline(self) -> float:
        return self.U_plus[-1] * self.Re_tau

    @property
    def c_f(self) -> float:
        """Skin friction coefficient, 2 tau_w / (rho U_b^2)."""
        return 2.0 / self.U_bulk_plus**2

    def summary(self) -> str:
        return (
            f"Re_tau={self.Re_tau:g}  U_b+={self.U_bulk_plus:.4f}  "
            f"Re_b={self.Re_bulk:.1f}  Cf={self.c_f:.5e}"
        )


def solve_quadrature(
    Re_tau: float,
    y_out: np.ndarray | None = None,
    model: MixingLength | None = None,
    n_out: int = 512,
    constant_stress: bool = False,
) -> Solution:
    """Integrate the exact gradient. Reference solution.

    The bulk velocity is obtained by parts,

        int_0^L U dy = L U(L) - int_0^L y (dU/dy) dy,

    so that only the gradient is ever integrated numerically.

    Setting `constant_stress` replaces the channel's linear shear decay with
    tau+ = 1 everywhere. That is not a channel any more, but it isolates the
    closure from the flow: with the shear held constant the model should
    return exactly the log-law constants it was given, so any deviation is
    the closure's and not the geometry's.
    """
    model = model or MixingLength()

    def shear(y):
        y = np.asarray(y, dtype=float)
        return np.ones_like(y) if constant_stress else total_shear(y, Re_tau)

    def gradient(y):
        y = np.atleast_1d(np.asarray(y, dtype=float))
        return model.shear_to_gradient(y, shear(y))

    if y_out is None:
        # log-spaced from the wall, so the sublayer is resolved in the output
        y_out = np.concatenate(
            ([0.0], np.logspace(-3, np.log10(Re_tau), n_out - 1))
        )
    y_out = np.asarray(y_out, dtype=float)

    U = np.zeros_like(y_out)
    for i in range(1, y_out.size):
        U[i] = U[i - 1] + _segment_integrate(gradient, y_out[i - 1], y_out[i], n_seg=4)

    moment = _segment_integrate(lambda y: np.asarray(y) * gradient(y), 0.0, Re_tau, n_seg=64)
    if np.isclose(y_out[-1], Re_tau):
        U_centre = float(U[-1])
    else:
        U_centre = _segment_integrate(gradient, 0.0, Re_tau, n_seg=64)
    U_bulk = U_centre - moment / Re_tau

    dUdy = model.shear_to_gradient(y_out, shear(y_out))
    nu_t = model.nu_t(y_out, dUdy)
    return Solution(
        y_plus=y_out,
        U_plus=U,
        Re_tau=Re_tau,
        U_bulk_plus=float(U_bulk),
        nu_t_plus=nu_t,
        dUdy_plus=dUdy,
        meta={"solver": "quadrature", "constant_stress": constant_stress},
    )


def solve_fvm(
    grid: Grid,
    model: MixingLength | None = None,
    relaxation: float = 0.7,
    tol: float = 1e-12,
    max_iter: int = 5000,
) -> Solution:
    """Finite-volume solution with Picard iteration on the eddy viscosity.

    Diffusion is discretised conservatively at faces. The wall face uses the
    molecular viscosity only, since nu_t+ vanishes there under van Driest
    damping, and the centreline face carries zero flux by symmetry.
    """
    model = model or MixingLength()
    grid.check()

    yc = grid.centres
    yf = grid.faces
    dy = grid.widths
    n = grid.n_cells
    Re_tau = grid.Re_tau

    # distance between adjacent cell centres, with the wall face using the
    # half-cell distance to the wall itself
    h = np.empty(n + 1)
    h[0] = yc[0]
    h[1:n] = np.diff(yc)
    h[n] = np.inf  # symmetry face carries no flux

    U = np.zeros(n)
    source = -dy / Re_tau
    ab = np.zeros((3, n))
    residual = np.inf
    it = 0

    for it in range(1, max_iter + 1):
        # face gradients from the current field
        grad = np.zeros(n + 1)
        grad[0] = U[0] / h[0]
        grad[1:n] = np.diff(U) / h[1:n]
        grad[n] = 0.0

        gamma = 1.0 + model.nu_t(yf, grad)
        gamma[0] = 1.0   # nu_t = 0 at the wall
        gamma[n] = 1.0   # irrelevant, zero-flux face

        coef = np.zeros(n + 1)
        coef[:n] = gamma[:n] / h[:n]
        coef[n] = 0.0

        ab[0, 1:] = coef[1:n]                    # upper diagonal
        ab[1, :] = -(coef[:n] + coef[1 : n + 1])  # main diagonal
        ab[2, :-1] = coef[1:n]                   # lower diagonal

        U_new = solve_banded((1, 1), ab, source)
        residual = float(np.max(np.abs(U_new - U)) / max(np.max(np.abs(U_new)), 1e-30))
        U = (1.0 - relaxation) * U + relaxation * U_new
        if residual < tol:
            break

    U_bulk = float(np.sum(U * dy) / Re_tau)
    grad = np.zeros(n + 1)
    grad[0] = U[0] / h[0]
    grad[1:n] = np.diff(U) / h[1:n]
    nu_t = model.nu_t(yf, grad)
    nu_t[0] = 0.0

    return Solution(
        y_plus=yc,
        U_plus=U,
        Re_tau=Re_tau,
        U_bulk_plus=U_bulk,
        nu_t_plus=nu_t,
        dUdy_plus=grad,
        y_faces_plus=yf,
        iterations=it,
        residual=residual,
        meta={"solver": "fvm", "n_cells": n, "y1_plus": grid.y1_plus},
    )
