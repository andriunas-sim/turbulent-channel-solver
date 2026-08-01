"""Figures and diagnostics.

Every plot here is chosen to answer one specific question that gets asked in
an interview or a review, not to decorate a report.
"""

from __future__ import annotations

import numpy as np

from .reference import log_law, spalding_u_of_y
from .solver import Solution, total_shear

__all__ = [
    "plot_velocity_profile",
    "plot_shear_balance",
    "plot_eddy_viscosity",
    "plot_grid_convergence",
    "profile_error",
]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def profile_error(sol: Solution, reference, y_min: float = 1.0, y_max_frac: float = 1.0):
    """RMS and maximum deviation in U+ over a wall-distance window.

    `reference` is any callable mapping y+ to U+. The window matters: comparing
    at y+ < 1 flatters the model because everything collapses onto U+ = y+
    there, and comparing beyond the log layer against a constant-stress
    reference measures the reference's invalidity rather than the model's.
    """
    y = sol.y_plus
    mask = (y >= y_min) & (y <= y_max_frac * sol.Re_tau)
    ref = np.asarray(reference(y[mask]), dtype=float)
    err = sol.U_plus[mask] - ref
    finite = np.isfinite(err)
    err = err[finite]
    return {
        "rms": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))),
        "n_points": int(err.size),
    }


def plot_velocity_profile(solutions: dict[str, Solution], Re_tau: float, dns=None, path=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    yl = np.logspace(-1, np.log10(Re_tau), 400)
    ax.plot(yl, yl, "k:", lw=1.0, label=r"$U^+=y^+$")
    ylog = yl[yl >= 20]
    ax.plot(ylog, log_law(ylog), "k--", lw=1.0, label=r"$U^+=\ln y^+/\kappa + B$")
    ax.plot(yl, spalding_u_of_y(yl), color="0.55", lw=1.2, label="Spalding")

    if dns is not None:
        ax.plot(dns.y_plus, dns.U_plus, "o", ms=3.0, mfc="none", color="crimson",
                label=f"DNS, {dns.source}")

    for name, sol in solutions.items():
        ax.plot(sol.y_plus, sol.U_plus, lw=1.6, label=name)

    ax.set_xscale("log")
    ax.set_xlim(0.1, Re_tau)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_title(rf"Mean velocity, $Re_\tau={Re_tau:g}$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig


def plot_shear_balance(sol: Solution, path=None):
    """Viscous and turbulent shear against the exact linear total.

    This is the sharpest verification plot in the package. The total shear is
    known analytically for this flow, so if the two computed components do not
    sum to the straight line, the discretisation is wrong. No amount of
    plausible-looking velocity profile substitutes for it.
    """
    plt = _mpl()
    grad = sol.dUdy_plus
    nu_t = sol.nu_t_plus
    if grad is None or nu_t is None:
        raise ValueError("solution carries no gradient or eddy viscosity")
    # gradients and eddy viscosity live on faces for the FVM solution and on
    # the output points for the quadrature one; pair them correctly rather
    # than truncating, which would shift everything by half a cell
    y = sol.y_faces_plus if sol.y_faces_plus is not None else sol.y_plus
    if not (y.size == grad.size == nu_t.size):
        raise ValueError("shear balance arrays are not co-located")

    visc = grad
    turb = nu_t * grad
    exact = total_shear(y, sol.Re_tau)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(y / sol.Re_tau, visc, lw=1.4, label=r"viscous $dU^+/dy^+$")
    ax.plot(y / sol.Re_tau, turb, lw=1.4, label=r"turbulent $\nu_t^+ dU^+/dy^+$")
    ax.plot(y / sol.Re_tau, visc + turb, lw=1.6, label="sum")
    ax.plot(y / sol.Re_tau, exact, "k--", lw=1.0, label=r"exact $1-y/\delta$")
    ax.set_ylabel(r"$\tau^+$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("Shear stress balance")

    axr.plot(y / sol.Re_tau, visc + turb - exact, lw=1.2, color="crimson")
    axr.set_xlabel(r"$y/\delta$")
    axr.set_ylabel("residual")
    axr.grid(alpha=0.25)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig


def plot_eddy_viscosity(sol: Solution, path=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    y = sol.y_faces_plus if sol.y_faces_plus is not None else sol.y_plus
    ax.plot(y, sol.nu_t_plus, lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$\nu_t^+$")
    ax.set_title("Eddy viscosity, van Driest damped mixing length")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig


def plot_grid_convergence(n_cells, errors, path=None):
    plt = _mpl()
    n_cells = np.asarray(n_cells, dtype=float)
    errors = np.asarray(errors, dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.loglog(n_cells, errors, "o-", lw=1.5, label=r"$|U_b^+ - U_{b,\mathrm{ref}}^+|$")
    ref = errors[0] * (n_cells / n_cells[0]) ** -2.0
    ax.loglog(n_cells, ref, "k--", lw=1.0, label="second order")
    ax.set_xlabel("cells")
    ax.set_ylabel("bulk velocity error")
    ax.set_title("Grid convergence against the quadrature reference")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig
