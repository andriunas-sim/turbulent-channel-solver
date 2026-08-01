"""Analytical and empirical references, so the solver can be validated with
no data files present.

Spalding's composite profile is a single implicit expression covering the
viscous sublayer, buffer layer and log layer. It is derived for a constant
stress layer, so it is a fair comparison only where tau+ is close to unity,
roughly y+ < 0.3 Re_tau. Above that the channel's linear shear decay makes
the two diverge for physical reasons rather than numerical ones, and any
comparison there is meaningless.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .turbulence import KARMAN

__all__ = [
    "spalding_y_of_u",
    "spalding_u_of_y",
    "log_law",
    "fit_log_law",
    "dean_cf",
    "dean_Re_tau",
]

B_LOG = 5.0


def spalding_y_of_u(U_plus: np.ndarray, kappa: float = KARMAN, B: float = B_LOG) -> np.ndarray:
    """Spalding's law, explicit in y+ as a function of U+."""
    U = np.asarray(U_plus, dtype=float)
    kU = kappa * U
    return U + np.exp(-kappa * B) * (
        np.exp(kU) - 1.0 - kU - kU**2 / 2.0 - kU**3 / 6.0
    )


def spalding_u_of_y(y_plus: np.ndarray, kappa: float = KARMAN, B: float = B_LOG) -> np.ndarray:
    """Invert Spalding's law numerically to give U+(y+)."""
    y = np.atleast_1d(np.asarray(y_plus, dtype=float))
    out = np.zeros_like(y)
    for i, yi in enumerate(y):
        if yi <= 0.0:
            out[i] = 0.0
            continue
        # start from the log law, which bounds U+ from below for y+ > 1
        hi = max(np.log(max(yi, 1.001)) / kappa + B + 5.0, 5.0)
        with np.errstate(over="ignore"):
            while spalding_y_of_u(np.array([hi]), kappa, B)[0] < yi:
                hi += 5.0
        out[i] = brentq(
            lambda u: spalding_y_of_u(np.array([u]), kappa, B)[0] - yi,
            0.0,
            hi,
            xtol=1e-14,
            rtol=1e-15,
        )
    return out if np.ndim(y_plus) else float(out[0])


def log_law(y_plus: np.ndarray, kappa: float = KARMAN, B: float = B_LOG) -> np.ndarray:
    return np.log(np.asarray(y_plus, dtype=float)) / kappa + B


def fit_log_law(
    y_plus: np.ndarray,
    U_plus: np.ndarray,
    Re_tau: float,
    lower: float = 50.0,
    upper_frac: float = 0.15,
) -> tuple[float, float, int]:
    """Least-squares fit of U+ = ln(y+)/kappa + B over the log layer.

    Returns (kappa, B, n_points). The window is the conventional one: above
    the buffer layer and below the point where the outer layer takes over.
    Recovering kappa near 0.41 and B near 5 is the first thing to check,
    because those constants are inputs to the model and should come back out.
    """
    mask = (y_plus >= lower) & (y_plus <= upper_frac * Re_tau)
    if mask.sum() < 3:
        raise ValueError("log-layer window contains too few points; raise Re_tau or n")
    slope, intercept = np.polyfit(np.log(y_plus[mask]), U_plus[mask], 1)
    return 1.0 / slope, intercept, int(mask.sum())


def dean_cf(Re_bulk: float) -> float:
    """Dean (1978) skin friction correlation for plane channel flow.

    Cf = 0.073 Re_b^(-1/4), with Re_b on the full channel height and bulk
    velocity. Quoted accuracy is a few per cent, so treat a 5 per cent
    agreement as a pass and anything under 2 per cent as suspiciously good.
    """
    return 0.073 * Re_bulk ** (-0.25)


def dean_Re_tau(Re_bulk: float) -> float:
    """Dean's companion relation, Re_tau = 0.09 Re_b^0.88."""
    return 0.09 * Re_bulk**0.88
