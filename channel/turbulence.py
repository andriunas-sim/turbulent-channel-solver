"""Algebraic eddy-viscosity closure: Prandtl mixing length with van Driest
damping, in wall units.

    l+  = kappa * y+ * (1 - exp(-y+ / A+))
    nu_t+ = l+^2 * |dU+/dy+|

The damping factor is what makes this work at all near the wall: without
it the eddy viscosity stays finite as y+ -> 0 and the model never recovers
the viscous sublayer. Note the model is defined against distance from the
nearest wall, so on a half-channel it is applied on y+ directly and is
strictly only valid up to the outer layer, where a Klebanoff-type outer
cut-off would normally take over. Tier 1 deliberately omits that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MixingLength", "KARMAN", "A_PLUS"]

KARMAN = 0.41
A_PLUS = 26.0


@dataclass(frozen=True)
class MixingLength:
    """Damped mixing length, optionally capped in the outer layer.

    `lambda_outer` applies the Escudier/Nikuradse limit l <= lambda * delta,
    with lambda around 0.09 for channel flow. It is off by default so that the
    bare model's outer-layer deficiency is visible in the validation report
    rather than quietly corrected. Turning it on is the first refinement worth
    making, and the report quantifies what it buys.
    """

    kappa: float = KARMAN
    a_plus: float = A_PLUS
    lambda_outer: float | None = None
    Re_tau: float | None = None

    def __post_init__(self):
        if self.lambda_outer is not None and self.Re_tau is None:
            raise ValueError("lambda_outer requires Re_tau to set the cap in wall units")

    def length(self, y_plus: np.ndarray) -> np.ndarray:
        """Damped mixing length l+ at the given wall distances."""
        y_plus = np.asarray(y_plus, dtype=float)
        ell = self.kappa * y_plus * (1.0 - np.exp(-y_plus / self.a_plus))
        if self.lambda_outer is not None:
            ell = np.minimum(ell, self.lambda_outer * self.Re_tau)
        return ell

    def nu_t(self, y_plus: np.ndarray, dUdy: np.ndarray) -> np.ndarray:
        """Eddy viscosity nu_t+ from wall distance and velocity gradient."""
        ell = self.length(y_plus)
        return ell**2 * np.abs(np.asarray(dUdy, dtype=float))

    def shear_to_gradient(
        self, y_plus: np.ndarray, tau_plus: np.ndarray
    ) -> np.ndarray:
        """Invert (1 + nu_t+) dU+/dy+ = tau+ for dU+/dy+.

        Substituting the closure gives a quadratic in S = dU+/dy+,

            l+^2 S^2 + S - tau+ = 0.

        The textbook root (-1 + sqrt(1 + 4 l+^2 tau+)) / (2 l+^2) is written
        here in its rationalised form

            S = 2 tau+ / (1 + sqrt(1 + 4 l+^2 tau+)),

        which is algebraically identical but numerically stable. The textbook
        form subtracts two nearly equal numbers as l+ -> 0 and loses most of
        its significant figures through the viscous sublayer, which is exactly
        where the answer has to be right. The rationalised form also takes the
        wall limit S -> tau+ automatically, with no special case.
        """
        y_plus = np.asarray(y_plus, dtype=float)
        tau_plus = np.asarray(tau_plus, dtype=float)
        ell2 = self.length(y_plus) ** 2
        return 2.0 * tau_plus / (1.0 + np.sqrt(1.0 + 4.0 * ell2 * tau_plus))
