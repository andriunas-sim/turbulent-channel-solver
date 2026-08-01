"""Tier 1: fully developed turbulent channel flow with an algebraic
mixing-length closure, validated against DNS.

Wall units throughout. The only dimensional inputs are Re_tau and, at Tier 2,
the Prandtl numbers.
"""

from .grid import Grid, grid_for_y1plus, tanh_grid, uniform_grid
from .turbulence import A_PLUS, KARMAN, MixingLength
from .solver import Solution, solve_fvm, solve_quadrature, total_shear
from .reference import (
    dean_Re_tau,
    dean_cf,
    fit_log_law,
    log_law,
    spalding_u_of_y,
    spalding_y_of_u,
)
from .dns import DNS_ROOT, DnsProfile, dns_path, header_Re_tau, load_case, load_means

__version__ = "0.1.0"

__all__ = [
    "Grid",
    "uniform_grid",
    "tanh_grid",
    "grid_for_y1plus",
    "MixingLength",
    "KARMAN",
    "A_PLUS",
    "Solution",
    "solve_quadrature",
    "solve_fvm",
    "total_shear",
    "spalding_y_of_u",
    "spalding_u_of_y",
    "log_law",
    "fit_log_law",
    "dean_cf",
    "dean_Re_tau",
    "DnsProfile",
    "load_means",
    "load_case",
    "dns_path",
    "header_Re_tau",
    "DNS_ROOT",
]
