"""Tests for the Tier 1 channel package.

Run with `pytest -q` from the repository root. The suite is split into
verification (does the code solve the equations it claims) and validation
(do the answers match physics), because those failures need different fixes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channel import (
    MixingLength,
    dean_cf,
    fit_log_law,
    grid_for_y1plus,
    load_means,
    log_law,
    solve_fvm,
    solve_quadrature,
    spalding_u_of_y,
    spalding_y_of_u,
    tanh_grid,
    total_shear,
    uniform_grid,
)

RE_TAU = 395.0


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

def test_uniform_grid_spans_domain():
    g = uniform_grid(RE_TAU, 100)
    assert g.n_cells == 100
    assert g.faces[0] == 0.0
    assert np.isclose(g.faces[-1], RE_TAU)
    assert np.allclose(g.widths, RE_TAU / 100)


def test_tanh_grid_clusters_at_wall():
    g = tanh_grid(RE_TAU, 100, gamma=2.0)
    assert g.widths[0] < g.widths[-1]
    assert np.all(np.diff(g.faces) > 0.0)


def test_grid_hits_target_y1_plus():
    for target in (0.05, 0.1, 0.5, 0.9):
        g = grid_for_y1plus(RE_TAU, 200, target)
        assert np.isclose(g.y1_plus, target, rtol=1e-9)


def test_grid_rejects_target_coarser_than_uniform():
    """The stretch family only refines toward the wall.

    Asking for a y1+ coarser than the uniform grid would need clustering away
    from the wall, which this family cannot express. Failing loudly is better
    than silently returning a near-uniform grid and a misleading y1+ label on
    the near-wall study.
    """
    try:
        grid_for_y1plus(RE_TAU, 200, 5.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a target coarser than uniform")


# --------------------------------------------------------------------------
# closure
# --------------------------------------------------------------------------

def test_mixing_length_vanishes_at_wall():
    m = MixingLength()
    assert m.length(np.array([0.0]))[0] == 0.0


def test_mixing_length_recovers_kappa_y_far_from_wall():
    m = MixingLength()
    y = np.array([500.0])
    assert np.isclose(m.length(y)[0], m.kappa * y[0], rtol=1e-6)


def test_outer_cap_limits_length():
    m = MixingLength(lambda_outer=0.09, Re_tau=RE_TAU)
    assert np.isclose(m.length(np.array([RE_TAU]))[0], 0.09 * RE_TAU)


def test_gradient_inversion_satisfies_shear_balance():
    m = MixingLength()
    y = np.logspace(-3, np.log10(RE_TAU), 400)
    tau = total_shear(y, RE_TAU)
    s = m.shear_to_gradient(y, tau)
    assert np.allclose((1.0 + m.nu_t(y, s)) * s, tau, rtol=1e-10, atol=1e-12)


def test_gradient_inversion_wall_limit_is_viscous():
    m = MixingLength()
    s = m.shear_to_gradient(np.array([0.0]), np.array([1.0]))
    assert np.isclose(s[0], 1.0)


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def test_viscous_sublayer_is_linear():
    """U+ = y+ holds to within a few tenths of a per cent only for y+ <~ 1.

    By y+ = 3 the damped eddy viscosity is already order 1e-2 and the shear
    has decayed by y+/Re_tau, so a tight tolerance there would be wrong
    rather than strict.
    """
    sol = solve_quadrature(RE_TAU, y_out=np.linspace(0.0, 1.0, 40))
    assert np.allclose(sol.U_plus, sol.y_plus, rtol=3e-3, atol=1e-9)


def test_fvm_matches_quadrature_reference():
    ref = solve_quadrature(RE_TAU)
    fvm = solve_fvm(grid_for_y1plus(RE_TAU, 256, 0.5))
    assert abs(fvm.U_bulk_plus - ref.U_bulk_plus) / ref.U_bulk_plus < 1e-3


def test_fvm_is_second_order_in_bulk_velocity():
    ref = solve_quadrature(RE_TAU)
    errs = []
    for n in (128, 256, 512):
        g = grid_for_y1plus(RE_TAU, n, 128.0 / n)
        errs.append(abs(solve_fvm(g).U_bulk_plus - ref.U_bulk_plus))
    order = np.polyfit(np.log([128, 256, 512]), np.log(errs), 1)[0]
    assert -order > 1.8, f"observed order {-order:.2f} is below second"


def test_fvm_reproduces_exact_shear_profile():
    g = grid_for_y1plus(RE_TAU, 400, 0.3)
    sol = solve_fvm(g)
    m = MixingLength()
    tau = (1.0 + m.nu_t(g.faces, sol.dUdy_plus)) * sol.dUdy_plus
    exact = total_shear(g.faces, RE_TAU)
    # the centreline face carries zero flux by construction, so drop it
    assert np.max(np.abs(tau[:-1] - exact[:-1])) < 5e-3


def test_solver_converges():
    sol = solve_fvm(grid_for_y1plus(RE_TAU, 200, 0.5))
    assert sol.residual < 1e-11
    assert sol.iterations < 500


def test_derived_quantities_are_consistent():
    sol = solve_quadrature(RE_TAU)
    assert np.isclose(sol.c_f, 2.0 / sol.U_bulk_plus**2)
    assert np.isclose(sol.Re_bulk, 2.0 * sol.U_bulk_plus * RE_TAU)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def test_log_law_constants_are_recovered_under_constant_stress():
    """With the shear held constant the closure must return its own inputs.

    The channel's linear shear decay biases the fitted constants low at
    moderate Re_tau, so this isolates the closure from the geometry.
    """
    sol = solve_quadrature(5200.0, constant_stress=True)
    kappa, B, _ = fit_log_law(sol.y_plus, sol.U_plus, 5200.0, lower=100.0, upper_frac=0.5)
    assert abs(kappa - 0.41) < 0.02
    assert abs(B - 5.2) < 0.4


def test_profile_is_close_to_spalding_in_the_inner_layer():
    sol = solve_quadrature(RE_TAU)
    mask = (sol.y_plus >= 1.0) & (sol.y_plus <= 0.2 * RE_TAU)
    err = sol.U_plus[mask] - spalding_u_of_y(sol.y_plus[mask])
    assert np.sqrt(np.mean(err**2)) < 0.5


def test_skin_friction_within_dean_scatter_with_outer_cap():
    for Re_tau in (395.0, 590.0, 1000.0, 2000.0):
        m = MixingLength(lambda_outer=0.09, Re_tau=Re_tau)
        sol = solve_quadrature(Re_tau, model=m)
        rel = abs(sol.c_f - dean_cf(sol.Re_bulk)) / dean_cf(sol.Re_bulk)
        assert rel < 0.05, f"Re_tau={Re_tau}: {100 * rel:.1f} per cent from Dean"


def test_bare_model_overpredicts_friction():
    """Documents a known deficiency rather than hiding it.

    Without an outer-layer limit the mixing length grows unchecked to the
    centreline, the profile flattens, and the friction comes out high. If this
    test ever fails, the closure has been changed and the report needs redoing.
    """
    sol = solve_quadrature(1000.0)
    assert sol.c_f > dean_cf(sol.Re_bulk)


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------

def test_spalding_round_trip():
    u = np.linspace(0.1, 25.0, 60)
    assert np.allclose(spalding_u_of_y(spalding_y_of_u(u)), u, rtol=1e-9)


def test_log_law_fit_recovers_synthetic_constants():
    y = np.logspace(1.5, 2.5, 200)
    kappa, B, _ = fit_log_law(y, log_law(y, 0.39, 5.5), 2000.0, lower=50.0, upper_frac=0.2)
    assert np.isclose(kappa, 0.39, rtol=1e-8)
    assert np.isclose(B, 5.5, atol=1e-8)


def test_dean_correlation_value():
    assert np.isclose(dean_cf(10000.0), 0.073 * 10000.0**-0.25)


# --------------------------------------------------------------------------
# DNS reader
# --------------------------------------------------------------------------

def _write(tmp: Path, text: str) -> Path:
    tmp.write_text(text)
    return tmp


def test_dns_reader_parses_commented_file(tmp_path):
    f = _write(
        tmp_path / "chan.means",
        "% Re_tau = 197.5\n% y y+ Umean dUmean/dy\n"
        "0.0 0.0 0.0 1.0\n0.001 0.4 0.40 0.99\n0.01 4.0 3.90 0.80\n1.0 197.5 18.0 0.01\n",
    )
    prof = load_means(f)
    assert prof.y_plus.size == 4
    assert np.isclose(prof.Re_tau, 197.5)
    assert np.isclose(prof.interpolate(np.array([4.0]))[0], 3.90)


def test_dns_reader_rejects_wrong_column(tmp_path):
    f = _write(tmp_path / "bad.means", "% h\n0.0 0.0 5.0\n0.1 4.0 3.0\n0.2 8.0 9.0\n")
    # no Re_tau in the header, so only the monotonicity check can fire
    try:
        load_means(f, y_col=2, u_col=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-monotonic y+ column")


# --------------------------------------------------------------------------
# DNS header parsing
# --------------------------------------------------------------------------

def test_header_Re_tau_ignores_the_citation_line(tmp_path):
    """The paper title contains the nominal label, the declaration the real one.

    A substring search returns 590 from the reference line before reaching the
    declaration below it. The match must be anchored to the start of the
    comment body.
    """
    from channel import header_Re_tau

    f = _write(
        tmp_path / "chan590.means",
        "# Reference: DNS of Turbulent Channel Flow up to Re_tau=590, 1999,\n"
        "# Re_tau = 587.19\n"
        "# Normalization: U_tau, h\n"
        "0.0 0.0 0.0 1.0\n1.0 587.19 21.26 0.0\n",
    )
    assert np.isclose(header_Re_tau(f), 587.19)


def test_load_means_rejects_header_data_mismatch(tmp_path):
    """Centreline y+ must equal the stated Re_tau, or a column is misread."""
    f = _write(
        tmp_path / "chan590.means",
        "# Re_tau = 587.19\n0.0 0.0 0.0\n0.5 100.0 15.0\n1.0 200.0 18.0\n",
    )
    try:
        load_means(f)
    except ValueError as exc:
        assert "587.19" in str(exc)
        return
    raise AssertionError("expected ValueError for a header/data mismatch")


def test_header_Re_tau_returns_none_when_absent(tmp_path):
    from channel import header_Re_tau

    f = _write(tmp_path / "plain.means", "# no declaration here\n0.0 0.0 0.0\n1.0 10.0 9.0\n")
    assert header_Re_tau(f) is None
