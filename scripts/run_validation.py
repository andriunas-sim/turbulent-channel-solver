"""Run verification and validation for the Tier 1 channel solver.

    python scripts/run_validation.py --Re-tau 395 --dns data/chan395.means

With no --dns argument the script falls back to Spalding's composite profile
and says so in the report. Figures are written to figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channel import (  # noqa: E402
    MixingLength,
    dns_path,
    header_Re_tau,
    dean_Re_tau,
    dean_cf,
    fit_log_law,
    grid_for_y1plus,
    load_means,
    solve_fvm,
    solve_quadrature,
    spalding_u_of_y,
)
from channel.postprocess import (  # noqa: E402
    plot_eddy_viscosity,
    plot_grid_convergence,
    plot_shear_balance,
    plot_velocity_profile,
    profile_error,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--Re-tau", type=float, default=None,
        help="friction Reynolds number; defaults to the value stated in the "
             "DNS header, or 395 if no DNS file is given",
    )
    p.add_argument(
        "--dns", type=str, default=None,
        help="either a case label (180, 395, 590) resolved inside "
             "data/chandata, or an explicit path to a .means file",
    )
    p.add_argument("--y-col", type=int, default=1)
    p.add_argument("--u-col", type=int, default=2)
    p.add_argument("--cells", type=int, default=256)
    p.add_argument("--y1-plus", type=float, default=0.5)
    p.add_argument("--lambda-outer", type=float, default=0.09,
                   help="Escudier outer-layer mixing length cap, l <= lambda*delta")
    p.add_argument("--figures", type=str, default="figures")
    args = p.parse_args()

    # Resolve the DNS file first, because it carries the authoritative Re_tau.
    # The filename label is nominal: chan590 was run at 587.19.
    dns_file = None
    if args.dns:
        dns_file = (
            dns_path(args.dns) if args.dns.isdigit() and len(args.dns) <= 4
            else Path(args.dns)
        )

    Re_tau = args.Re_tau
    if Re_tau is None:
        Re_tau = (header_Re_tau(dns_file) if dns_file else None) or 395.0
    figdir = Path(args.figures)
    figdir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"Tier 1 channel, mixing length with van Driest damping, Re_tau = {Re_tau:g}")
    print("=" * 68)

    ref = solve_quadrature(Re_tau)
    capped_model = MixingLength(lambda_outer=args.lambda_outer, Re_tau=Re_tau)
    capped = solve_quadrature(Re_tau, model=capped_model)
    grid = grid_for_y1plus(Re_tau, args.cells, args.y1_plus)
    fvm = solve_fvm(grid)

    print("\n-- solutions -------------------------------------------------")
    print(f"quadrature : {ref.summary()}")
    print(f"  + cap     : {capped.summary()}   (lambda = {args.lambda_outer:g})")
    print(f"fvm        : {fvm.summary()}")
    print(f"             {fvm.iterations} iterations, residual {fvm.residual:.2e}, "
          f"{grid.n_cells} cells, y1+ = {grid.y1_plus:.3f}")

    print("\n-- verification: two independent discretisations --------------")
    d_ub = abs(fvm.U_bulk_plus - ref.U_bulk_plus)
    print(f"bulk velocity difference : {d_ub:.3e}  "
          f"({100 * d_ub / ref.U_bulk_plus:.4f} per cent)")

    ns, errs = [], []
    for n in (64, 128, 256, 512, 1024):
        g = grid_for_y1plus(Re_tau, n, 0.5 * 256.0 / n * args.y1_plus)
        s = solve_fvm(g)
        e = abs(s.U_bulk_plus - ref.U_bulk_plus)
        ns.append(n)
        errs.append(e)
        print(f"  n = {n:5d}   y1+ = {g.y1_plus:7.4f}   U_b+ = {s.U_bulk_plus:.6f}"
              f"   error = {e:.3e}")
    order = np.polyfit(np.log(ns), np.log(np.maximum(errs, 1e-16)), 1)[0]
    print(f"observed order of accuracy : {-order:.2f}")

    print("\n-- validation: constants recovered from the profile -----------")
    kappa_fit, B_fit, npts = fit_log_law(ref.y_plus, ref.U_plus, Re_tau)
    print(f"channel shear   : kappa = {kappa_fit:.4f}, B = {B_fit:.3f}, {npts} points "
          f"in y+ = [50, {0.15 * Re_tau:.0f}]")
    cs = solve_quadrature(Re_tau, constant_stress=True)
    kappa_cs, B_cs, npts_cs = fit_log_law(cs.y_plus, cs.U_plus, Re_tau, upper_frac=0.5)
    print(f"constant stress : kappa = {kappa_cs:.4f}, B = {B_cs:.3f}, {npts_cs} points")
    print("model inputs      kappa = 0.41, A+ = 26 (B is an outcome, not an input)")
    if npts < 15:
        print("NOTE: the log-layer window is short at this Re_tau, so the fitted")
        print("      constants are biased by the buffer and outer layers.")

    print("\n-- validation: integral quantities ----------------------------")
    for label, sol in (("bare mixing length", ref), ("with outer cap    ", capped)):
        d = dean_cf(sol.Re_bulk)
        print(f"{label} : Re_b = {sol.Re_bulk:8.0f}  Cf = {sol.c_f:.4e}  "
              f"vs Dean {d:.4e}  ({100 * (sol.c_f - d) / d:+.2f} per cent)")
    ret_dean = dean_Re_tau(ref.Re_bulk)
    print(f"Re_tau {Re_tau:.0f} vs Dean's companion relation {ret_dean:.0f} "
          f"({100 * (Re_tau - ret_dean) / ret_dean:+.2f} per cent)")
    print("Dean's own scatter is a few per cent, so anything inside 5 is a pass.")

    dns = None
    if dns_file:
        dns = load_means(dns_file, y_col=args.y_col, u_col=args.u_col)
        print(f"\n-- validation: DNS, {dns.source} ------------------------------")
        print(f"{'closure':<20}{'rms, y+>=1':>12}{'max':>9}{'log rms':>10}"
              f"{'centre U+':>12}")
        for label, sol in (("bare", ref), (f"cap {args.lambda_outer:g}", capped)):
            a = profile_error(sol, dns.interpolate, y_min=1.0, y_max_frac=1.0)
            lg = profile_error(sol, dns.interpolate, y_min=30.0, y_max_frac=0.3)
            print(f"{label:<20}{a['rms']:>12.4f}{a['max_abs']:>9.4f}"
                  f"{lg['rms']:>10.4f}{sol.U_plus[-1]:>12.3f}")
        print(f"{'DNS':<20}{'':>12}{'':>9}{'':>10}{dns.U_plus[-1]:>12.3f}")

        print("\nerror by wall-distance band, mean U+ deviation from DNS")
        bands = [(1.0, 5.0), (5.0, 30.0), (30.0, 0.2 * Re_tau),
                 (0.2 * Re_tau, 0.6 * Re_tau), (0.6 * Re_tau, Re_tau)]
        print(f"{'band':<22}{'bare':>10}{'capped':>10}")
        for lo, hi in bands:
            row = []
            for sol in (ref, capped):
                m = (sol.y_plus >= lo) & (sol.y_plus <= hi)
                e = sol.U_plus[m] - dns.interpolate(sol.y_plus[m])
                e = e[np.isfinite(e)]
                row.append(float(np.mean(e)) if e.size else float("nan"))
            print(f"y+ [{lo:6.1f}, {hi:6.1f}]  {row[0]:>+10.3f}{row[1]:>+10.3f}")
        print("The two closures are identical below the outer layer, so any")
        print("difference there localises the deficiency to the mixing length")
        print("growing unchecked toward the centreline.")
    else:
        print("\n-- validation: Spalding fallback ------------------------------")
        print("no DNS file supplied, comparing against the composite profile")
        err = profile_error(ref, spalding_u_of_y, y_min=1.0, y_max_frac=0.3)
        print(f"y+ in [1, 0.3 Re_tau] : rms {err['rms']:.4f}, "
              f"max {err['max_abs']:.4f} wall units")

    plot_velocity_profile({"quadrature, bare": ref,
                           f"quadrature, cap {args.lambda_outer:g}": capped,
                           "fvm": fvm}, Re_tau, dns=dns,
                          path=figdir / "velocity_profile.png")
    plot_shear_balance(fvm, path=figdir / "shear_balance.png")
    plot_eddy_viscosity(ref, path=figdir / "eddy_viscosity.png")
    plot_grid_convergence(ns, errs, path=figdir / "grid_convergence.png")
    print(f"\nfigures written to {figdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
