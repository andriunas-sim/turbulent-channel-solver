# Tier 1: fully developed turbulent channel flow

A wall-resolved solver for fully developed turbulent flow between parallel
plates, closed with a van Driest damped mixing length, verified against an
independent discretisation and validated against DNS.

The physical case is a cooling channel between prismatic cells, or a passage in
a cold plate. It is the smallest problem that contains the thing worth
computing: the wall heat transfer coefficient, produced from geometry and flow
conditions rather than assumed or fitted.

## Physics

Fully developed flow makes the momentum equation an exact statement about
total shear:

    (1 + nu_t+) dU+/dy+ = tau+(y+) = 1 - y+/Re_tau

with `U+(0) = 0` and `dU+/dy+ = 0` on the centreline. The closure is

    l+    = kappa * y+ * (1 - exp(-y+/A+)),   kappa = 0.41, A+ = 26
    nu_t+ = l+^2 * |dU+/dy+|

with an optional Escudier outer-layer limit `l+ <= lambda * Re_tau`.

Because the shear distribution is known analytically, the closure can be
inverted for the gradient in closed form. That gives a reference solution
requiring no iteration at all, which is what makes the verification below
possible.

## Layout

    channel/
      __init__.py      public API
      grid.py          wall-clustered finite-volume grids, y1+ targeting
      turbulence.py    mixing length, damping, stable gradient inversion
      solver.py        quadrature reference and finite-volume Picard solver
      reference.py     Spalding profile, log law, Dean correlations
      dns.py           reader for DNS mean-profile files
      postprocess.py   figures and error metrics
    tests/             pytest suite, 27 tests
    scripts/
      run_validation.py
    data/              DNS datasets, gitignored, see data/README.md
    figures/           regenerated on every run, gitignored

Run:

    python scripts/run_validation.py --dns 590
    pytest -q

Both from the project root. `--dns` takes a case label and resolves it inside
`data/chandata`, or an explicit path. With a label the friction Reynolds
number is read from the file header, so `--Re-tau` is only needed when running
without data.

## Verification and validation are kept separate

Verification asks whether the equations are being solved correctly.
Validation asks whether they are the correct equations. Conflating the two is
how a plausible-looking profile hides a discretisation error.

**Verification.** Two independent solvers: analytic inversion plus Gauss
quadrature, and a conservative finite-volume discretisation with Picard
iteration on the eddy viscosity. At Re_tau = 395 they agree on bulk velocity
to 0.007 per cent, and the finite-volume error against the reference falls at
an observed order of 2.00 under grid refinement. The shear balance figure is
the sharper check: the computed viscous and turbulent stresses must sum to the
exact straight line, and the residual panel shows what the discretisation is
actually costing.

**Validation.** Three independent references, in increasing order of value:

At Re_tau = 587.19, against MKM chan590:

| Check | Bare | Outer cap, lambda = 0.09 |
|---|---|---|
| rms deviation from DNS, y+ >= 1 | 0.664 | 0.298 |
| max deviation from DNS | 2.023 | 0.543 |
| Centreline U+ (DNS: 21.263) | 19.241 | 21.806 |
| Skin friction vs Dean | +7.0 per cent | -3.1 per cent |
| Fitted kappa, B (inputs 0.41, expected B ~ 5.2) | 0.4095, 5.013 | |

Error by wall-distance band, mean U+ deviation:

| y+ band | Bare | Capped |
|---|---|---|
| 1 to 5 | -0.006 | -0.006 |
| 5 to 30 | -0.216 | -0.216 |
| 30 to 117 | -0.361 | -0.361 |
| 117 to 352 | -0.788 | -0.405 |
| 352 to 587 | -1.790 | +0.206 |

The two closures are identical below the outer layer. All of the improvement
sits above y+ = 117, which localises the deficiency precisely.

## Known deficiencies, stated rather than hidden

1. **The bare model over-predicts friction by 7 to 9 per cent** across
   Re_tau = 395 to 2000, and under-predicts centreline velocity by 9.5 per
   cent at Re_tau = 587. The mixing length grows as kappa*y right to the
   centreline, so the outer layer carries too much eddy viscosity, the profile
   flattens and the bulk velocity comes out low. The Escudier cap brings both
   inside a few per cent. There is a test that asserts the deficiency exists,
   so that it cannot be silently removed.

   Note that the cap improves the profile but over-corrects the friction, from
   +7.0 to -3.1 per cent. These are not in conflict: Cf = 2/Ub+^2 is set by the
   integral of the profile, and a one-parameter algebraic closure cannot
   satisfy the local profile and the integral simultaneously. Tuning lambda
   until both agree would be fitting, not modelling.

2. **Fitted log-law constants are biased low at moderate Re_tau.** At
   Re_tau = 395 the conventional window y+ in [50, 0.15 Re_tau] contains six
   points and the shear has already decayed 15 per cent across it. Running the
   same solver with `constant_stress=True` removes the geometry and returns
   kappa and B close to their expected values, which localises the bias to the
   flow rather than the closure.

3. **No outer-layer wake, no transition, no roughness, no property variation.**
   An algebraic closure carries no history and no transport of turbulence, so
   nothing here extends to separated or developing flow.

## DNS data

Not versioned. See `data/README.md` for the source, the expected directory
layout and the column indices. Two things bite:

* The filename label is nominal. `chan590` was run at Re_tau = 587.19, stated
  in its header. Running the solver at 590 introduces a shear-distribution
  mismatch that then gets misattributed to the turbulence model. The header
  value is read automatically and cross-checked against the centreline y+.
* The header's citation line contains `Re_tau=590` inside the paper title, so
  the parser anchors its match to the start of the comment body. A substring
  search returns the wrong number silently.

With no file present the validation falls back to Spalding's composite
profile, which exercises the inner layer but says nothing about the outer one.

## Next

* **Tier 2, energy equation.** Same grid, same eddy viscosity, constant wall
  heat flux, `Pr_t = 0.85` then a variable `Pr_t`. Extract
  `h = q_w'' / (T_w - T_b)` and `Nu = h D_h / k`, validate against Gnielinski
  rather than Dittus-Boelter, whose own scatter is around 20 per cent and
  would swallow the model error.
* **Tier 3, near-wall resolution study.** Sweep y1+ from 0.5 to 100 with the
  grid targeting already implemented, plot Nusselt error against y1+, then
  repeat with a wall function and show it recovers above y1+ of about 30 and
  that both fail in the buffer layer.
* **Reynolds stress comparison.** `chan590.reystress` gives DNS -u'v'+ directly.
  The Boussinesq closure is being asked to reproduce that profile, so plotting
  it against the modelled turbulent stress is more diagnostic than comparing
  velocity, which is an integral and hides local error.
* **Tier 4, Wilcox k-omega.** Two transport equations on the same finite-volume
  machinery, wall boundary condition `omega_w = 6 nu / (beta_1 y_1^2)`. SST
  adds little in a fully developed channel, where the blending functions and
  the shear limiter are close to dormant.
