"""Reader for DNS channel-flow mean profiles.

Written against the Moser, Kim and Mansour (1999) ASCII layout. Those files
carry a header of lines beginning with '#' followed by whitespace-separated
numeric columns; for the `.means` files the columns are, zero-indexed as the
code addresses them,

    0: y   1: y+   2: Umean   3: dUmean/dy   4: Wmean   5: dWmean/dy   6: Pmean

so the defaults are y_col=1 and u_col=2. Umean is normalised by u_tau per the
header, so it is U+ directly and needs no rescaling. Note that column 0 is
also monotonic, so passing y_col=0 by mistake will not trip the monotonicity
check; it silently compares against a profile compressed into y+ in [0, 1].
The first data row is the quick check: y+ should be much larger than y.

The data is not redistributed with this package. Download the tarball from the
turbulence file server hosted by the Oden Institute at UT Austin and unpack it
into `data/` keeping its directory structure. Verify the current URL before
use. With no file present, `scripts/run_validation.py` falls back to
Spalding's profile, which validates the near-wall behaviour but not the outer
layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["DnsProfile", "load_means", "dns_path", "load_case", "DNS_ROOT", "NOMINAL_CASES"]

COMMENT_CHARS = ("#", "%", "!")

DNS_ROOT = Path("data/chandata")

#: Nominal case labels used in the filenames. The achieved friction Reynolds
#: numbers differ from these labels and are read from each file's header at
#: load time rather than hard-coded, because the label is a name and the
#: header value is the number the solver has to match. For chan590 the header
#: states 587.19.
NOMINAL_CASES = (180, 395, 590)


@dataclass(frozen=True)
class DnsProfile:
    y_plus: np.ndarray
    U_plus: np.ndarray
    Re_tau: float
    source: str

    def interpolate(self, y_plus: np.ndarray) -> np.ndarray:
        """U+ interpolated onto arbitrary wall distances, in log space in y+.

        Linear interpolation against log(y+) rather than y+, because the data
        are clustered geometrically at the wall and linear interpolation on a
        linear abscissa loses the sublayer.
        """
        y = np.asarray(y_plus, dtype=float)
        ok = self.y_plus > 0.0
        out = np.interp(
            np.log(np.maximum(y, 1e-12)),
            np.log(self.y_plus[ok]),
            self.U_plus[ok],
            left=np.nan,
            right=np.nan,
        )
        return np.where(y <= 0.0, 0.0, out)


def dns_path(case: int | str, kind: str = "means", root: Path | str = DNS_ROOT) -> Path:
    """Locate a profile file inside the unpacked MKM tarball.

    The tarball structure is kept exactly as distributed:

        data/chandata/chan590/profiles/chan590.means

    so the case label is enough to find any of the profile files, and the
    directory itself records the provenance. `kind` selects the file
    extension: means, reystress, quad, vortvar, flat, skew or velp.
    """
    case = str(case)
    path = Path(root) / f"chan{case}" / "profiles" / f"chan{case}.{kind}"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Unpack the MKM 1999 tarball into "
            f"{Path(root)} keeping its directory structure, or pass an "
            "explicit path instead of a case label."
        )
    return path


def header_Re_tau(path: str | Path) -> float | None:
    """Read the achieved Re_tau from the file header, if it is stated.

    The filename is a nominal label and the header carries the real value.
    Running the solver at the label rather than the achieved value introduces
    a shear-distribution mismatch that then gets misattributed to the
    turbulence model.

    The match is anchored to the start of the comment body rather than run as
    a substring search, because the header's citation line reads

        # Reference: DNS of Turbulent Channel Flow up to Re_tau=590, 1999,

    and a substring search returns the nominal 590 from the paper title before
    ever reaching the declaration `# Re_tau = 587.19` four lines below it. That
    failure is silent and gives a plausible wrong number, which is the worst
    kind.
    """
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text[0] not in COMMENT_CHARS:
            continue
        body = text.lstrip("".join(COMMENT_CHARS)).strip().lower().replace(" ", "")
        for key in ("re_tau=", "retau="):
            if body.startswith(key):
                try:
                    return float(body[len(key):].split(",")[0])
                except ValueError:
                    continue
    return None


def _numeric_rows(path: Path) -> np.ndarray:
    rows = []
    width = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s[0] in COMMENT_CHARS:
            continue
        try:
            vals = [float(t) for t in s.split()]
        except ValueError:
            continue
        if width is None:
            width = len(vals)
        if len(vals) == width:
            rows.append(vals)
    if not rows:
        raise ValueError(f"no numeric rows found in {path}")
    return np.asarray(rows, dtype=float)


def load_means(
    path: str | Path,
    y_col: int = 1,
    u_col: int = 2,
    Re_tau: float | None = None,
) -> DnsProfile:
    """Load a DNS mean-velocity profile.

    If `Re_tau` is not supplied it is taken as the largest y+ in the file,
    which is correct for a half-channel profile that runs to the centreline.
    A ValueError is raised if the nominated y+ column is not monotonic, which
    catches the usual mistake of picking the wrong column index.
    """
    path = Path(path)
    data = _numeric_rows(path)
    stated = header_Re_tau(path)
    if Re_tau is None:
        Re_tau = stated
    if max(y_col, u_col) >= data.shape[1]:
        raise ValueError(
            f"{path.name} has {data.shape[1]} columns; requested indices "
            f"{y_col} and {u_col}"
        )
    y = data[:, y_col]
    u = data[:, u_col]
    if np.any(np.diff(y) <= 0.0):
        raise ValueError(
            f"column {y_col} of {path.name} is not strictly increasing; "
            "check the column indices against the file header"
        )
    if y[0] < 0.0:
        raise ValueError("y+ column contains negative values")
    # The last y+ point is the centreline, so it must equal Re_tau. If the
    # header value and the data disagree, one of them is being misread.
    if stated is not None and not np.isclose(y[-1], stated, rtol=5e-3):
        raise ValueError(
            f"{path.name}: header states Re_tau = {stated:g} but the y+ column "
            f"ends at {y[-1]:g}. Check the column indices, or pass Re_tau "
            "explicitly if this file is not a half-channel profile."
        )
    return DnsProfile(
        y_plus=y,
        U_plus=u,
        Re_tau=float(Re_tau if Re_tau is not None else y[-1]),
        source=path.name,
    )


def load_case(case: int | str, root: Path | str = DNS_ROOT, **kwargs) -> DnsProfile:
    """Load a mean profile by case label, e.g. `load_case(590)`."""
    return load_means(dns_path(case, "means", root), **kwargs)
