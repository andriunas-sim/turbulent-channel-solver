# DNS validation data

Not versioned. `data/chandata/` is in `.gitignore`; this file records how to
recreate it.

## Source

Moser, Kim and Mansour (1999), direct numerical simulation of fully developed
plane turbulent channel flow, spectral method of Kim, Moin and Moser (1987).
Published as Physics of Fluids **11**(4), 943-945.

Download the complete statistical database, a gzipped tar of about 1.3 MB,
from the turbulence file server hosted by the Oden Institute at UT Austin:

    https://turbulence.oden.utexas.edu/MKM_1999.html

Verify the URL before use; the site has moved once already, and the header of
each file still points at the retired TAM Illinois address.

## Layout

Unpack into `data/` keeping the archive's own directory structure. The code
resolves paths from a case label via `channel.dns.dns_path`, which expects:

    data/chandata/chan590/profiles/chan590.means
    data/chandata/chan590/profiles/chan590.reystress
    data/chandata/chan590/balances/...
    data/chandata/chan395/...
    data/chandata/chan180/...

Keeping the structure preserves provenance, and the `README` inside each
`profiles/` directory documents the column layout for all eight file types.

## Nominal labels are not the achieved Reynolds numbers

The filenames carry round labels. The header of each file states the value the
simulation actually reached, and that is the number the solver must be run at.
For chan590 the header reads `Re_tau = 587.19`. Check the other two against
their own headers rather than assuming the pattern.

`channel.dns.header_Re_tau` reads this automatically, and `load_means` raises
if the stated value disagrees with the last y+ in the data, which catches a
misread column. Note that the citation line in the header contains the nominal
label inside the paper title, so the parser anchors its match to the start of
the comment body rather than searching for a substring.

## Columns in the `.means` files

Zero-indexed, as the code addresses them:

    0: y   1: y+   2: Umean   3: dUmean/dy   4: Wmean   5: dWmean/dy   6: Pmean

`Umean` is normalised by u_tau per the header, so it is U+ directly. Defaults
are `y_col=1`, `u_col=2`.

Column 0 is also monotonic, so passing `y_col=0` by mistake will not trip the
monotonicity check. It will trip the centreline cross-check instead, provided
the header states Re_tau.
