#!/usr/bin/env python3
"""Detect cone markers on official_ev_endurance_2026.png and write tools/ev_cones.pkl."""

import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from evov import evcoef, ev2z18  # noqa: E402
from geo import gpx2ll, LAT0, LON0  # noqa: E402

CO = evcoef(0, 113, 1.0, 13.5)
BX = [-9.99629818e-01, 6.92960356e-04, 8.99722210e01]
BZ = [6.47160377e-04, 1.00637171e00, -6.10772977e01]


def uv2ll(u, v):
    x, y = ev2z18(u, v, CO)
    return gpx2ll(69727 * 256 + x, 97241 * 256 + y, 18)


def ll2en(lon, lat):
    return (
        (lon - LON0) * 111320 * math.cos(math.radians(LAT0)),
        (lat - LAT0) * 110574,
    )


def en2ac(e, n):
    return BX[0] * e + BX[1] * n + BX[2], BZ[0] * e + BZ[1] * n + BZ[2]


def main():
    im = np.array(Image.open(ROOT / "official_ev_endurance_2026.png").convert("RGB"))
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    orange = (r > 190) & (g < 130) & (b < 100) & (r.astype(int) - g.astype(int) > 50)
    blue = (b > 170) & (r < 120) & (g < 150)
    ys, xs = np.where(orange | blue)
    cells = defaultdict(list)
    for x, y in zip(xs, ys):
        cells[(x // 5, y // 5)].append((x, y, bool(orange[y, x])))
    centers = []
    for v in cells.values():
        u = float(np.mean([p[0] for p in v]))
        v_ = float(np.mean([p[1] for p in v]))
        kind = "ConeOrange" if sum(p[2] for p in v) >= len(v) / 2 else "ConeBlue"
        centers.append((u, v_, kind))
    out = []
    for u, v, kind in centers:
        lo, la = uv2ll(u, v)
        e, n = ll2en(lo, la)
        out.append([lo, la, e, n, kind, *en2ac(e, n)])
    pkl = ROOT / "tools" / "ev_cones.pkl"
    pickle.dump(out, open(pkl, "wb"))
    print(f"Wrote {len(out)} cones -> {pkl}")


if __name__ == "__main__":
    main()