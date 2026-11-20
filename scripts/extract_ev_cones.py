#!/usr/bin/env python3
"""Extract cone/gate markers from official EV map linework (not the red racing line)."""

from __future__ import annotations

import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

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


def component_centers(mask: np.ndarray, min_px: int = 6, max_px: int = 180, max_aspect: float = 3.5):
    labeled, n = ndimage.label(mask)
    if n == 0:
        return []
    slices = ndimage.find_objects(labeled)
    out = []
    for i, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        sub = labeled[slc] == i
        area = int(sub.sum())
        if area < min_px or area > max_px:
            continue
        h, w = sub.shape
        if max(h, w) / max(1, min(h, w)) > max_aspect:
            continue
        ys, xs = np.where(sub)
        cy = ys.mean() + slc[0].start
        cx = xs.mean() + slc[1].start
        out.append((float(cx), float(cy)))
    return out


def main():
    path = ROOT / "official_ev_endurance_2026.png"
    im = np.array(Image.open(path).convert("RGB"))
    L = im.mean(2)
    # Red/orange = traced racing line on the PDF — not cone glyphs.
    race = (im[..., 0] > 175) & (im[..., 0].astype(int) - im[..., 1].astype(int) > 45) & (im[..., 2] < 120)
    dark = (L < 88) & (~race)
    blue = (im[..., 2] > 175) & (im[..., 0] < 115) & (im[..., 1] < 145) & (~race)

    cone_centers = component_centers(dark, min_px=4, max_px=400, max_aspect=8.0)
    gate_centers = component_centers(blue, min_px=4, max_px=400, max_aspect=12.0)

    # De-duplicate dense cells (cone icons overlap when rasterised)
    def cluster(points, cell=7):
        cells: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
        for u, v in points:
            cells[(int(u) // cell, int(v) // cell)].append((u, v))
        return [(float(np.mean([p[0] for p in v])), float(np.mean([p[1] for p in v]))) for v in cells.values() if v]

    cone_uv = cluster(cone_centers)
    gate_uv = cluster(gate_centers)

    out = []
    for u, v in cone_uv:
        lo, la = uv2ll(u, v)
        e, n = ll2en(lo, la)
        out.append([lo, la, e, n, "ConeOrange", *en2ac(e, n)])
    for u, v in gate_uv:
        lo, la = uv2ll(u, v)
        e, n = ll2en(lo, la)
        out.append([lo, la, e, n, "ConeBlue", *en2ac(e, n)])

    pkl = ROOT / "tools" / "ev_cones_raw.pkl"
    pickle.dump(out, open(pkl, "wb"))
    print(f"Wrote {len(out)} raw markers ({len(cone_uv)} dark, {len(gate_uv)} blue) -> {pkl}")


if __name__ == "__main__":
    main()