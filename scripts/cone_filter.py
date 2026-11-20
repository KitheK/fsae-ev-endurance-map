"""Filter map markers into edge cones vs deliberate on-course slalom/gates."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point


def filter_cones(
    markers: list,
    line_en: np.ndarray,
    *,
    edge_min: float = 2.5,
    edge_max: float = 12.0,
    on_line_max: float = 1.8,
    spot_clear_m: float = 4.0,
    spot_xy: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float, str]]:
    """
    markers: entries [lon, lat, e, n, kind, acx, acz] from extract_ev_cones.
    Returns (lon, lat, kind) for placement.
    """
    line = LineString(line_en)
    pts = []
    for m in markers:
        lon, lat, e, n, kind = float(m[0]), float(m[1]), float(m[2]), float(m[3]), m[4]
        pts.append({"lon": lon, "lat": lat, "e": e, "n": n, "kind": kind, "d": line.distance(Point(e, n))})

    blue_en = [(p["e"], p["n"]) for p in pts if p["kind"] == "ConeBlue"]
    on_line = [p for p in pts if p["d"] <= on_line_max]

    def near_blue(e, n, r=10.0):
        return any(math.hypot(e - be, n - bn) <= r for be, bn in blue_en)

    def slalom_ok(p) -> bool:
        if p["kind"] == "ConeBlue":
            return p["d"] <= on_line_max + 2.0
        if p["d"] > on_line_max:
            return False
        if near_blue(p["e"], p["n"], 12.0):
            return True
        # Chicane: another on-line cone within ~4 m (map scale)
        for q in on_line:
            if q is p:
                continue
            if math.hypot(p["e"] - q["e"], p["n"] - q["n"]) < 4.5:
                return True
        return False

    kept: list[tuple[float, float, str]] = []
    for p in pts:
        if p["kind"] == "ConeBlue" and slalom_ok(p):
            kept.append((p["lon"], p["lat"], "ConeBlue"))
            continue
        if p["kind"] != "ConeOrange":
            continue
        if edge_min <= p["d"] <= edge_max:
            kept.append((p["lon"], p["lat"], "ConeOrange"))
        elif slalom_ok(p):
            kept.append((p["lon"], p["lat"], "ConeOrange"))

    kept = fill_edge_cones(kept, line_en, spacing_m=7.0, offset_m=4.5, match_m=3.2)

    if spot_xy:
        filtered = []
        for lo, la, kind in kept:
            x, z = _ll2ac_simple(lo, la)
            if all(math.hypot(x - sx, z - sz) >= spot_clear_m for sx, sz in spot_xy):
                filtered.append((lo, la, kind))
        kept = filtered

    return kept


def fill_edge_cones(
    kept: list[tuple[float, float, str]],
    line_en: np.ndarray,
    spacing_m: float = 7.0,
    offset_m: float = 4.5,
    match_m: float = 3.0,
) -> list[tuple[float, float, str]]:
    """Place edge cones along centreline where the map has no nearby marker (open pavement edges)."""
    LAT0, LON0 = 42.0665, -84.2390
    line = LineString(line_en)
    existing = [Point(ll2en_m(lo, la)) for lo, la, _ in kept]

    def add_point(e, n, kind="ConeOrange"):
        lon = e / (111320 * math.cos(math.radians(LAT0))) + LON0
        lat = n / 110574 + LAT0
        kept.append((lon, lat, kind))

    def near_marker(e, n) -> bool:
        p = Point(e, n)
        return any(p.distance(ex) < match_m for ex in existing)

    d = 0.0
    while d < line.length:
        p = line.interpolate(d)
        p2 = line.interpolate(min(d + 0.5, line.length))
        de, dn = p2.x - p.x, p2.y - p.y
        L = math.hypot(de, dn) + 1e-9
        nx, ny = -dn / L, de / L
        for s in (1, -1):
            e, n = p.x + s * offset_m * nx, p.y + s * offset_m * ny
            if near_marker(e, n):
                continue
            add_point(e, n)
            existing.append(Point(e, n))
        d += spacing_m
    return kept


def ll2en_m(lon, lat):
    LAT0, LON0 = 42.0665, -84.2390
    return (
        (lon - LON0) * 111320 * math.cos(math.radians(LAT0)),
        (lat - LAT0) * 110574,
    )


def _ll2ac_simple(lon, lat):
    LAT0, LON0 = 42.0665, -84.2390
    e = (lon - LON0) * 111320 * math.cos(math.radians(LAT0))
    n = (lat - LAT0) * 110574
    x = -9.99629818e-01 * e + 6.92960356e-04 * n + 8.99722210e01
    z = 6.47160377e-04 * e + 1.00637171e00 * n - 6.10772977e01
    return x, z


def road_edges_ac(cx, cz, half_w: float):
    """Left/right edge polylines in AC (x,z) for debug drawings."""
    left, right = [], []
    n = len(cx)
    for i in range(n):
        if i == 0:
            dx, dz = cx[1] - cx[0], cz[1] - cz[0]
        elif i == n - 1:
            dx, dz = cx[-1] - cx[-2], cz[-1] - cz[-2]
        else:
            dx, dz = cx[i + 1] - cx[i - 1], cz[i + 1] - cz[i - 1]
        L = math.hypot(dx, dz) + 1e-9
        nx, nz = -dz / L, dx / L
        left.append((cx[i] + half_w * nx, cz[i] + half_w * nz))
        right.append((cx[i] - half_w * nx, cz[i] - half_w * nz))
    return left, right