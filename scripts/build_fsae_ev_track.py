#!/usr/bin/env python3
"""Build standalone Assetto Corsa track: FSAE Michigan 2026 EV Endurance."""

from __future__ import annotations

import json
import math
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "assetto_corsa" / "fsae_michigan_2026_ev"
ARTIFACTS = ROOT / "artifacts"

sys.path.insert(0, str(SCRIPTS))
from cone_filter import filter_cones, road_edges_ac  # noqa: E402
from kn5_writer import (  # noqa: E402
    Material,
    Mesh,
    Node,
    Texture,
    dds_bc1,
    transform_node,
    write_kn5,
)

LAT0, LON0 = 42.0665, -84.2390
# ENU (m) -> AC world (m), fitted to fsae_michigan18 spots (~2 mm RMS)
EN2AC = np.array(
    [
        [-9.99629818e-01, 6.92960356e-04, 8.99722210e01],
        [6.47160377e-04, 1.00637171e00, -6.10772977e01],
        [0.0, 0.0, 1.0],
    ]
)
Z18_X0, Z18_Y0 = 69727, 97241
RES18_M = 0.4437


def ll2en(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = np.asarray(lon), np.asarray(lat)
    e = (lon - LON0) * 111320 * math.cos(math.radians(LAT0))
    n = (lat - LAT0) * 110574
    return e, n


def en2ac(e, n) -> tuple[np.ndarray, np.ndarray]:
    e, n = np.asarray(e), np.asarray(n)
    x = EN2AC[0, 0] * e + EN2AC[0, 1] * n + EN2AC[0, 2]
    z = EN2AC[1, 0] * e + EN2AC[1, 1] * n + EN2AC[1, 2]
    return x, z


def ll2ac(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    return en2ac(*ll2en(lon, lat))


def ll2gpx(lon, lat, z):
    lon, lat = np.asarray(lon), np.asarray(lat)
    n = 256 * 2**z
    x = (lon + 180) / 360 * n
    y = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * n
    return x, y


def en2z18px(e, n):
    lon = np.asarray(e) / (111320 * math.cos(math.radians(LAT0))) + LON0
    lat = np.asarray(n) / 110574 + LAT0
    x, y = ll2gpx(lon, lat, 18)
    return float(x - Z18_X0 * 256), float(y - Z18_Y0 * 256)


def sample_line_xyz(lon, lat, step=2.0):
    e, n = ll2en(lon, lat)
    line = LineString(np.c_[e, n])
    dists = np.arange(0, line.length, step)
    pts = [line.interpolate(d) for d in dists]
    if line.length - dists[-1] > 0.5:
        pts.append(Point(line.coords[-1]))
    xy = np.array([[p.x, p.y] for p in pts])
    x, z = en2ac(xy[:, 0], xy[:, 1])
    y = np.zeros_like(x)
    return x, y, z, xy, float(line.length)


def heading_at(x, z, i=0):
    dx = x[min(i + 1, len(x) - 1)] - x[max(i - 1, 0)]
    dz = z[min(i + 1, len(z) - 1)] - z[max(i - 1, 0)]
    return math.atan2(dx, dz)


def strip_mesh(cx, cz, half_w: float, mat: int, name: str) -> Node:
    """Ribbon along center polyline cx,cz with constant width."""
    n = len(cx)
    verts_pos, verts_uv, idx_list = [], [], []
    for i in range(n):
        if i == 0:
            dx, dz = cx[1] - cx[0], cz[1] - cz[0]
        elif i == n - 1:
            dx, dz = cx[-1] - cx[-2], cz[-1] - cz[-2]
        else:
            dx, dz = cx[i + 1] - cx[i - 1], cz[i + 1] - cz[i - 1]
        L = math.hypot(dx, dz) + 1e-9
        nx, nz = -dz / L, dx / L
        for s in (-1, 1):
            verts_pos.append([cx[i] + s * half_w * nx, 0.0, cz[i] + s * half_w * nz])
            verts_uv.append([i / max(n - 1, 1), (s + 1) * 0.5])
    verts_pos = np.array(verts_pos, np.float32)
    verts_uv = np.array(verts_uv, np.float32)
    verts_nrm = np.tile([0, 1, 0], (len(verts_pos), 1)).astype(np.float32)
    for i in range(n - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1
        idx_list += [a, c, b, b, c, d]
    return Node(
        name,
        Mesh(verts_pos, verts_nrm, verts_uv, np.array(idx_list, np.uint16), mat),
    )


def triangulate_polygon(poly: Polygon, y: float, uvs: np.ndarray | None = None) -> Mesh:
    from shapely.geometry import mapping

    ext = np.array(poly.exterior.coords[:-1])
    # fan triangulation (convex hull is fine for our pads)
    cen = ext.mean(0)
    verts = []
    uvs_out = []
    idx = []
    ci = len(verts)
    verts.append([cen[0], y, cen[1]])
    uvs_out.append([0.5, 0.5])
    for i, (e, n) in enumerate(ext):
        verts.append([e, y, n])
        if uvs is not None:
            uvs_out.append(uvs[i])
        else:
            uvs_out.append([0, 0])
        idx.extend([ci, ci + 1 + i, ci + 1 + ((i + 1) % len(ext))])
    return Mesh(
        np.array(verts, np.float32),
        np.tile([0, 1, 0], (len(verts), 1)).astype(np.float32),
        np.array(uvs_out, np.float32),
        np.array(idx, np.uint16),
        0,
    )


def cone_cylinder(x, z, radius=0.15, height=0.7, mat=0, segments=8):
    verts, nrms, uvs, idx = [], [], [], []
    base = len(verts)
    for s in range(segments):
        ang = 2 * math.pi * s / segments
        verts.append([x + radius * math.cos(ang), 0, z + radius * math.sin(ang)])
        verts.append([x + radius * math.cos(ang), height, z + radius * math.sin(ang)])
        nrms += [[math.cos(ang), 0, math.sin(ang)]] * 2
        uvs += [[0, 0], [0, 1]]
    for s in range(segments):
        i0 = base + 2 * s
        i1 = base + 2 * ((s + 1) % segments)
        idx += [i0, i1, i0 + 1, i0 + 1, i1, i1 + 1]
    return verts, nrms, uvs, idx


def merge_meshes(parts: list[tuple], mat: int) -> Mesh:
    vp, vn, vu, idx = [], [], [], []
    off = 0
    for pos, nrm, uv, ix in parts:
        vp.extend(pos)
        vn.extend(nrm)
        vu.extend(uv)
        idx.extend([i + off for i in ix])
        off += len(pos)
    return Mesh(
        np.array(vp, np.float32),
        np.array(vn, np.float32),
        np.array(vu, np.float32),
        np.array(idx, np.uint16),
        mat,
    )


def wall_mesh_segments(poly: Polygon, height=1.2, thick=0.35, mat=0) -> list:
    ext = np.array(poly.exterior.coords)
    parts = []
    for i in range(len(ext) - 1):
        x0, z0 = ext[i]
        x1, z1 = ext[i + 1]
        dx, dz = x1 - x0, z1 - z0
        L = math.hypot(dx, dz) + 1e-9
        nx, nz = -dz / L, dx / L
        corners = [
            [x0 - nx * thick, 0, z0 - nz * thick],
            [x0 + nx * thick, 0, z0 + nz * thick],
            [x1 + nx * thick, 0, z1 + nz * thick],
            [x1 - nx * thick, 0, z1 - nz * thick],
            [x0 - nx * thick, height, z0 - nz * thick],
            [x0 + nx * thick, height, z0 + nz * thick],
            [x1 + nx * thick, height, z1 + nz * thick],
            [x1 - nx * thick, height, z1 - nz * thick],
        ]
        idx = [
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
            4, 5, 6, 4, 6, 7,
        ]
        nrm = [[0, 1, 0]] * 8
        uv = [[0, 0]] * 8
        parts.append((corners, nrm, uv, idx))
    return parts


def write_fast_lane(path: Path, cx, cy, cz, half_width=5.0):
    xs = np.r_[cx, cx[0]]
    ys = np.r_[cy, cy[0]]
    zs = np.r_[cz, cz[0]]
    pts_n = len(xs)
    dist = [0.0]
    for i in range(1, pts_n):
        dist.append(dist[-1] + math.hypot(xs[i] - xs[i - 1], zs[i] - zs[i - 1]))
    dist = np.array(dist, np.float32)

    buf = bytearray()
    buf += struct.pack("<4i", 0, pts_n, 0, 7)
    for i in range(pts_n):
        buf += struct.pack("<4fi", float(xs[i]), float(ys[i]), float(zs[i]), float(dist[i]), i)
    for i in range(pts_n):
        dx = xs[(i + 1) % pts_n] - xs[i - 1 if i else pts_n - 2]
        dz = zs[(i + 1) % pts_n] - zs[i - 1 if i else pts_n - 2]
        L = math.hypot(dx, dz) + 1e-9
        fx, fz = dx / L, dz / L
        detail = [
            half_width,
            half_width,
            60.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            fx,
            0.0,
            fz,
            0.0,
        ]
        buf += struct.pack("<18f", *detail)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tex_dir = OUT / "texture"
    tex_dir.mkdir(exist_ok=True)

    cl = json.loads((ROOT / "official_line.geojson").read_text())
    lon, lat = np.array(cl["features"][0]["geometry"]["coordinates"]).T
    cx, cy, cz, en_xy, lap_m = sample_line_xyz(lon, lat, step=2.0)

    raw_pkl = TOOLS / "ev_cones_raw.pkl"
    if not raw_pkl.exists():
        raise SystemExit("Run scripts/extract_ev_cones.py first (ev_cones_raw.pkl missing)")
    raw_markers = pickle.loads(raw_pkl.read_bytes())

    road_half = 5.0
    line_en = LineString(en_xy)
    road_poly = line_en.buffer(road_half, cap_style=2, join_style=2)
    bx0, by0, bx1, by1 = road_poly.buffer(50).bounds

    # Ground texture crop from MIS z18 mosaic
    sat = np.array(Image.open(TOOLS / "mis_z18.jpg").convert("RGB"))
    sh, sw = sat.shape[:2]
    px0, py0 = en2z18px(bx0, by0)
    px1, py1 = en2z18px(bx1, by1)
    ix0, iy0 = int(max(0, min(px0, px1) - 30)), int(max(0, min(py0, py1) - 30))
    ix1, iy1 = int(min(sw, max(px0, px1) + 30)), int(min(sh, max(py0, py1) + 30))
    crop = sat[iy0:iy1, ix0:ix1]
    # Power-of-two for DDS
    cw, ch = 1 << (crop.shape[1] - 1).bit_length(), 1 << (crop.shape[0] - 1).bit_length()
    cw, ch = min(cw, 2048), min(ch, 2048)
    ground_tex = np.array(Image.fromarray(crop).resize((cw, ch), Image.LANCZOS))
    dds_name = "ground_sat.dds"
    (tex_dir / dds_name).write_bytes(dds_bc1(ground_tex))
    Image.fromarray(crop).save(tex_dir / "ground_sat_src.png")

    ax0, az0 = en2ac(bx0, by0)
    ax1, az1 = en2ac(bx1, by1)

    def ac_to_uv(ax, az):
        det = EN2AC[0, 0] * EN2AC[1, 1] - EN2AC[0, 1] * EN2AC[1, 0]
        e = (EN2AC[1, 1] * (ax - EN2AC[0, 2]) - EN2AC[0, 1] * (az - EN2AC[1, 2])) / det
        n = (-EN2AC[1, 0] * (ax - EN2AC[0, 2]) + EN2AC[0, 0] * (az - EN2AC[1, 2])) / det
        px, py = en2z18px(e, n)
        u = (px - ix0) / max(ix1 - ix0, 1)
        v = (py - iy0) / max(iy1 - iy0, 1)
        return float(np.clip(u, 0, 1)), float(np.clip(v, 0, 1))

    pad_corners_ac = [
        (min(ax0, ax1), min(az0, az1)),
        (max(ax0, ax1), min(az0, az1)),
        (max(ax0, ax1), max(az0, az1)),
        (min(ax0, ax1), max(az0, az1)),
    ]
    pad_poly = Polygon(pad_corners_ac)
    ext = np.array(pad_poly.exterior.coords[:-1])
    uvs = np.array([ac_to_uv(p[0], p[1]) for p in ext], np.float32)

    grass_vis = triangulate_polygon(pad_poly, y=-0.15, uvs=uvs)
    grass_phys = triangulate_polygon(pad_poly, y=-0.05, uvs=uvs)

    road_vis = strip_mesh(cx, cz, road_half + 0.35, 1, "ROAD_0")
    road_phys = strip_mesh(cx, cz, road_half, 0, "1ROAD_0")

  # Staging / pit lon-lat (for cone keep-out before meshes)
    lon0, lat0 = lon[0], lat[0]
    e0, n0 = ll2en(lon0, lat0)
    de, dn = en_xy[1, 0] - e0, en_xy[1, 1] - n0
    de, dn = de / (math.hypot(de, dn) + 1e-9), dn / (math.hypot(de, dn) + 1e-9)
    lat_dir_e, lat_dir_n = -dn, de
    spot_specs: list[tuple[str, float, float, float]] = []
    spot_xy: list[tuple[float, float]] = []
    h0 = heading_at(cx, cz, 0)
    for i in range(20):
        along = 8.0 + (i // 2) * 12.0
        side = (i % 2) * 3.5
        e = e0 - de * along + lat_dir_e * side
        n = n0 - dn * along + lat_dir_n * side
        lon_p = e / (111320 * math.cos(math.radians(LAT0))) + LON0
        lat_p = n / 110574 + LAT0
        spot_specs.append((f"AC_START_{i}", lon_p, lat_p, h0))
        x, z = ll2ac(lon_p, lat_p)
        spot_xy.append((float(x), float(z)))
        e_pit = e + lat_dir_e * 4.0
        n_pit = n + lat_dir_n * 4.0
        lon_pit = e_pit / (111320 * math.cos(math.radians(LAT0))) + LON0
        lat_pit = n_pit / 110574 + LAT0
        spot_specs.append((f"AC_PIT_{i}", lon_pit, lat_pit, h0))
        xp, zp = ll2ac(lon_pit, lat_pit)
        spot_xy.append((float(xp), float(zp)))

    cone_pts = filter_cones(raw_markers, en_xy, spot_xy=spot_xy, spot_clear_m=6.0)
    left_edge, right_edge = road_edges_ac(cx, cz, road_half)

    cone_o, cone_b = [], []
    for lo, la, kind in cone_pts:
        x, z = ll2ac(lo, la)
        blk = cone_cylinder(float(x), float(z))
        (cone_b if kind == "ConeBlue" else cone_o).append(blk)
    cone_o_mesh = merge_meshes(cone_o, 3) if cone_o else None
    cone_b_mesh = merge_meshes(cone_b, 4) if cone_b else None

    margin = 55
    wx0, wz0 = float(cx.min() - margin), float(cz.min() - margin)
    wx1, wz1 = float(cx.max() + margin), float(cz.max() + margin)
    wall_loop = Polygon([(wx0, wz0), (wx1, wz0), (wx1, wz1), (wx0, wz1)])
    wall_parts = wall_mesh_segments(wall_loop, mat=5)
    wall_mesh = merge_meshes(wall_parts, 5)

    mats = [
        Material(
            "Tarmac",
            "ksPerPixelMultiMap",
            [("ksAmbient", 0.05), ("ksDiffuse", 0.65), ("ksSpecular", 0.1), ("ksSpecularEXP", 10.0)],
            [("txDiffuse", 0, "road.dds")],
        ),
        Material(
            "TarmacVis",
            "ksPerPixelMultiMap",
            [("ksAmbient", 0.08), ("ksDiffuse", 0.75), ("ksSpecular", 0.15), ("ksSpecularEXP", 12.0)],
            [("txDiffuse", 0, "road.dds")],
        ),
        Material(
            "GrassGround",
            "ksPerPixelMultiMap",
            [("ksAmbient", 0.1), ("ksDiffuse", 0.8), ("ksSpecular", 0.02), ("ksSpecularEXP", 5.0)],
            [("txDiffuse", 0, dds_name)],
        ),
        Material(
            "ConeOrange",
            "ksPerPixelMultiMap",
            [("ksAmbient", 0.1), ("ksDiffuse", 0.85), ("ksSpecular", 0.05), ("ksSpecularEXP", 8.0)],
            [("txDiffuse", 0, "cone_orange.dds")],
        ),
        Material(
            "ConeBlue",
            "ksPerPixelMultiMap",
            [("ksAmbient", 0.1), ("ksDiffuse", 0.85), ("ksSpecular", 0.05), ("ksSpecularEXP", 8.0)],
            [("txDiffuse", 0, "cone_blue.dds")],
        ),
        Material(
            "Wall",
            "ksPerPixelMultiMap",
            [("ksAmbient", 0.05), ("ksDiffuse", 0.55), ("ksSpecular", 0.05), ("ksSpecularEXP", 6.0)],
            [("txDiffuse", 0, "wall.dds")],
        ),
    ]

    def solid_dds(rgb):
        return dds_bc1(np.full((8, 8, 3), rgb, np.uint8))

    road_dds = dds_bc1(np.full((8, 8, 3), [55, 55, 58], np.uint8))
    (tex_dir / "road.dds").write_bytes(road_dds)
    (tex_dir / "cone_orange.dds").write_bytes(solid_dds([220, 120, 25]))
    (tex_dir / "cone_blue.dds").write_bytes(solid_dds([30, 80, 200]))
    (tex_dir / "wall.dds").write_bytes(solid_dds([140, 140, 135]))

    textures = [
        Texture("ground_sat.dds", (tex_dir / "ground_sat.dds").read_bytes()),
        Texture("road.dds", road_dds),
        Texture("cone_orange.dds", (tex_dir / "cone_orange.dds").read_bytes()),
        Texture("cone_blue.dds", (tex_dir / "cone_blue.dds").read_bytes()),
        Texture("wall.dds", (tex_dir / "wall.dds").read_bytes()),
    ]

    grass_vis.material = 2
    grass_phys.material = 2
    road_vis.mesh.material = 1
    road_phys.mesh.material = 0
    wall_mesh.material = 5

    children: list[Node] = [
        Node("1GRASS_0", grass_phys),
        Node("GRASS_0", grass_vis),
        road_phys,
        road_vis,
        Node("1WALL_0", wall_mesh),
    ]
    if cone_o_mesh:
        children.append(Node("CONES_ORANGE", cone_o_mesh))
    if cone_b_mesh:
        children.append(Node("CONES_BLUE", cone_b_mesh))

    def place_grid(name: str, lon_p, lat_p, heading):
        x, z = ll2ac(lon_p, lat_p)
        return transform_node(name, float(x), 0.0, float(z), heading)

    for name, lon_p, lat_p, hd in spot_specs:
        children.append(place_grid(name, lon_p, lat_p, hd))

    # Start/finish timing beam
    te = e0 + lat_dir_e * 6.0
    tn = n0 + lat_dir_n * 6.0
    for tag, side in (("AC_TIME_0_L", -5.5), ("AC_TIME_0_R", 5.5)):
        e = te + de * side
        n = tn + dn * side
        lon_p = e / (111320 * math.cos(math.radians(LAT0))) + LON0
        lat_p = n / 110574 + LAT0
        xp, zp = ll2ac(lon_p, lat_p)
        children.append(transform_node(tag, float(xp), 0.0, float(zp), h0 + math.pi / 2))

    # Second beam on back straight (~mid lap)
    mid_i = len(lon) // 2
    h1 = heading_at(cx, cz, mid_i)
    children.append(
        transform_node("AC_TIME_1_L", float(cx[mid_i] - 6 * math.sin(h1)), 0.0, float(cz[mid_i] - 6 * math.cos(h1)), h1 + math.pi / 2)
    )
    children.append(
        transform_node("AC_TIME_1_R", float(cx[mid_i] + 6 * math.sin(h1)), 0.0, float(cz[mid_i] + 6 * math.cos(h1)), h1 + math.pi / 2)
    )

    root = Node("Scene", children=children)
    kn5_bytes = write_kn5(textures, mats, root, version=5)
    kn5_path = OUT / "fsae_michigan_2026_ev.kn5"
    kn5_path.write_bytes(kn5_bytes)

    # Config files
    data = OUT / "data"
    data.mkdir(exist_ok=True)
    (data / "surfaces.ini").write_text(
        """[SURFACE_0]
KEY=ROAD
FRICTION=1.0
DAMPING=0
WAV=0
WAV_LEN=0
FF_EFFECT=0
BLACK_FLAG_TIME=0
IS_VALID_TRACK=1

[SURFACE_1]
KEY=GRASS
FRICTION=0.7
DAMPING=0.05
WAV=0
WAV_LEN=0
FF_EFFECT=0
BLACK_FLAG_TIME=0
IS_VALID_TRACK=0

[SURFACE_2]
KEY=WALL
FRICTION=0.1
DAMPING=0
WAV=0
WAV_LEN=0
FF_EFFECT=0
BLACK_FLAG_TIME=0
IS_VALID_TRACK=0
"""
    )
    (data / "map.ini").write_text(
        f"""[PARAMETERS]
WIDTH=1024
HEIGHT=1024
MARGIN=30
X_OFFSET={float(np.mean(cx)):.2f}
Z_OFFSET={float(np.mean(cz)):.2f}
SCALE_FACTOR=1.0
"""
    )
    write_fast_lane(data / "ai" / "fast_lane.ai", cx, cy, cz)

    ui = OUT / "ui"
    ui.mkdir(exist_ok=True)
    (ui / "ui_track.json").write_text(
        json.dumps(
            {
                "name": "FSAE Michigan 2026 EV Endurance",
                "description": "FSAE Electric 2026 EV Endurance layout at Michigan International Speedway "
                "(infield / back-straight paddock). Flat ~1.22 km lap from official fsaeonline map.",
                "tags": ["circuit", "USA", "original", "fsae"],
                "geotags": ["42.068", "-84.237"],
                "country": "USA",
                "city": "Brooklyn, MI",
                "length": str(int(round(lap_m))),
                "width": "12",
                "pitboxes": 20,
                "year": 2026,
            },
            indent=2,
        )
        + "\n"
    )

    render_validation(cx, cz, lon, lat, cone_pts, ix0, iy0, sat, lap_m, road_half, left_edge, right_edge, spot_xy)
    make_previews(cx, cz, en_xy, cone_pts, lap_m, ix0, iy0, ix1, iy1, crop, lon, lat, left_edge, right_edge)

    # Validate kn5
    sys.path.insert(0, str(TOOLS))
    import kn5 as kn5_reader

    ver, tex, mats_r, meshes = kn5_reader.parse(str(kn5_path))
    print(f"KN5 OK v{ver}: {len(tex)} textures, {len(mats_r)} mats, {len(meshes)} meshes, lap={lap_m:.1f}m")
    meta = OUT / "build_meta.json"
    line = LineString(en_xy)
    cone_d = [line.distance(Point(*ll2en(lo, la))) for lo, la, _ in cone_pts]
    on_line = sum(1 for d in cone_d if d < 1.8)
    edge = sum(1 for d in cone_d if 2.8 <= d <= 11.0)
    near_start = sum(
        1
        for lo, la, _ in cone_pts
        if min(math.hypot(ll2ac(lo, la)[0] - sx, ll2ac(lo, la)[1] - sz) for sx, sz in spot_xy) < 6.0
    )
    meta.write_text(
        json.dumps(
            {
                "lap_m": lap_m,
                "meshes": len(meshes),
                "cones_orange": sum(1 for _lo, _la, k in cone_pts if k == "ConeOrange"),
                "cones_blue": sum(1 for _lo, _la, k in cone_pts if k == "ConeBlue"),
                "cone_on_line_m": on_line,
                "cone_edge_m": edge,
                "cones_within_spot_clear_m": near_start,
                "spot_clear_m": 6.0,
                "cone_mesh_visual_only": True,
            },
            indent=2,
        )
    )


def _ac_plot(
    da: ImageDraw.ImageDraw,
    x0,
    z0,
    x1,
    z1,
    cx,
    cz,
    left_edge,
    right_edge,
    cone_pts,
    spot_xy=None,
    plot_w: float = 820,
    plot_h: float = 620,
    mx: float = 40,
    my: float = 660,
):
    def tp(x, z):
        return (
            int(mx + (x - x0) / (x1 - x0) * plot_w),
            int(my - (z - z0) / (z1 - z0) * plot_h),
        )

    le = [tp(x, z) for x, z in left_edge]
    ri = [tp(x, z) for x, z in right_edge]
    ctr = [tp(x, z) for x, z in zip(cx, cz)]
    if len(le) > 2:
        da.line(le + [le[0]], fill=(100, 220, 255), width=2)
        da.line(ri + [ri[0]], fill=(100, 255, 180), width=2)
    da.line(ctr + [ctr[0]], fill=(255, 70, 60), width=2)
    for lo, la, kind in cone_pts:
        x, z = ll2ac(lo, la)
        px, py = tp(x, z)
        col = (240, 150, 40) if kind == "ConeOrange" else (50, 110, 230)
        da.ellipse([px - 3, py - 3, px + 3, py + 3], fill=col, outline=(30, 30, 30))
    if spot_xy:
        for sx, sz in spot_xy[:4]:
            px, py = tp(sx, sz)
            da.rectangle([px - 5, py - 5, px + 5, py + 5], outline=(255, 255, 255), width=2)


def render_validation(
    cx, cz, lon, lat, cone_pts, ix0, iy0, sat, lap_m, road_half, left_edge, right_edge, spot_xy
):
    """Top-down QA: road edges, centreline, cones; includes zoom inset."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pad = 8
    sat_draw = Image.fromarray(sat).copy()
    d = ImageDraw.Draw(sat_draw, "RGBA")

    def draw_en_edges(line_pts, col):
        for i in range(len(line_pts) - 1):
            p1 = en2z18px(line_pts[i][0], line_pts[i][1])
            p2 = en2z18px(line_pts[i + 1][0], line_pts[i + 1][1])
            d.line([p1, p2], fill=col, width=2)

    line_en = np.array([ll2en(lo, la) for lo, la in zip(lon, lat)])
    le_en, ri_en = [], []
    for i in range(len(lon)):
        if i == 0:
            de, dn = line_en[1] - line_en[0]
        elif i == len(lon) - 1:
            de, dn = line_en[-1] - line_en[-2]
        else:
            de, dn = line_en[i + 1] - line_en[i - 1]
        L = math.hypot(de, dn) + 1e-9
        nx, ny = -dn / L, de / L
        le_en.append((line_en[i, 0] + road_half * nx, line_en[i, 1] + road_half * ny))
        ri_en.append((line_en[i, 0] - road_half * nx, line_en[i, 1] - road_half * ny))

    draw_en_edges(le_en, (100, 220, 255, 200))
    draw_en_edges(ri_en, (100, 255, 160, 200))
    for i in range(len(lon) - 1):
        d.line(
            [en2z18px(*ll2en(lon[i], lat[i])), en2z18px(*ll2en(lon[i + 1], lat[i + 1]))],
            fill=(255, 60, 60, 240),
            width=2,
        )
    for lo, la, kind in cone_pts:
        px, py = en2z18px(*ll2en(lo, la))
        col = (255, 160, 40, 230) if kind == "ConeOrange" else (60, 130, 255, 230)
        d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=col)

    x0, z0, x1, z1 = float(cx.min() - pad), float(cz.min() - pad), float(cx.max() + pad), float(cz.max() + pad)
    ac_draw = Image.new("RGB", (900, 700), (25, 28, 22))
    da = ImageDraw.Draw(ac_draw)
    _ac_plot(da, x0, z0, x1, z1, cx, cz, left_edge, right_edge, cone_pts)
    da.text(
        (12, 8),
        f"AC: cyan/green = road edges ({road_half:.0f} m half-width), red = centreline, orange/blue = cones",
        fill=(230, 230, 230),
    )
    da.text((12, 28), f"Lap ~{lap_m:.0f} m | cones placed: {len(cone_pts)}", fill=(200, 200, 200))
    # crop satellite panel around track
    px_c, py_c = en2z18px(*ll2en(lon.mean(), lat.mean()))
    cw, ch = 700, 520
    crop2 = sat_draw.crop(
        (
            max(0, int(px_c - cw // 2)),
            max(0, int(py_c - ch // 2)),
            min(sat_draw.width, int(px_c + cw // 2)),
            min(sat_draw.height, int(py_c + ch // 2)),
        )
    )
    # Zoom on back-straight chicane (~35% lap)
    zi = max(1, int(len(cx) * 0.35))
    zr = 45
    x0z, x1z = float(cx[zi] - zr), float(cx[zi] + zr)
    z0z, z1z = float(cz[zi] - zr), float(cz[zi] + zr)
    zoom = Image.new("RGB", (420, 320), (20, 22, 18))
    dz = ImageDraw.Draw(zoom)
    _ac_plot(dz, x0z, z0z, x1z, z1z, cx, cz, left_edge, right_edge, cone_pts, plot_w=400, plot_h=285, mx=10, my=305)
    dz.text((8, 6), "ZOOM: edges / centreline / cones", fill=(240, 240, 220))
    dz.text((8, 22), "(chicane sector)", fill=(180, 180, 180))

    combo = Image.new("RGB", (1620, 900))
    combo.paste(crop2.resize((900, 700)), (0, 10))
    combo.paste(ac_draw, (710, 10))
    combo.paste(zoom, (710, 560))
    combo.save(ARTIFACTS / "check_satellite_and_ac_world.png")
    off = Image.open(ROOT / "official_ev_endurance_2026.png").convert("RGBA")
    ow, oh = off.size
    off_small = off.resize((1100, int(1100 * oh / ow)))
    ov = Image.new("RGB", (1100, 720), (240, 240, 235))
    ov.paste(off_small.crop((0, 0, 1100, 720)), (0, 0))
    do = ImageDraw.Draw(ov)
    sys.path.insert(0, str(TOOLS))
    from evov import ev2z18, evcoef

    CO = evcoef(0, 113, 1.0, 13.5)

    def ll_to_uv(lo, la):
        # binary search on map pixels (map is small)
        best, bestd = (800, 95), 1e9
        for u in range(0, 1600, 4):
            for v in range(0, 191, 4):
                x, y = ev2z18(u, v, CO)
                zlo, zla = gpx2ll(69727 * 256 + x, 97241 * 256 + y, 18)
                d = (zlo - lo) ** 2 + (zla - la) ** 2
                if d < bestd:
                    bestd, best = d, (u, v)
        su = min(1599, best[0] + 2)
        sv = min(190, best[1] + 2)
        for u in range(max(0, best[0] - 3), su):
            for v in range(max(0, best[1] - 3), sv):
                x, y = ev2z18(u, v, CO)
                zlo, zla = gpx2ll(69727 * 256 + x, 97241 * 256 + y, 18)
                d = (zlo - lo) ** 2 + (zla - la) ** 2
                if d < bestd:
                    bestd, best = d, (u, v)
        sx = int(best[0] * 1100 / 1600)
        sy = int(best[1] * 720 / 191)
        return sx, sy

    pts_uv = [ll_to_uv(lo, la) for lo, la in zip(lon[::2], lat[::2])]
    if len(pts_uv) > 2:
        do.line(pts_uv + [pts_uv[0]], fill=(255, 40, 40), width=3)
    for lo, la, kind in cone_pts:
        sx, sy = ll_to_uv(lo, la)
        col = (220, 120, 20) if kind == "ConeOrange" else (30, 80, 200)
        do.ellipse([sx - 2, sy - 2, sx + 2, sy + 2], fill=col)
    do.text((10, 10), "Red = centreline; cones = filtered edge/slalom only (not map racing line)", fill=(80, 20, 20))
    ov.save(ARTIFACTS / "check_road_on_official_map.png")
    leg = Image.new("RGB", (1620, 140), (18, 18, 18))
    dl = ImageDraw.Draw(leg)
    dl.text(
        (12, 12),
        "Cone QA: orange should sit between cyan/green edges and NOT on red centreline except chicane/gate clusters.",
        fill=(230, 230, 230),
    )
    dl.text(
        (12, 38),
        "CONES_* meshes have no 1ROAD_/1CONE_ physics prefix (visual only). Spots cleared ±4.5 m in filter.",
        fill=(190, 190, 190),
    )
    stacked = Image.new("RGB", (1620, 1040))
    stacked.paste(combo, (0, 0))
    stacked.paste(leg, (0, 900))
    stacked.save(ARTIFACTS / "check_track_validation.png")


def gpx2ll(x, y, z):
    n = 256 * 2**z
    lon = x / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
    return float(lon), float(lat)


def make_previews(cx, cz, en_xy, cones, lap_m, ix0, iy0, ix1, iy1, crop, lon, lat, left_edge, right_edge):
    ui = OUT / "ui"
    W, H = 512, 512
    pad = 40
    x0, x1 = cx.min() - pad, cx.max() + pad
    z0, z1 = cz.min() - pad, cz.max() + pad

    def to_px(x, z):
        u = (x - x0) / (x1 - x0)
        v = (z - z0) / (z1 - z0)
        return int(u * (W - 40) + 20), int((1 - v) * (H - 40) + 20)

    for name in ("preview.png", "outline.png"):
        im = Image.new("RGB", (W, H), (30, 35, 28))
        d = ImageDraw.Draw(im)
        le = [to_px(x, z) for x, z in left_edge]
        ri = [to_px(x, z) for x, z in right_edge]
        pts = [to_px(x, z) for x, z in zip(cx, cz)]
        if len(le) > 2:
            d.line(le, fill=(80, 200, 240), width=1)
            d.line(ri, fill=(80, 240, 160), width=1)
        if len(pts) > 2:
            d.line(pts + [pts[0]], fill=(240, 70, 60), width=2)
        for lo, la, kind in cones[:: max(1, len(cones) // 400)]:
            x, z = ll2ac(lo, la)
            px, py = to_px(x, z)
            col = (240, 150, 40) if kind == "ConeOrange" else (50, 110, 230)
            d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=col)
        d.text((12, 8), f"EV Endurance ~{lap_m/1000:.2f} km", fill=(220, 220, 220))
        im.save(ui / name)

    # Map top-down for AC UI
    mh = min(1024, crop.shape[0])
    mw = min(1024, crop.shape[1])
    map_im = Image.fromarray(crop).resize((mw, mh))
    d = ImageDraw.Draw(map_im)
    for i in range(len(lon) - 1):
        px0, py0 = en2z18px(*ll2en(lon[i], lat[i]))
        px1, py1 = en2z18px(*ll2en(lon[i + 1], lat[i + 1]))
        d.line(
            [(px0 - ix0, py0 - iy0), (px1 - ix0, py1 - iy0)],
            fill=(255, 50, 50),
            width=2,
        )
    map_im.save(OUT / "map.png")
    map_im.copy().save(ui / "map.png")


if __name__ == "__main__":
    main()