#!/usr/bin/env python3
"""Byte-level kn5 audit aligned with AcTools.Kn5File (v5) reader."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

REQ_PROPS = {"ksAmbient", "ksDiffuse", "ksSpecular", "ksSpecularEXP", "ksEmissive", "ksAlphaRef"}
VALID_SHADERS = {
    "ksPerPixel",
    "ksPerPixelMultiMap",
    "ksPerPixelAT",
    "ksPerPixelNM",
    "ksPerPixelAlpha",
}


class R:
    def __init__(self, b: bytes):
        self.b = b
        self.p = 0

    def i(self) -> int:
        v = struct.unpack_from("<i", self.b, self.p)[0]
        self.p += 4
        return v

    def u(self) -> int:
        v = struct.unpack_from("<I", self.b, self.p)[0]
        self.p += 4
        return v

    def u8(self) -> int:
        v = self.b[self.p]
        self.p += 1
        return v

    def u16(self) -> int:
        v = struct.unpack_from("<H", self.b, self.p)[0]
        self.p += 2
        return v

    def f(self) -> float:
        v = struct.unpack_from("<f", self.b, self.p)[0]
        self.p += 4
        return v

    def bool(self) -> bool:
        return self.u8() != 0

    def s(self) -> str:
        n = self.i()
        raw = self.b[self.p : self.p + n]
        self.p += n
        return raw.decode("latin1", errors="replace")

    def vec2(self):
        return self.f(), self.f()

    def vec3(self):
        return self.f(), self.f(), self.f()

    def vec4(self):
        return self.f(), self.f(), self.f(), self.f()

    def mat4(self):
        return struct.unpack_from("<16f", self.b, self.p)


def validate_dds(data: bytes, name: str, errors: list, warns: list) -> None:
    if len(data) < 128 or data[:4] != b"DDS ":
        errors.append(f"texture {name}: not a DDS file (missing DDS header)")
        return
    hsize, flags, height, width = struct.unpack_from("<IIII", data, 4)
    if hsize != 124:
        warns.append(f"texture {name}: DDS header size {hsize} (expected 124)")
    if width < 1 or height < 1:
        errors.append(f"texture {name}: invalid dimensions {width}x{height}")


def parse_node(r: R, errors: list, stats: dict, depth: int = 0) -> None:
    if depth > 64:
        errors.append("node tree too deep")
        return
    klass = r.i()
    name = r.s()
    nchild = r.i()
    active = r.bool()
    if klass == 1:
        r.p += 64
        stats["base"] += 1
        if name.startswith("AC_"):
            stats["spots"] += 1
    elif klass == 2:
        cast = r.bool()
        vis = r.bool()
        trans = r.bool()
        vc = r.u()
        for _ in range(vc):
            pos = r.vec3()
            nrm = r.vec3()
            tex = r.vec2()
            tan = r.vec3()
            if not stats.get("checked_vert"):
                stats["checked_vert"] = True
                # AcTools order: pos, normal, tex, tangent
        ic = r.u()
        for _ in range(ic):
            r.u16()
        mid = r.u()
        layer = r.u()
        r.f()
        r.f()
        bc = r.vec3()
        br = r.f()
        rend = r.bool()
        stats["mesh"] += 1
        if name.startswith("1"):
            stats["phys"] += 1
        if br <= 0:
            errors.append(f"mesh {name}: bounding sphere radius <= 0")
        if vc > 65535:
            errors.append(f"mesh {name}: too many vertices")
    elif klass == 3:
        errors.append(f"unexpected skinned mesh {name}")
        return
    else:
        errors.append(f"unknown node class {klass} ({name})")
        return
    for _ in range(nchild):
        parse_node(r, errors, stats, depth + 1)


def validate(path: Path) -> int:
    data = path.read_bytes()
    errors: list[str] = []
    warns: list[str] = []
    stats = {"base": 0, "mesh": 0, "phys": 0, "spots": 0}

    if data[:6] != b"sc6969":
        errors.append("missing sc6969 magic")
        return report(errors, warns, stats)

    r = R(data)
    r.p = 6
    ver = r.i()
    if ver != 5:
        warns.append(f"kn5 version {ver} (AcTools Kn5ActualVersion=5)")
    if ver > 5:
        r.i()

    nt = r.i()
    for _ in range(nt):
        active = r.i()
        tname = r.s()
        tlen = r.i()
        blob = data[r.p : r.p + tlen]
        r.p += tlen
        validate_dds(blob, tname, errors, warns)
        if active != 1:
            warns.append(f"texture {tname}: active flag {active}")

    nm = r.i()
    for mi in range(nm):
        mname = r.s()
        shader = r.s()
        blend = r.u8()
        alpha = r.bool()
        depth = r.i()
        np_ = r.i()
        props = set()
        for _ in range(np_):
            pname = r.s()
            props.add(pname)
            r.f()
            r.vec2()
            r.vec3()
            r.vec4()
        ntex = r.i()
        for _ in range(ntex):
            r.s()
            r.i()
            r.s()
        if shader not in VALID_SHADERS and not shader.startswith("ksPerPixel"):
            warns.append(f"material {mname}: shader {shader}")
        missing = REQ_PROPS - props
        if missing:
            warns.append(f"material {mname}: missing props {sorted(missing)}")

    parse_node(r, errors, stats)
    if r.p != len(data):
        warns.append(f"trailing bytes: {len(data) - r.p}")

    if stats["phys"] < 2:
        errors.append("expected at least 1ROAD + 1GRASS physics meshes")
    if stats["spots"] < 40:
        errors.append(f"too few AC spot nodes ({stats['spots']})")

    return report(errors, warns, stats)


def report(errors, warns, stats) -> int:
    print("stats:", stats)
    for w in warns:
        print("WARN:", w)
    for e in errors:
        print("ERROR:", e)
    if errors:
        return 1
    print("OK: kn5 passes AcTools-aligned audit")
    return 0


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "assetto_corsa/fsae_michigan_2026_ev/fsae_michigan_2026_ev.kn5")
    raise SystemExit(validate(p))