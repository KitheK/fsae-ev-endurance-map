"""Write Assetto Corsa .kn5 v5 (AcTools.Kn5File compatible)."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field

import numpy as np

KN5_MAX_VERTICES = 65535

_DEFAULT_PBR = [
    ("ksAmbient", 0.2),
    ("ksDiffuse", 0.75),
    ("ksSpecular", 0.15),
    ("ksSpecularEXP", 12.0),
    ("ksEmissive", 0.0),
    ("ksAlphaRef", 0.0),
]


@dataclass
class Texture:
    name: str
    data: bytes


@dataclass
class Material:
    name: str
    shader: str = "ksPerPixelMultiMap"
    properties: list[tuple[str, float]] | None = None
    textures: list[tuple[str, int, str]] | None = None
    alpha_tested: bool = False
    blend_mode: int = 0
    depth_mode: int = 0

    def __post_init__(self):
        if self.properties is None:
            self.properties = list(_DEFAULT_PBR)
        else:
            names = {p[0] for p in self.properties}
            for k, v in _DEFAULT_PBR:
                if k not in names:
                    self.properties.append((k, v))
        if self.textures is None:
            self.textures = []


@dataclass
class Mesh:
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    material: int
    cast_shadows: bool = True
    renderable: bool = True
    is_visible: bool = True
    is_transparent: bool = False


@dataclass
class Node:
    name: str
    mesh: Mesh | None = None
    transform: np.ndarray | None = None
    children: list[Node] = field(default_factory=list)
    active: bool = True


def _str(buf: io.BytesIO, s: str) -> None:
    b = s.encode("latin1", errors="replace")
    buf.write(struct.pack("<i", len(b)))
    buf.write(b)


def _write_shader_prop(buf: io.BytesIO, name: str, value: float) -> None:
    _str(buf, name)
    buf.write(struct.pack("<f", value))
    buf.write(struct.pack("<2f", 0.0, 0.0))
    buf.write(struct.pack("<3f", 0.0, 0.0, 0.0))
    buf.write(struct.pack("<4f", 0.0, 0.0, 0.0, 0.0))


def _write_node(buf: io.BytesIO, n: Node) -> None:
    buf.write(struct.pack("<i", 2 if n.mesh is not None else 1))
    _str(buf, n.name)
    buf.write(struct.pack("<i", len(n.children)))
    buf.write(struct.pack("<B", 1 if n.active else 0))
    M = n.transform if n.transform is not None else np.eye(4, dtype=np.float32)
    if n.mesh is None:
        buf.write(M.astype("<f4").tobytes())
    else:
        m = n.mesh
        n_verts = len(m.positions)
        if n_verts > KN5_MAX_VERTICES:
            raise ValueError(f"{n.name}: {n_verts} vertices (max {KN5_MAX_VERTICES})")
        buf.write(
            struct.pack(
                "<BBB",
                1 if m.cast_shadows else 0,
                1 if m.is_visible else 0,
                1 if m.is_transparent else 0,
            )
        )
        tangents = np.zeros((n_verts, 3), np.float32)
        tangents[:, 0] = 1.0
        # AcTools/Kn5 vertex order: Position, Normal, Tex, Tangent
        verts = np.hstack([m.positions, m.normals, m.uvs, tangents]).astype("<f4")
        buf.write(struct.pack("<I", n_verts))
        buf.write(verts.tobytes())
        idx = np.asarray(m.indices, dtype=np.uint16)
        buf.write(struct.pack("<I", len(idx)))
        buf.write(idx.tobytes())
        center = (m.positions.max(0) + m.positions.min(0)) / 2
        radius = float(np.linalg.norm(m.positions - center, axis=1).max()) + 0.05
        buf.write(struct.pack("<II", int(m.material), 0))
        buf.write(struct.pack("<ff", 0.0, 0.0))
        buf.write(struct.pack("<3ff", float(center[0]), float(center[1]), float(center[2]), radius))
        buf.write(struct.pack("<B", 1 if m.renderable else 0))
    for c in n.children:
        _write_node(buf, c)


def write_kn5(textures: list[Texture], materials: list[Material], root: Node, version: int = 5) -> bytes:
    buf = io.BytesIO()
    buf.write(b"sc6969")
    buf.write(struct.pack("<i", version))
    if version > 5:
        buf.write(struct.pack("<i", 0))
    buf.write(struct.pack("<i", len(textures)))
    for t in textures:
        buf.write(struct.pack("<i", 1))
        _str(buf, t.name)
        buf.write(struct.pack("<i", len(t.data)))
        buf.write(t.data)
    buf.write(struct.pack("<i", len(materials)))
    for mat in materials:
        _str(buf, mat.name)
        _str(buf, mat.shader)
        buf.write(struct.pack("<BB", mat.blend_mode, 1 if mat.alpha_tested else 0))
        buf.write(struct.pack("<i", mat.depth_mode))
        buf.write(struct.pack("<i", len(mat.properties)))
        for pname, pval in mat.properties:
            _write_shader_prop(buf, pname, pval)
        buf.write(struct.pack("<i", len(mat.textures)))
        for sampler, slot, tex in mat.textures:
            _str(buf, sampler)
            buf.write(struct.pack("<i", slot))
            _str(buf, tex)
    _write_node(buf, root)
    return buf.getvalue()


def transform_node(name: str, x: float, y: float, z: float, heading_rad: float) -> Node:
    """Y-up AC spawn node (row-major 4x4, translation in row 3)."""
    c, s = float(np.cos(heading_rad)), float(np.sin(heading_rad))
    M = np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [x, y, z, 1.0],
        ],
        dtype=np.float32,
    )
    return Node(name, transform=M, active=True)


# --- DDS BC1 (DXT1) ---


def _mips(img: np.ndarray, min_size: int = 1) -> list[np.ndarray]:
    levels = [img]
    while min(levels[-1].shape[:2]) > min_size:
        a = levels[-1].astype(np.float32)
        h, w = a.shape[:2]
        a = a[: h - h % 2, : w - w % 2]
        a = (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) / 4
        levels.append(np.clip(a + 0.5, 0, 255).astype(np.uint8))
    return levels


def _header(w: int, h: int, mips: int, fourcc: bytes, linear_size: int) -> bytes:
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000
    pf = struct.pack("<2I4s5I", 32, 0x4, fourcc, 0, 0, 0, 0, 0)
    head = struct.pack("<4s7I44x", b"DDS ", 124, flags, h, w, linear_size, 0, mips)
    caps = struct.pack("<5I", 0x1000 | 0x400000 | 0x8, 0, 0, 0, 0)
    return head + pf + caps


def _to565(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.uint32)
    return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)


def _from565(v: np.ndarray) -> np.ndarray:
    r = ((v >> 11) & 31) * 255 // 31
    g = ((v >> 5) & 63) * 255 // 63
    b = (v & 31) * 255 // 31
    return np.stack([r, g, b], -1).astype(np.float32)


def _bc1_blocks(blocks: np.ndarray) -> bytes:
    mean = blocks.mean(1, keepdims=True)
    axis = blocks.max(1) - blocks.min(1)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-6
    proj = np.einsum("bpc,bc->bp", blocks - mean, axis)
    c0 = mean[:, 0] + axis * proj.max(1, keepdims=True)
    c1 = mean[:, 0] + axis * proj.min(1, keepdims=True)
    e0, e1 = _to565(np.clip(c0, 0, 255)), _to565(np.clip(c1, 0, 255))
    swap = e0 < e1
    e0, e1 = np.where(swap, e1, e0), np.where(swap, e0, e1)
    p0, p1 = _from565(e0), _from565(e1)
    palette = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], 1)
    dist = ((blocks[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(-1)
    idx = dist.argmin(-1).astype(np.uint32)
    idx[e0 == e1] = 0
    bits = np.zeros(len(blocks), np.uint32)
    for p in range(16):
        bits |= idx[:, p] << (2 * p)
    out = np.zeros((len(blocks), 2), np.uint32)
    out[:, 0] = e0 | (e1 << 16)
    out[:, 1] = bits
    return out.astype("<u4").tobytes()


def _bc1(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[:2]
    ph, pw = (h + 3) // 4 * 4, (w + 3) // 4 * 4
    img = np.pad(rgb, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    blocks = img.reshape(ph // 4, 4, pw // 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 3)
    step = 65536
    return b"".join(_bc1_blocks(blocks[i : i + step].astype(np.float32)) for i in range(0, len(blocks), step))


def dds_bc1(rgb: np.ndarray) -> bytes:
    levels = _mips(rgb[..., :3])
    body = b"".join(_bc1(l) for l in levels)
    w, h = rgb.shape[1], rgb.shape[0]
    return _header(w, h, len(levels), b"DXT1", max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 8) + body