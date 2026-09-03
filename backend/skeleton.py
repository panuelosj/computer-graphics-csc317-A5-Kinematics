"""Skeleton / Bone data structures and loaders (non-student infrastructure).

A :class:`Skeleton` is simply a Python ``list`` of
:class:`Bone` objects; each bone stores an index to its parent (``-1`` for the
root), the rigid ``T_rest`` transformation mapping the *canonical bone* (lying
along the x-axis with its tail at the origin and tip at ``(length, 0, 0)``) to
its *rest bone* placement inside the model, the bone ``length``, the current
pose as three ``xzx`` extrinsic Euler angles (degrees), per-bone joint limits
``xzx_min`` / ``xzx_max`` (degrees), and a ``weight_index`` into the columns of
the linear-blend-skinning weight matrix (``-1`` for no weights).

Transforms are 4x4 homogeneous ``numpy`` arrays.

The student-facing ``src/`` functions never need to import this module: they
just receive ``Bone`` objects (or lists of them) and read their attributes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

Skeleton = List["Bone"]


@dataclass
class Bone:
    """One rigid bone in a skeleton tree.

    Attributes
    ----------
    parent_index : int
        Index into the skeleton list of this bone's parent (``-1`` = root).
    weight_index : int
        Column index into the LBS weight matrix (``-1`` = no weights).
    T_rest : (4, 4) ndarray
        Rigid transform mapping the canonical bone to its rest placement.
    length : float
        Length of the bone (canonical tip is at ``(length, 0, 0)``).
    xzx : (3,) ndarray
        Current pose: extrinsic Euler angles about x, then z, then x (degrees).
    xzx_min, xzx_max : (3,) ndarray
        Lower / upper joint limits for ``xzx`` (degrees).
    """

    parent_index: int
    weight_index: int
    T_rest: np.ndarray
    length: float
    xzx: np.ndarray = field(default_factory=lambda: np.zeros(3))
    xzx_min: np.ndarray = field(default_factory=lambda: np.full(3, -np.inf))
    xzx_max: np.ndarray = field(default_factory=lambda: np.full(3, np.inf))

    def __post_init__(self):
        self.T_rest = np.asarray(self.T_rest, dtype=float).reshape(4, 4)
        self.length = float(self.length)
        self.xzx = np.asarray(self.xzx, dtype=float).reshape(3)
        self.xzx_min = np.asarray(self.xzx_min, dtype=float).reshape(3)
        self.xzx_max = np.asarray(self.xzx_max, dtype=float).reshape(3)


def translation(t) -> np.ndarray:
    """4x4 homogeneous translation by the 3-vector ``t``."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def apply_transform(T: np.ndarray, p) -> np.ndarray:
    """Apply 4x4 homogeneous transform ``T`` to a 3-point ``p``."""
    p = np.asarray(p, dtype=float).reshape(3)
    return (T[:3, :3] @ p) + T[:3, 3]


# --- JSON (de)serialisation -------------------------------------------------

def save_skeleton(path: str, skeleton: Skeleton) -> None:
    """Write a skeleton to a JSON file."""
    bones = []
    for b in skeleton:
        bones.append(
            {
                "parent_index": int(b.parent_index),
                "weight_index": int(b.weight_index),
                "T_rest": b.T_rest.tolist(),
                "length": float(b.length),
                "xzx": b.xzx.tolist(),
                "xzx_min": [None if not np.isfinite(x) else x for x in b.xzx_min],
                "xzx_max": [None if not np.isfinite(x) else x for x in b.xzx_max],
            }
        )
    with open(path, "w") as f:
        json.dump({"bones": bones}, f, indent=2)


def _to_limit(values, fill):
    out = []
    for v in values:
        out.append(fill if v is None else float(v))
    return np.array(out, dtype=float)


def load_skeleton(path: str) -> Skeleton:
    """Read a skeleton from a JSON file written by :func:`save_skeleton`."""
    with open(path) as f:
        data = json.load(f)
    skeleton: Skeleton = []
    for bd in data["bones"]:
        skeleton.append(
            Bone(
                parent_index=int(bd["parent_index"]),
                weight_index=int(bd["weight_index"]),
                T_rest=np.array(bd["T_rest"], dtype=float),
                length=float(bd["length"]),
                xzx=np.array(bd.get("xzx", [0, 0, 0]), dtype=float),
                xzx_min=_to_limit(bd.get("xzx_min", [None, None, None]), -np.inf),
                xzx_max=_to_limit(bd.get("xzx_max", [None, None, None]), np.inf),
            )
        )
    return skeleton


def load_keyframes(path: str):
    """Read a keyframe animation file.

    Returns a list of ``(time, angles)`` pairs where ``angles`` is an
    ``(nbones, 3)`` array of Euler angles for that key time.
    """
    with open(path) as f:
        data = json.load(f)
    frames = []
    for kf in data["keyframes"]:
        frames.append((float(kf["t"]), np.array(kf["angles"], dtype=float)))
    return frames


def bone_keyframes(frames, b: int):
    """Project per-skeleton keyframes onto a single bone ``b``.

    Returns a list of ``(time, vec3)`` suitable for :func:`interpolate_keyframes`.
    """
    return [(t, angles[b].copy()) for (t, angles) in frames]


def interpolate_keyframes(keyframes, t):
    """Catmull-Rom interpolation of vec3 keyframes at time ``t``.

    Backend code: this is what smooths the sample animations for the demo.
    **Implementing this is Assignment 4's task** (`catmull_rom_interpolation`);
    it lives here so this assignment's viewer works on its own, without
    depending on your A4 code. It is kept identical to A4's reference, so the
    motion you see here is exactly what your A4 implementation should produce.

    Standard (uniform) Catmull-Rom: each key's tangent is the centred
    difference across its neighbours, ``m_i = tau * (P_{i+1} - P_{i-1})`` with
    ``tau = 0.5``; at the ends the missing neighbour is the endpoint itself.
    That makes the curve C1 at interior keys -- position and velocity are both
    continuous. Outside the range the nearest endpoint is returned.
    """
    if len(keyframes) == 0:
        return np.zeros(3)

    k1 = 0
    for k in range(len(keyframes)):
        if keyframes[k][0] < t:
            k1 = k

    if keyframes[k1][0] > t:
        return np.array(keyframes[k1][1], dtype=float)
    if k1 == len(keyframes) - 1:
        return np.array(keyframes[k1][1], dtype=float)

    k2 = k1 + 1
    s = (t - keyframes[k1][0]) / (keyframes[k2][0] - keyframes[k1][0])
    P1 = np.asarray(keyframes[k1][1], dtype=float)
    P2 = np.asarray(keyframes[k2][1], dtype=float)

    tau = 0.5
    P0 = np.asarray(keyframes[k1 - 1][1], dtype=float) if k1 > 0 else P1
    P3 = (np.asarray(keyframes[k2 + 1][1], dtype=float)
          if k2 < len(keyframes) - 1 else P2)

    D1 = tau * (P2 - P0)
    D2 = tau * (P3 - P1)

    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * P1 + h10 * D1 + h01 * P2 + h11 * D2


def load_rig(data_dir: str):
    """Load the full sample rig: ``(skeleton, V, F, W)``.

    ``V`` is the rest mesh vertices ``(n, 3)``, ``F`` the triangle faces
    ``(m, 3)`` and ``W`` the LBS weights ``(n, nbones)``.
    """
    from cgcommon.objio import read_obj, triangulate

    skeleton = load_skeleton(os.path.join(data_dir, "rig.json"))
    V, F = read_obj(os.path.join(data_dir, "mesh.obj"))
    if not isinstance(F, np.ndarray) or F.shape[1] != 3:
        F = triangulate(F)
    W = np.load(os.path.join(data_dir, "weights.npy"))
    return skeleton, V, F, W
