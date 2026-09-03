#!/usr/bin/env python3
"""Assignment 5 (Kinematics) demo backend.

Drives your ``src/`` functions end-to-end on the sample 3-bone rig in ``data/``.

Interactive (default):

    python main.py

  * **IK mode**: a teal dot marks the end-effector target. **Click and drag
    it** — the arm re-solves every frame with projected gradient descent.
  * **FK mode**: keyframe animation, interpolated with your Catmull-Rom code.
  * Keys (also available as GUI buttons):
      [space]  play / pause the FK animation
      I        toggle IK mode <-> FK mode
      R        reset all bone angles to zero (and the IK target to the tip)

Headless (used by CI / grading; no window needed):

    python main.py --mode fk --headless     # write posed OBJ frames + joints
    python main.py --mode ik --headless     # solve fixed target, report error
    python main.py --mode fk --anim curl    # choose the 'curl' keyframes
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# `cgcommon` (shared helpers) normally sits next to this file. If several
# assignments are checked out together it lives two levels up instead.
if not os.path.isdir(os.path.join(HERE, "cgcommon")):
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from backend.skeleton import (
    load_rig, load_keyframes, bone_keyframes, interpolate_keyframes,
    apply_transform,
)

from src.forward_kinematics import forward_kinematics
from src.transformed_tips import transformed_tips
from src.linear_blend_skinning import linear_blend_skinning
from src.copy_skeleton_at import copy_skeleton_at
from src.end_effectors_objective_and_gradient import (
    end_effectors_objective_and_gradient,
)
from src.projected_gradient_descent import projected_gradient_descent

DATA = os.path.join(HERE, "data")

# The rigged sample models. Each is a directory under data/ holding mesh.obj,
# weights.npy, rig.json and -- where the model defines one -- keyframes.json
# (an FK animation) and ik.json (which bone tips the IK solver may pin).
MODELS = ["robot-arm", "knight", "ikea-lamp", "beast", "chimpanzee-hand"]


def resolve_model(name, anim):
    """Return ``(data_dir, keyframe_path, end_effectors_or_None)``.

    Without ``--model`` this is the small built-in rig and its wave/curl
    animations, which is what the unit checks use.
    """
    if not name:
        return DATA, os.path.join(DATA, f"{anim}.json"), None
    d = os.path.join(DATA, name)
    if not os.path.isdir(d):
        raise SystemExit(f"unknown model {name!r} (expected one of {MODELS})")
    kf = os.path.join(d, "keyframes.json")
    ik = os.path.join(d, "ik.json")
    ee = None
    if os.path.isfile(ik):
        with open(ik) as f:
            ee = np.array(json.load(f)["end_effectors"], dtype=int)
    return d, (kf if os.path.isfile(kf) else None), ee
OUT = os.path.join(HERE, "out")

# One IK solve per rendered frame, warm-started from the previous solution —
# an "iterate while you drag" behaviour.
IK_ITERS_PER_FRAME = 20


def skeleton_nodes_edges(skel):
    """Return (nodes, edges) describing the posed skeleton for visualisation."""
    T = forward_kinematics(skel)
    nodes = []
    edges = []
    for i, bone in enumerate(skel):
        rest_tail = bone.T_rest[:3, 3]
        rest_tip = apply_transform(bone.T_rest, [bone.length, 0, 0])
        pose_tail = apply_transform(T[i], rest_tail)
        pose_tip = apply_transform(T[i], rest_tip)
        a = len(nodes)
        nodes.append(pose_tail)
        nodes.append(pose_tip)
        edges.append([a, a + 1])
    return np.array(nodes), np.array(edges, dtype=int)


def pose_at_time(skel, frames, t):
    """Set the skeleton's joint angles from interpolated keyframes at time t."""
    angles = np.zeros((len(skel), 3))
    for b in range(len(skel)):
        kf = bone_keyframes(frames, b)
        angles[b] = interpolate_keyframes(kf, t)
    return copy_skeleton_at(skel, angles.reshape(-1))


def bone_colors(W):
    """Per-vertex colours from the dominant skinning weight."""
    palette = np.array([
        [0.80, 0.36, 0.36], [0.36, 0.65, 0.80], [0.44, 0.76, 0.44],
        [0.83, 0.71, 0.38], [0.66, 0.48, 0.77], [0.85, 0.55, 0.35],
        [0.47, 0.72, 0.69], [0.72, 0.51, 0.58],
    ])
    dom = np.argmax(W, axis=1) % len(palette)
    return palette[dom]


def write_obj(path, V, F):
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in F:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# --- headless modes (used by CI/grading) -------------------------------------

def run_fk_headless(skel, V, F, W, frames, anim):
    os.makedirs(OUT, exist_ok=True)
    t0, t1 = frames[0][0], frames[-1][0]
    n_frames = 6
    times = np.linspace(t0, t1, n_frames)
    joints_all = []
    for k, t in enumerate(times):
        posed = pose_at_time(skel, frames, float(t))
        T = forward_kinematics(posed)
        U = linear_blend_skinning(V, posed, T, W)
        write_obj(os.path.join(OUT, f"{anim}_frame{k:02d}.obj"), U, F)
        nodes, _ = skeleton_nodes_edges(posed)
        joints_all.append(nodes)
        print(f"  t={t:5.2f}  wrote out/{anim}_frame{k:02d}.obj  "
              f"(mesh bbox x:[{U[:,0].min():.2f},{U[:,0].max():.2f}])")
    np.save(os.path.join(OUT, f"{anim}_joints.npy"), np.array(joints_all))
    print(f"  wrote out/{anim}_joints.npy  {np.array(joints_all).shape}")
    print(f"\nForward-kinematics animation '{anim}' written to out/.")


def run_ik_headless(skel, V, F, W, end_effectors=None):
    os.makedirs(OUT, exist_ok=True)
    nbones = len(skel)
    b = (np.asarray(end_effectors, dtype=int) if end_effectors is not None
         else np.array([nbones - 1], dtype=int))
    # A fixed, reachable target derived from a known bent pose, clamped to the
    # joint limits so locked bones (e.g. the zero-length root) stay put.
    target_pose = np.zeros((nbones, 3))
    target_pose[:, 1] = ([35.0, 30.0, 25.0] * nbones)[:nbones]
    for i, bone in enumerate(skel):
        target_pose[i] = np.clip(target_pose[i], bone.xzx_min, bone.xzx_max)
    target_skel = copy_skeleton_at(skel, target_pose.reshape(-1))
    xb0 = transformed_tips(target_skel, b)

    f, grad_f, proj_z = end_effectors_objective_and_gradient(skel, b, xb0)
    z0 = np.zeros(nbones * 3)
    print(f"  target tip: {xb0}")
    print(f"  initial objective: {f(z0):.6e}")
    z = projected_gradient_descent(f, grad_f, proj_z, 200, z0)
    solved = copy_skeleton_at(skel, z)
    tip = transformed_tips(solved, b)
    err = float(np.linalg.norm(tip - xb0))
    print(f"  final objective:   {f(z):.6e}")
    print(f"  final tip:  {tip}")
    print(f"  final tip error: {err:.6e}")

    T = forward_kinematics(solved)
    U = linear_blend_skinning(V, solved, T, W)
    write_obj(os.path.join(OUT, "ik_solved.obj"), U, F)
    print("\nInverse kinematics solution written to out/ik_solved.obj.")


# --- interactive app (IK drag + FK animation) -------------------------------

def run_interactive(skel, V, F, W, frames, anim, start_mode,
                    max_frames=None, end_effectors=None):
    import polyscope as ps
    import polyscope.imgui as psim
    from cgcommon import viewer
    from cgcommon.interact import PointDragger, key_pressed

    viewer.ensure_init()

    nbones = len(skel)
    b = np.array([nbones - 1], dtype=int)
    rest_tip = transformed_tips(skel, b)
    t0, t1 = frames[0][0], frames[-1][0]

    state = {
        "ik": start_mode == "ik",
        "playing": True,
        "t": t0,
        "z": np.zeros(nbones * 3),
        "target": rest_tip.astype(float).copy(),
        "frame": 0,
    }

    T = forward_kinematics(skel)
    U = linear_blend_skinning(V, skel, T, W)
    mesh = ps.register_surface_mesh("skin", U, F, smooth_shade=True)
    mesh.add_color_quantity("bone", bone_colors(W), enabled=True)
    nodes, edges = skeleton_nodes_edges(skel)
    net = ps.register_curve_network("skeleton", nodes, edges, radius=0.012)

    bbd = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    target_pc = ps.register_point_cloud(
        "target", state["target"][None, :], radius=0.03)
    target_pc.set_color((0.1, 0.75, 0.75))          # teal, like the original
    tip_pc = ps.register_point_cloud("tip", rest_tip[None, :], radius=0.02)
    tip_pc.set_color((0.9, 0.45, 0.65))             # pink

    dragger = PointDragger(grab_radius=0.08 * bbd)

    def reset():
        state["z"] = np.zeros(nbones * 3)
        state["t"] = t0
        state["target"] = transformed_tips(skel, b).astype(float).copy()

    def redraw(posed):
        T = forward_kinematics(posed)
        U = linear_blend_skinning(V, posed, T, W)
        mesh.update_vertex_positions(U)
        nodes, _ = skeleton_nodes_edges(posed)
        net.update_node_positions(nodes)
        tip = transformed_tips(posed, b)
        tip_pc.update_point_positions(tip[None, :])
        target_pc.update_point_positions(state["target"][None, :])

    def callback():
        # --- keyboard ---
        if key_pressed("I"):
            state["ik"] = not state["ik"]
        if key_pressed("R"):
            reset()
        if key_pressed("Space"):
            state["playing"] = not state["playing"]

        # --- GUI ---
        _, state["ik"] = psim.Checkbox("IK mode (I)", state["ik"])
        psim.SameLine()
        if psim.Button("reset (R)"):
            reset()
        if state["ik"]:
            psim.TextUnformatted("drag the teal target with the mouse")
        else:
            _, state["playing"] = psim.Checkbox("play (space)", state["playing"])
            changed, state["t"] = psim.SliderFloat("time", state["t"], t0, t1)

        if state["ik"]:
            # mouse drag moves the effector target on the view plane
            event, _, pos = dragger.update(state["target"][None, :])
            if event == "drag":
                state["target"] = np.asarray(pos, dtype=float)
            # warm-started projected gradient descent toward the target
            f, grad_f, proj_z = end_effectors_objective_and_gradient(
                skel, b, state["target"])
            state["z"] = projected_gradient_descent(
                f, grad_f, proj_z, IK_ITERS_PER_FRAME, state["z"])
            posed = copy_skeleton_at(skel, state["z"])
        else:
            if state["playing"]:
                state["t"] += 0.02
                if state["t"] > t1:
                    state["t"] = t0
            posed = pose_at_time(skel, frames, state["t"])
            state["z"] = np.concatenate([bn.xzx for bn in posed])

        redraw(posed)
        state["frame"] += 1

    ps.set_user_callback(callback)
    print("Interactive kinematics: drag the teal IK target; keys: "
          "[space] play/pause FK, I toggle IK/FK, R reset.")
    if max_frames is not None:
        for _ in range(max_frames):
            ps.frame_tick()
    else:
        viewer.show()


def main():
    parser = argparse.ArgumentParser(description="A5 Kinematics demo backend.")
    parser.add_argument("--mode", choices=["fk", "ik"], default="ik",
                        help="headless mode / interactive starting mode")
    parser.add_argument("--anim", choices=["wave", "curl"], default="wave",
                        help="keyframe animation for FK (built-in rig only)")
    parser.add_argument("--model", default=None, metavar="NAME",
                        help="load one of the rigged sample models instead of "
                             "the built-in rig: " + ", ".join(MODELS))
    parser.add_argument("--headless", action="store_true",
                        help="run numerics and save outputs, no display")
    parser.add_argument("--frames", type=int, default=None,
                        help=argparse.SUPPRESS)  # debug: tick N frames and exit
    args = parser.parse_args()

    data_dir, kf_path, end_effectors = resolve_model(args.model, args.anim)
    skel, V, F, W = load_rig(data_dir)
    label = args.model or args.anim
    if kf_path is None:
        # A model with no FK animation of its own: hold it at rest, so --mode fk
        # still shows the rig rather than failing.
        frames = [(0.0, np.zeros((len(skel), 3))), (1.0, np.zeros((len(skel), 3)))]
        if args.mode == "fk" and args.model:
            print(f"  {args.model} has no keyframe animation; showing the rest pose")
    else:
        frames = load_keyframes(kf_path)
    print(f"  model: {args.model or 'built-in rig'} "
          f"({len(skel)} bones, {V.shape[0]} verts)")

    if args.headless:
        if args.mode == "fk":
            run_fk_headless(skel, V, F, W, frames, label)
        else:
            run_ik_headless(skel, V, F, W, end_effectors)
        return

    try:
        run_interactive(skel, V, F, W, frames, label, args.mode,
                        max_frames=args.frames, end_effectors=end_effectors)
    except Exception as exc:  # noqa: BLE001
        print(f"[viewer unavailable: {exc}]\nFalling back to --headless.")
        if args.mode == "fk":
            run_fk_headless(skel, V, F, W, frames, label)
        else:
            run_ik_headless(skel, V, F, W, end_effectors)


if __name__ == "__main__":
    main()
