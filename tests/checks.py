"""Unit checks for Assignment 5 (Kinematics).

Each check exercises one function in ``src/`` against the golden fixtures in
``tests/golden/a5_golden.npz`` and the sample rig in ``data/``. The same checks
are used by ``run_tests.py`` (for students) and by the grader. Marks per check
mirror the marking scheme.
"""

import copy
import os

import numpy as np

from cgcommon.testkit import Check, fixture_path, data_dir

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIGN = os.path.dirname(HERE)
DATA = data_dir(os.path.join(ASSIGN, "data"))

_GOLDEN = None
_RIG = None


def golden():
    global _GOLDEN
    if _GOLDEN is None:
        _GOLDEN = np.load(
            fixture_path(os.path.join(HERE, "golden", "a5_golden.npz"))
        )
    return _GOLDEN


def rig():
    """Load (a fresh deep copy of) the sample rig: (skeleton, V, F, W)."""
    global _RIG
    if _RIG is None:
        from backend.skeleton import load_rig
        _RIG = load_rig(DATA)
    skel, V, F, W = _RIG
    return copy.deepcopy(skel), V, F, W


def _posed_skeleton(pose):
    skel, V, F, W = rig()
    for i in range(len(skel)):
        skel[i].xzx = np.asarray(pose[i], dtype=float).copy()
    return skel


# --- individual checks ------------------------------------------------------

def check_euler_angles_to_transform():
    from src.euler_angles_to_transform import euler_angles_to_transform
    g = golden()
    for inp, exp in zip(g["euler_inputs"], g["euler_outputs"]):
        out = np.asarray(euler_angles_to_transform(inp), dtype=float)
        assert out.shape == (4, 4), f"euler_angles_to_transform: shape {out.shape}"
        assert np.allclose(out, exp, atol=1e-9), (
            "euler_angles_to_transform: wrong matrix for "
            f"angles {inp.tolist()}"
        )
        R = out[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), (
            "euler_angles_to_transform: rotation block not orthonormal"
        )
        assert abs(np.linalg.det(R) - 1.0) < 1e-9, (
            "euler_angles_to_transform: determinant must be 1"
        )
        assert np.allclose(out[3], [0, 0, 0, 1], atol=1e-12), (
            "euler_angles_to_transform: bottom row must be [0,0,0,1]"
        )


def check_forward_kinematics():
    from src.forward_kinematics import forward_kinematics
    g = golden()
    skel = _posed_skeleton(g["fk_pose"])
    T = forward_kinematics(skel)
    assert len(T) == len(skel), "forward_kinematics: wrong number of transforms"
    got = np.stack([np.asarray(t, dtype=float) for t in T])
    assert np.allclose(got, g["fk_T"], atol=1e-7), (
        "forward_kinematics: transforms differ from reference"
    )


def check_transformed_tips():
    from src.transformed_tips import transformed_tips
    g = golden()
    skel = _posed_skeleton(g["fk_pose"])
    out = np.asarray(transformed_tips(skel, g["b_indices"]), dtype=float)
    assert out.shape == g["tips"].shape, f"transformed_tips: shape {out.shape}"
    assert np.allclose(out, g["tips"], atol=1e-7), (
        "transformed_tips: tip positions differ from reference"
    )


def check_linear_blend_skinning():
    from src.linear_blend_skinning import linear_blend_skinning
    g = golden()
    skel, V, F, W = rig()
    T = [g["fk_T"][i] for i in range(len(skel))]
    U = np.asarray(linear_blend_skinning(V, skel, T, W), dtype=float)
    assert U.shape == g["lbs_U"].shape, f"linear_blend_skinning: shape {U.shape}"
    assert np.allclose(U, g["lbs_U"], atol=1e-7), (
        "linear_blend_skinning: deformed vertices differ from reference"
    )


def check_copy_skeleton_at():
    from src.copy_skeleton_at import copy_skeleton_at
    g = golden()
    skel, V, F, W = rig()
    before = [b.xzx.copy() for b in skel]
    A = g["copy_A"]
    out = copy_skeleton_at(skel, A)
    assert len(out) == len(skel), "copy_skeleton_at: wrong skeleton length"
    for i in range(len(skel)):
        assert np.allclose(out[i].xzx, A[3 * i : 3 * i + 3]), (
            f"copy_skeleton_at: bone {i} xzx not set from A"
        )
        # Other attributes preserved.
        assert out[i].parent_index == skel[i].parent_index, (
            "copy_skeleton_at: parent_index must be preserved"
        )
        assert np.allclose(out[i].T_rest, skel[i].T_rest), (
            "copy_skeleton_at: T_rest must be preserved"
        )
        # Original left untouched.
        assert np.allclose(skel[i].xzx, before[i]), (
            "copy_skeleton_at: must not mutate the input skeleton"
        )


def check_kinematics_jacobian():
    from src.kinematics_jacobian import kinematics_jacobian
    from src.transformed_tips import transformed_tips
    g = golden()
    skel = _posed_skeleton(g["fk_pose"])
    b = g["b_indices"]
    J = np.asarray(kinematics_jacobian(skel, b), dtype=float)
    assert J.shape == g["jac"].shape, f"kinematics_jacobian: shape {J.shape}"
    assert np.allclose(J, g["jac"], atol=1e-4), (
        "kinematics_jacobian: differs from reference finite-difference Jacobian"
    )
    # Also approximately matches an independent numerical Jacobian of
    # transformed_tips (different step size).
    eps = 1e-4
    n = len(skel)
    Jnum = np.zeros((3 * len(b), 3 * n))
    for si in range(n):
        for a in range(3):
            sp = copy.deepcopy(skel)
            sp[si].xzx = sp[si].xzx.copy()
            sp[si].xzx[a] += eps
            sm = copy.deepcopy(skel)
            sm[si].xzx = sm[si].xzx.copy()
            sm[si].xzx[a] -= eps
            Jnum[:, si * 3 + a] = (
                np.asarray(transformed_tips(sp, b)) - np.asarray(transformed_tips(sm, b))
            ) / (2 * eps)
    assert np.allclose(J, Jnum, atol=1e-2), (
        "kinematics_jacobian: does not match a numerical Jacobian of transformed_tips"
    )


def check_end_effectors_objective_and_gradient():
    from src.end_effectors_objective_and_gradient import (
        end_effectors_objective_and_gradient,
    )
    g = golden()
    skel, V, F, W = rig()
    b, xb0 = g["ik_b"], g["xb0"]
    f, grad_f, proj_z = end_effectors_objective_and_gradient(skel, b, xb0)
    z = g["obj_z"]
    # Objective value matches reference.
    assert abs(float(f(z)) - float(g["obj_f"])) < 1e-7, (
        "end_effectors: objective value differs from reference"
    )
    # Gradient matches reference and a finite difference of f.
    gv = np.asarray(grad_f(z), dtype=float)
    assert np.allclose(gv, g["obj_grad"], atol=1e-4), (
        "end_effectors: gradient differs from reference"
    )
    eps = 1e-4
    gnum = np.zeros_like(z)
    for i in range(len(z)):
        zp = z.copy(); zp[i] += eps
        zm = z.copy(); zm[i] -= eps
        gnum[i] = (f(zp) - f(zm)) / (2 * eps)
    assert np.allclose(gv, gnum, atol=1e-2), (
        "end_effectors: gradient does not match a finite difference of f"
    )
    # Projection respects the joint limits.
    pr = np.asarray(proj_z(g["proj_in"]), dtype=float)
    assert np.allclose(pr, g["proj_out"], atol=1e-9), (
        "end_effectors: proj_z differs from reference"
    )
    for si, bone in enumerate(skel):
        for a in range(3):
            val = pr[si * 3 + a]
            assert bone.xzx_min[a] - 1e-9 <= val <= bone.xzx_max[a] + 1e-9, (
                "end_effectors: proj_z must clamp angles within joint limits"
            )


def check_line_search():
    from src.end_effectors_objective_and_gradient import (
        end_effectors_objective_and_gradient,
    )
    from src.line_search import line_search
    g = golden()
    skel, V, F, W = rig()
    f, grad_f, proj_z = end_effectors_objective_and_gradient(skel, g["ik_b"], g["xb0"])
    z = np.zeros(int(g["nbones"]) * 3)
    dz = -np.asarray(grad_f(z), dtype=float)
    alpha = float(line_search(f, proj_z, z, dz, 100000.0))
    assert alpha > 0.0, "line_search: step must be positive"
    f0 = f(z)
    f1 = f(proj_z(z + alpha * dz))
    assert f1 < f0, (
        f"line_search: step must decrease the objective ({f1:.6g} !< {f0:.6g})"
    )


def check_projected_gradient_descent():
    from src.end_effectors_objective_and_gradient import (
        end_effectors_objective_and_gradient,
    )
    from src.projected_gradient_descent import projected_gradient_descent
    from src.transformed_tips import transformed_tips
    from src.copy_skeleton_at import copy_skeleton_at
    g = golden()
    skel, V, F, W = rig()
    b, xb0 = g["ik_b"], g["xb0"]
    f, grad_f, proj_z = end_effectors_objective_and_gradient(skel, b, xb0)
    z0 = np.zeros(int(g["nbones"]) * 3)
    z = np.asarray(projected_gradient_descent(f, grad_f, proj_z, 200, z0), dtype=float)
    # Objective strictly reduced.
    assert f(z) < f(z0), "projected_gradient_descent: must reduce the objective"
    # Reachable target: tip error driven near zero.
    tip = transformed_tips(copy_skeleton_at(skel, z), b)
    err = float(np.linalg.norm(np.asarray(tip) - np.asarray(xb0)))
    assert err < 5e-2, (
        f"projected_gradient_descent: tip should reach reachable target (err={err:.4g})"
    )


def get_checks():
    """Return the ordered list of checks for this assignment."""
    return [
        Check("euler_angles_to_transform", 17, check_euler_angles_to_transform,
              "XZX Euler angles to a 4x4 rotation"),
        Check("forward_kinematics", 17, check_forward_kinematics,
              "global bone pose transforms down the tree"),
        Check("linear_blend_skinning", 17, check_linear_blend_skinning,
              "weighted blend of bone transforms"),
        Check("copy_skeleton_at", 17, check_copy_skeleton_at,
              "copy skeleton with new joint angles"),
        Check("transformed_tips", 17, check_transformed_tips,
              "bone tip positions after FK"),
        Check("kinematics_jacobian", 5, check_kinematics_jacobian,
              "finite-difference kinematic Jacobian"),
        Check("line_search", 4, check_line_search,
              "backtracking line search decreases the objective"),
        Check("end_effectors_objective_and_gradient", 3,
              check_end_effectors_objective_and_gradient,
              "IK objective, gradient and projection"),
        Check("projected_gradient_descent", 3, check_projected_gradient_descent,
              "projected gradient descent reaches the target"),
    ]
