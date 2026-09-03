import numpy as np

from src.copy_skeleton_at import copy_skeleton_at
from src.transformed_tips import transformed_tips
from src.kinematics_jacobian import kinematics_jacobian


def end_effectors_objective_and_gradient(skeleton, b, xb0):
    """Build the IK least-squares objective, gradient and projection closures.

    Return ``(f, grad_f, proj_z)`` where, for a flat vector of Euler angles
    ``z`` (length ``3 * len(skeleton)``):

    - ``f(z) = 0.5 * ||xb(z) - xb0||^2`` where ``xb(z)`` are the end-effector
      tip positions for the skeleton posed at ``z`` (use ``copy_skeleton_at``
      then ``transformed_tips``).
    - ``grad_f(z) = J(z)^T (xb(z) - xb0)`` where ``J`` is the kinematic Jacobian.
    - ``proj_z(z)`` returns a copy of ``z`` with each angle clamped to its
      bone's ``[xzx_min, xzx_max]`` joint limits.

    ``b`` lists the constrained end-effector bones; ``xb0`` stacks their target
    positions ``[x0 y0 z0 ...]``.
    """
    # TODO: return the three closures f, grad_f, proj_z.
    raise NotImplementedError("Implement end_effectors_objective_and_gradient")
