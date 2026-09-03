import numpy as np

from src.forward_kinematics import forward_kinematics


def transformed_tips(skeleton, b):
    """Positions of the listed bone tips after forward kinematics.

    Run :func:`forward_kinematics` to get the pose transforms ``T``. The
    canonical tip of bone ``i`` is ``(length, 0, 0)``; map it through that
    bone's rest transform ``T_rest[i]`` and then its pose transform ``T[i]``.

    ``b`` is a list of bone indices. Return a flat vector stacking the 3D tip
    positions as ``[x0 y0 z0 x1 y1 z1 ...]`` (length ``3 * len(b)``).
    """
    # TODO: forward kinematics, then transform each requested canonical tip.
    raise NotImplementedError("Implement transformed_tips")
