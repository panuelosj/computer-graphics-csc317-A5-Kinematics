import copy

import numpy as np

from src.transformed_tips import transformed_tips


def kinematics_jacobian(skeleton, b):
    """Kinematic Jacobian of the tip positions w.r.t. all joint angles.

    Build the ``(3 * len(b), 3 * len(skeleton))`` matrix by central finite
    differences. For each bone ``si`` and each of its three angles ``a``,
    perturb that angle by ``+/- epsilon`` (use ``epsilon = 1e-7``) on a copy of
    the skeleton, evaluate :func:`transformed_tips`, and set the corresponding
    column to ``(tips_plus - tips_minus) / (2 * epsilon)``.
    """
    # TODO: finite-difference each joint angle to fill the Jacobian columns.
    raise NotImplementedError("Implement kinematics_jacobian")
