import numpy as np


def euler_angles_to_transform(xzx):
    """Build a rotation (as a 4x4 homogeneous transform) from XZX Euler angles.

    ``xzx`` is a 3-vector of *extrinsic* Euler angles (in degrees) rotating
    about the x-, z-, and x-axes. Compose the three axis rotations as::

        R = R_x(xzx[0]) @ R_z(xzx[1]) @ R_x(xzx[2])

    (convert degrees to radians first). Return a ``(4, 4)`` homogeneous matrix
    whose upper-left 3x3 block is ``R`` and whose translation is zero.
    """
    # TODO: build the three axis rotations and compose them.
    raise NotImplementedError("Implement euler_angles_to_transform")
