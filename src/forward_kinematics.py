import numpy as np

from src.euler_angles_to_transform import euler_angles_to_transform


def forward_kinematics(skeleton):
    """Compute the global pose transform of every bone in the skeleton.

    ``skeleton`` is a list of bones. Each bone exposes ``.xzx`` (Euler angles),
    ``.parent_index`` (``-1`` for the root) and ``.T_rest`` (the 4x4 rigid
    transform placing the canonical bone into the rest model).

    For each bone ``i``, read the relative rotation in the canonical frame and
    bring it through the rest frame::

        T[i] = T_rest[i] @ R(xzx[i]) @ inv(T_rest[i])

    then compose with the parent's pose transform (recurse up the tree)::

        T[i] = T[parent] @ T[i]

    The root (``parent_index < 0``) has no parent composition. Return a list of
    ``(4, 4)`` homogeneous transforms, one per bone.
    """
    # TODO: recursively accumulate each bone's global transform.
    raise NotImplementedError("Implement forward_kinematics")
