import copy

import numpy as np


def copy_skeleton_at(skeleton, A):
    """Return a deep copy of ``skeleton`` with each bone's ``xzx`` set from ``A``.

    ``A`` is a flat vector of ``3 * len(skeleton)`` Euler angles; bone ``i``
    reads ``A[3*i : 3*i+3]``. Deep-copy the skeleton first (so the original is
    untouched and all other bone attributes are preserved), then overwrite each
    bone's ``xzx``.
    """
    # TODO: deep-copy and assign each bone's xzx from the flat vector A.
    raise NotImplementedError("Implement copy_skeleton_at")
