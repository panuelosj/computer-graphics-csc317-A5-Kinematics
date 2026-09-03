import numpy as np


def linear_blend_skinning(V, skeleton, T, W):
    """Deform mesh vertices by a weighted blend of bone pose transforms.

    Each posed vertex is a weighted combination of the bone transforms::

        U[v] = sum_b W[v, weight_index(b)] * (T[b] applied to V[v])

    Skip bones whose ``weight_index`` is negative.

    Parameters
    ----------
    V : (n, 3) rest vertices.
    skeleton : list of bones (each exposes ``.weight_index``).
    T : list of (4, 4) bone pose transforms (from forward kinematics).
    W : (n, nweights) skinning weights.

    Returns
    -------
    (n, 3) deformed vertices.
    """
    # TODO: accumulate the weighted, transformed vertex positions.
    raise NotImplementedError("Implement linear_blend_skinning")
