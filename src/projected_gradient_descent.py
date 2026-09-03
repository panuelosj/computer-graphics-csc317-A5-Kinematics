import numpy as np

from src.line_search import line_search


def projected_gradient_descent(f, grad_f, proj_z, max_iters, z):
    """Minimise ``f`` subject to bound constraints via projected gradient descent.

    Repeat ``max_iters`` times::

        dz = -grad_f(z)
        sigma = line_search(f, proj_z, z, dz, max_step)   # large max_step
        z = proj_z(z + sigma * dz)

    Return the optimised flat vector of Euler angles ``z``.
    """
    # TODO: iterate descent + line search + projection.
    raise NotImplementedError("Implement projected_gradient_descent")
