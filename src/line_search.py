import numpy as np


def line_search(f, proj_z, z, dz, max_step):
    """Backtracking line search along direction ``dz`` from point ``z``.

    Start from ``alpha = max_step`` and repeatedly halve it until the projected
    step sufficiently decreases the objective::

        zj = proj_z(z + alpha * dz)
        if f(z) - f(zj) >= alpha * t:   # t a small positive constant
            stop

    Cap the number of halvings (e.g. 200). Return the chosen step ``alpha``.
    """
    # TODO: exponentially shrink alpha until the (projected) step decreases f.
    raise NotImplementedError("Implement line_search")
