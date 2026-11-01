"""Optional compiled implementation of the same fixed-node Jacobi steps.

Numba is required only when this module is imported via the explicit backend.
There is no fast-math, in-place update, relaxation, or change to stopping rules.
"""

from numba import njit


@njit(cache=False, fastmath=False)
def compiled_jacobi(fixed_mask, potential, weight_x, weight_y, tol, nmax):
    """Apply the identical old-array Jacobi update with fixed nodes unchanged."""
    updated = potential.copy()
    for iteration in range(1, nmax + 1):
        final_change = 0.0
        for row in range(1, potential.shape[0] - 1):
            for col in range(1, potential.shape[1] - 1):
                if not fixed_mask[row, col]:
                    candidate = (
                        weight_x * potential[row, col - 1]
                        + weight_x * potential[row, col + 1]
                        + weight_y * potential[row - 1, col]
                        + weight_y * potential[row + 1, col]
                    )
                    updated[row, col] = candidate
                    change = abs(candidate - potential[row, col])
                    if change > final_change:
                        final_change = change
        potential, updated = updated, potential
        if final_change <= tol:
            return potential, iteration, final_change
    return potential, nmax, final_change
