# ----------------------------------------------------------------------
#  Expected coverage utility for facility-location pruning
# ----------------------------------------------------------------------
import numpy as np

def expected_coverage_gain(k: int, distances: np.ndarray) -> float:
    """
    Greedy farthest-point approximation of facility-location coverage.

    Parameters
    ----------
    k : int
        Number of exemplars to keep.
    distances : ndarray, shape (n, n)
        Pair-wise distance matrix in [0, 1].

    Returns
    -------
    cov : float
        Expected coverage (1 – mean minimal distance to an exemplar).
    """
    n = distances.shape[0]
    if k >= n:
        return 1.0

    # Seed with the medoid (smallest mean distance)
    current = [int(np.argmin(distances.mean(axis=1)))]
    min_d = distances[current[0]].copy()

    while len(current) < k:
        next_idx = int(np.argmax(min_d))
        current.append(next_idx)
        min_d = np.minimum(min_d, distances[next_idx])

    return 1.0 - min_d.mean() 