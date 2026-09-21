import numpy as np


def _occ_from_indices(K, indices):
    occ = np.zeros(K, dtype=np.float32)
    occ[list(indices)] = 1.0
    return occ


class StateBuilder:
    """Full spectrum waterfall, shape (T, K). Each row: the full binary
    occupancy of the spectrum at one past instant."""

    def __init__(self, T, K):
        self.T, self.K = T, K
        self._window = np.zeros((T, K), dtype=np.float32)

    def reset(self):
        self._window[:] = 0.0

    def push(self, full_occupancy):
        self._window = np.roll(self._window, shift=-1, axis=0)
        self._window[-1] = full_occupancy.astype(np.float32)
        return self._window.copy()

    def current(self):
        return self._window.copy()