from anti_jamming_rl.envs.state_builder import _occ_from_indices


class SweepJammer:
    """m contiguous tones, advancing by 1 slot per step, cyclic."""

    def __init__(self, K, m, step_shift=1):
        self.K, self.m, self.step_shift = K, m, step_shift
        self._p = 0

    def reset(self, rng):
        self._p = int(rng.integers(0, self.K))
        return self._occ()

    def step(self):
        self._p = (self._p + self.step_shift) % self.K
        return self._occ()

    def _occ(self):
        idx = [(self._p + i) % self.K for i in range(self.m)]
        return _occ_from_indices(self.K, idx)


class DynamicCombJammer:
    """m tones fixed for `dwell` steps, then a fresh uniform-random
    tuple is drawn. Never reacts to the agent's action."""

    def __init__(self, K, m, dwell):
        self.K, self.m, self.dwell = K, m, dwell
        self._slots, self._phase = None, 0

    def reset(self, rng):
        self._rng = rng
        self._slots = rng.choice(self.K, size=self.m, replace=False)
        self._phase = 0
        return _occ_from_indices(self.K, self._slots)

    def step(self):
        self._phase = (self._phase + 1) % self.dwell
        if self._phase == 0:
            self._slots = self._rng.choice(self.K, size=self.m, replace=False)
        return _occ_from_indices(self.K, self._slots)


class HybridJammer:
    """Alternates between sweep and dynamic-comb sub-regimes with
    probability rho of flipping each step."""

    def __init__(self, K, m, regime_switch_prob, dwell):
        self.K, self.m, self.rho, self.dwell = K, m, regime_switch_prob, dwell
        self._regime = "S"
        self._sweep_p, self._dc_slots, self._dc_phase = 0, None, 0

    def reset(self, rng):
        self._rng = rng
        self._regime = "S"
        self._sweep_p = int(rng.integers(0, self.K))
        self._dc_slots = rng.choice(self.K, size=self.m, replace=False)
        self._dc_phase = 0
        return self._occ()

    def step(self):
        if self._rng.random() < self.rho:
            self._regime = "DC" if self._regime == "S" else "S"
            if self._regime == "DC":
                self._dc_slots = self._rng.choice(self.K, size=self.m, replace=False)
                self._dc_phase = 0
                return self._occ()
            self._sweep_p = 0

        if self._regime == "S":
            self._sweep_p = (self._sweep_p + 1) % self.K
        else:
            self._dc_phase = (self._dc_phase + 1) % self.dwell
            if self._dc_phase == 0:
                self._dc_slots = self._rng.choice(self.K, size=self.m, replace=False)
        return self._occ()

    def _occ(self):
        if self._regime == "S":
            idx = [(self._sweep_p + i) % self.K for i in range(self.m)]
        else:
            idx = self._dc_slots
        return _occ_from_indices(self.K, idx)