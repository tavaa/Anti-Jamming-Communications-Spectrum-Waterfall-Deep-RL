import numpy as np
from anti_jamming_rl.envs.state_builder import StateBuilder
from anti_jamming_rl.envs.jammers import SweepJammer, DynamicCombJammer, HybridJammer


class AntiJammingEnv:
    """
    Anti-jamming environment. 
    The environment is defined by the number of channels K, the state history length T,
    a jammer object that defines the jamming strategy, a switching penalty lambda_switch, 
    and a number of burn-in steps to initialize the state.
    """
    def __init__(self, K, T, jammer, lambda_switch, burn_in_steps):
        self.K, self.T, self.jammer = K, T, jammer
        self.lambda_switch, self.burn_in_steps = lambda_switch, burn_in_steps
        self.sb = StateBuilder(T, K)
        self._last_action, self._pending_occ = None, None

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        self.sb.reset()
        self._last_action = None
        occ = self.jammer.reset(rng)
        for _ in range(self.burn_in_steps):
            self.sb.push(occ)
            occ = self.jammer.step()
        self._pending_occ = occ
        return self.sb.current(), {}

    def step(self, action):
        occ = self._pending_occ
        mu = 1.0 - occ[action]
        switched = 1.0 if (self._last_action is not None and action != self._last_action) else 0.0
        reward = mu - self.lambda_switch * switched
        self._last_action = action
        self.sb.push(occ)
        self._pending_occ = self.jammer.step()
        next_state = self.sb.current()
        info = {"mu": mu, "switched": switched, "collision": mu < 0.5, "true_occupancy": occ}
        return next_state, reward, False, False, info


def make_env(scenario, K=9, T=20, M=3, dwell=10, rho_hybrid=0.02, lambda_switch=0.2, burn_in_steps=20):
    if scenario == "sweep":
        jammer = SweepJammer(K, M, step_shift=1)
    elif scenario == "dynamic_comb":
        jammer = DynamicCombJammer(K, M, dwell=dwell)
    elif scenario == "hybrid":
        jammer = HybridJammer(K, M, regime_switch_prob=rho_hybrid, dwell=dwell)
    else:
        raise ValueError(scenario)
    return AntiJammingEnv(K, T, jammer, lambda_switch, burn_in_steps)