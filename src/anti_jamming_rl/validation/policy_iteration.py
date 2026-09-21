import random
from itertools import combinations
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def average_reward_policy_iteration(
    states: Sequence[Any],
    actions: Sequence[Any],
    transition_fn: Callable[[Any, Any], Dict[Any, float]],
    reward_fn: Callable[[Any, Any], float],
    max_iter: int = 10_000,
    seed: int = 0,
    epsilon: float = 1e-8,
    stable_iters: int = 10,
) -> Tuple[float, Dict[Any, float], Dict[Any, Any]]:
    """Solve average-reward MDP via policy iteration.

    Returns:
        Tuple of (optimal_average_reward, relative_values_dict, optimal_policy_dict).
    """
    states = list(states)
    actions = list(actions)
    n = len(states)
    ref = states[0]  # Reference state with fixed gauge h(ref) = 0
    var_index = {s: i for i, s in enumerate(states[1:], start=1)}
    rng = random.Random(seed)

    def random_policy():
        return {s: rng.choice(actions) for s in states}

    policy = random_policy()
    prev_rho, stable_count = None, 0

    for _ in range(max_iter):
        # Policy evaluation: solve h(s) + rho = r(s, pi(s)) + sum P(s'|s, pi(s)) h(s')
        rows, cols, vals = [], [], []
        b = np.zeros(n)

        for row, s in enumerate(states):
            a = policy[s]
            # Column 0 corresponds to average reward rho
            rows.append(row); cols.append(0); vals.append(1.0)
            if s != ref:
                rows.append(row); cols.append(var_index[s]); vals.append(1.0)
            for s2, p in transition_fn(s, a).items():
                if s2 != ref:
                    rows.append(row); cols.append(var_index[s2]); vals.append(-p)
            b[row] = reward_fn(s, a)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        try:
            x = spla.spsolve(A, b)
            if not np.all(np.isfinite(x)):
                raise np.linalg.LinAlgError("singular")
        except Exception:
            # Restart with a new random policy if system is singular
            policy = random_policy()
            prev_rho = None
            stable_count = 0
            continue

        rho = float(x[0])
        h = {ref: 0.0}
        for s in states[1:]:
            h[s] = float(x[var_index[s]])

        # Convergence check
        if prev_rho is not None and abs(rho - prev_rho) < epsilon:
            stable_count += 1
        else:
            stable_count = 0
        prev_rho = rho

        if stable_count >= stable_iters:
            return rho, h, policy

        # Policy improvement: update greedy actions using relative values h
        new_policy = {}
        for s in states:
            best_a, best_val = actions[0], -float("inf")
            for a in actions:
                v = reward_fn(s, a) + sum(p * h[s2] for s2, p in transition_fn(s, a).items())
                if v > best_val:
                    best_val, best_a = v, a
            new_policy[s] = best_a
        policy = new_policy

    return rho, h, policy


def check_unichain(
    states: Sequence[Any],
    transition_fn: Callable[[Any, Any], Dict[Any, float]],
    policy: Dict[Any, Any],
) -> Dict[str, bool]:
    """Check if the Markov chain induced by policy has a single recurrent class."""
    states = list(states)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    rows, cols = [], []

    for s in states:
        for s2, p in transition_fn(s, policy[s]).items():
            if p > 0:
                rows.append(idx[s])
                cols.append(idx[s2])

    # Strongly connected components analysis
    graph = csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=True, connection="strong")
    closed = [True] * n_comp
    for r, c in zip(rows, cols):
        if labels[r] != labels[c]:
            closed[labels[r]] = False
    rec = [i for i in range(n_comp) if closed[i]]

    return {"is_unichain": len(rec) == 1}


def build_sweep_mdp(
    K: int = 9,
    M: int = 3,
    lambda_switch: float = 0.2,
) -> Tuple[List[Any], List[int], Callable, Callable]:
    """Build MDP components for sweep jammer scenario."""
    states = [(p, a) for p in range(K) for a in range(K)]
    actions = list(range(K))

    def occ(p):
        return {(p + i) % K for i in range(M)}

    def transition_fn(s, a):
        p, _ = s
        # Jammer advances position deterministically by 1 slot
        return {((p + 1) % K, a): 1.0}

    def reward_fn(s, a):
        p, a_last = s
        p_next = (p + 1) % K
        mu = 0.0 if a in occ(p_next) else 1.0
        sw = 1.0 if a != a_last else 0.0
        return mu - lambda_switch * sw

    return states, actions, transition_fn, reward_fn


def build_dynamic_comb_mdp(
    K: int = 9,
    M: int = 3,
    dwell: int = 10,
    lambda_switch: float = 0.2,
) -> Tuple[List[Any], List[int], Callable, Callable]:
    """Build MDP components for dynamic comb jammer scenario."""
    triples = list(combinations(range(K), M))
    n_triples = len(triples)
    states = [(phi, t, a) for phi in range(dwell) for t in triples for a in range(K)]
    actions = list(range(K))

    def transition_fn(s, a):
        phi, triple, _ = s
        phi_next = (phi + 1) % dwell
        dist = {}
        # At end of dwell period, jammer uniformly draws a new tone combination
        if phi_next == 0:
            for t2 in triples:
                dist[(0, t2, a)] = 1.0 / n_triples
        else:
            dist[(phi_next, triple, a)] = 1.0
        return dist

    def reward_fn(s, a):
        phi, triple, a_last = s
        phi_next = (phi + 1) % dwell
        sw = 1.0 if a != a_last else 0.0
        # Expected collision rate when channels are reshuffled
        mu = (1.0 - M / K) if phi_next == 0 else (0.0 if a in triple else 1.0)
        return mu - lambda_switch * sw

    return states, actions, transition_fn, reward_fn


def build_hybrid_mdp(
    K: int = 9,
    M: int = 3,
    dwell: int = 10,
    rho_hybrid: float = 0.02,
    lambda_switch: float = 0.2,
) -> Tuple[List[Any], List[int], Callable, Callable]:
    """Build MDP components for hybrid (regime-switching) jammer scenario."""
    triples = list(combinations(range(K), M))
    n_triples = len(triples)
    actions = list(range(K))
    states = [("S", p, a) for p in range(K) for a in range(K)]
    states += [("DC", t, phi, a) for t in triples for phi in range(dwell) for a in range(K)]

    def sweep_occ(p):
        return {(p + i) % K for i in range(M)}

    def transition_fn(s, a):
        dist = {}

        def add(st, p):
            if p > 0:
                dist[st] = dist.get(st, 0.0) + p

        if s[0] == "S":
            _, p, _ = s
            p_next = (p + 1) % K
            # Stay in sweep mode or switch to dynamic comb mode
            add(("S", p_next, a), 1.0 - rho_hybrid)
            for t in triples:
                add(("DC", t, 0, a), rho_hybrid / n_triples)
        else:
            _, triple, phi, _ = s
            phi_next = (phi + 1) % dwell
            # Dynamic comb progression with probability of switching to sweep
            if phi_next == 0:
                for t in triples:
                    add(("DC", t, 0, a), (1.0 - rho_hybrid) / n_triples)
            else:
                add(("DC", triple, phi_next, a), 1.0 - rho_hybrid)
            add(("S", 1 % K, a), rho_hybrid)
        return dist

    def reward_fn(s, a):
        sw = 1.0 if a != s[-1] else 0.0
        if s[0] == "S":
            _, p, _ = s
            p_next = (p + 1) % K
            mu_sweep = 0.0 if a in sweep_occ(p_next) else 1.0
            mu = (1.0 - rho_hybrid) * mu_sweep + rho_hybrid * (1.0 - M / K)
        else:
            _, triple, phi, _ = s
            phi_next = (phi + 1) % dwell
            mu_dc = (1.0 - M / K) if phi_next == 0 else (0.0 if a in triple else 1.0)
            mu_sweep_entry = 0.0 if a in sweep_occ(1 % K) else 1.0
            mu = (1.0 - rho_hybrid) * mu_dc + rho_hybrid * mu_sweep_entry
        return mu - lambda_switch * sw

    return states, actions, transition_fn, reward_fn