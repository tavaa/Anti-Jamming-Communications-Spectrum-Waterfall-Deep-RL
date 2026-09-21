from typing import Any, Dict, Sequence, Tuple
import numpy as np


def evaluate_agent(agent: Any, env: Any, seed: int, n_steps: int) -> Dict[str, float]:
    """Evaluate agent greedily over n_steps.

    Args:
        agent: Agent instance with select_action and set_mode.
        env: Environment instance with reset and step.
        seed: Random seed for environment reset.
        n_steps: Evaluation horizon.

    Returns:
        Dictionary with mean reward (r_bar), transmission rate (mu_bar),
        and switching rate (sigma_bar).
    """
    agent.set_mode(False)
    state, _ = env.reset(seed=seed)
    rewards, mus, sws = [], [], []
    for _ in range(n_steps):
        a = agent.select_action(state)
        next_state, r, term, trunc, info = env.step(a)
        rewards.append(r)
        mus.append(info["mu"])
        sws.append(info["switched"])
        state = next_state
    return {
        "r_bar": float(np.mean(rewards)),
        "mu_bar": float(np.mean(mus)),
        "sigma_bar": float(np.mean(sws)),
    }


def evaluate_agent_multi_seed(agent: Any, env: Any, seeds: Sequence[int], n_steps: int) -> float:
    """Compute mean average reward across multiple random seeds."""
    return float(
        np.mean([evaluate_agent(agent, env, seed=s, n_steps=n_steps)["r_bar"] for s in seeds])
    )


def rolling_mean(x: Sequence[float], w: int) -> np.ndarray:
    """Compute 1D rolling average with window size w."""
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def rollout_waterfall(
    agent: Any, env: Any, seed: int, n_steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect ground-truth channel occupancy, selected actions, and collision outcomes.

    Returns:
        Tuple of (occupancy_history, actions, outcomes) where outcomes is boolean
        indicating successful transmission without collision.
    """
    agent.set_mode(False)
    state, _ = env.reset(seed=seed)
    occ_hist, actions, outcomes = [], [], []
    for _ in range(n_steps):
        a = agent.select_action(state)
        next_state, r, term, trunc, info = env.step(a)
        occ_hist.append(info["true_occupancy"].copy())
        actions.append(a)
        outcomes.append(not info["collision"])
        state = next_state
    return np.array(occ_hist), np.array(actions), np.array(outcomes)


def plot_waterfall(
    ax: Any,
    occ: np.ndarray,
    actions: np.ndarray,
    outcomes: np.ndarray,
    title: str,
    K: int = 9,
) -> None:
    """Render waterfall spectrogram with overlaid agent channel choices and outcomes."""
    ax.imshow(occ, cmap="gray_r", aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    ts = np.arange(len(actions))
    ax.scatter(
        actions[outcomes],
        ts[outcomes],
        marker="o",
        s=20,
        facecolors="none",
        edgecolors="#2ca02c",
        linewidths=1.3,
        label="Success",
    )
    ax.scatter(
        actions[~outcomes],
        ts[~outcomes],
        marker="x",
        s=24,
        color="#d62728",
        linewidths=1.6,
        label="Collision",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Channel k", fontsize=9)
    ax.set_ylabel("Time step", fontsize=9)
    ax.set_xticks(range(K))
    ax.set_yticks(range(0, len(actions) + 1, 25))