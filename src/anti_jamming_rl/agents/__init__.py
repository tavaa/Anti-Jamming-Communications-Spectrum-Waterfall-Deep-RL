import os
import numpy as np
import torch
from anti_jamming_rl.agents.base_agent import BaseAgent
from anti_jamming_rl.agents.baselines import FixedAgent, RandomAgent, ReactiveAgent
from anti_jamming_rl.agents.qlearning import QLearningAgent
from anti_jamming_rl.agents.dqn import DQNAgent, QNetwork, ReplayBuffer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Builds an agent based on the specified method and parameters
def make_agent(
    method,
    seed,
    K=9,
    T=20,
    gamma=0.99,
    ql_alpha=0.1,
    ql_eps_start=1.0,
    ql_eps_end=0.05,
    ql_eps_decay=2500,
    dqn_lr=5e-4,
    dqn_eps_start=1.0,
    dqn_eps_end=0.05,
    dqn_eps_decay=2500,
    dqn_replay=5000,
    dqn_warmup=200,
    dqn_batch=64,
    dqn_target_sync=100,
    fc_hidden=128,
    device=DEVICE,
):
    if method == "fixed":
        return FixedAgent(K, np.random.default_rng(seed))
    if method == "random":
        return RandomAgent(K, np.random.default_rng(seed))
    if method == "reactive":
        return ReactiveAgent(K, np.random.default_rng(seed))
    if method == "qlearning":
        return QLearningAgent(
            K, ql_alpha, gamma, ql_eps_start, ql_eps_end, ql_eps_decay, np.random.default_rng(seed)
        )
    if method == "dqn":
        return DQNAgent(
            T,
            K,
            K,
            gamma,
            dqn_lr,
            dqn_eps_start,
            dqn_eps_end,
            dqn_eps_decay,
            dqn_replay,
            dqn_warmup,
            dqn_batch,
            dqn_target_sync,
            seed,
            device,
            fc_hidden=fc_hidden,
        )
    raise ValueError(method)


def ckpt_path(algo, scenario, seed, weights_dir="weights"):
    ext = "pkl" if algo == "qlearning" else "pt"
    return os.path.join(weights_dir, algo, scenario, f"seed{seed}.{ext}")


__all__ = [
    "BaseAgent",
    "FixedAgent",
    "RandomAgent",
    "ReactiveAgent",
    "QLearningAgent",
    "DQNAgent",
    "QNetwork",
    "ReplayBuffer",
    "make_agent",
    "ckpt_path",
    "DEVICE",
]