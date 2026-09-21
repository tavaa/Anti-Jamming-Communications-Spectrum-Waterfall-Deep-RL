from typing import Any, Dict, List


def train_agent(agent: Any, env: Any, seed: int, n_steps: int) -> Dict[str, List[float]]:
    """Execute online training loop for an agent over a fixed step horizon.

    Args:
        agent: Agent instance providing set_mode, select_action, and observe methods.
        env: Environment instance providing reset and step methods.
        seed: Random seed used to reset the environment state.
        n_steps: Total interaction steps to execute.

    Returns:
        Dictionary containing the sequence of instantaneous rewards:
            - 'reward': List of float rewards obtained at each step.
    """
    agent.set_mode(True)
    state, _ = env.reset(seed=seed)
    rewards = []
    for _ in range(n_steps):
        a = agent.select_action(state)
        next_state, r, term, trunc, info = env.step(a)
        agent.observe(state, a, r, next_state)
        rewards.append(r)
        state = next_state
    return {"reward": rewards}