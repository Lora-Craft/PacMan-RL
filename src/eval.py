import numpy as np
import torch as t
from pathlib import Path
from torchrl.envs.utils import step_mdp
from env import make_env

def _run_episode(env, policy_module):
    """
    Runs one 1 eval episode (without gradients)
    """

    tensordict = env.reset()
    ep_reward = 0.0

    while True:
        with t.no_grad():
            tensordict = policy_module(tensordict)
        
        tensordict = env.step(tensordict)

        ep_reward += tensordict["next", "reward"].item()
        done = tensordict["next", "done"].any().item()

        if done:
            ep_length = tensordict["next", "step_count"].item()
            return ep_reward, ep_length

        tensordict = step_mdp(tensordict)

def _compute_stats(values, prefix):
    return {
        f"{prefix}/mean": np.mean(values),
        f"{prefix}/std": np.std(values),
        f"{prefix}/max": np.max(values),
        f"{prefix}/min": np.min(values),
    }

def evaluate(policy_module, num_eps=3, render=False, record_dir=None, device="cpu"):
    """
    Run for specified number of episodes on current policy
    """
    record = record_dir is not None 

    if record:
        record_path = Path(record_dir)
        record_path.mkdir(parents=True, exist_ok=True)
        record_dir = str(record_path)

    env = make_env(render=render, record=record, record_dir=record_dir)

    results = [_run_episode(env, policy_module) for _ in range(num_eps)]
    rewards, lengths = zip(*results)

    env.close()

    metrics = {
        "mean_reward": np.mean(rewards),
        "mean_length": np.mean(lengths),
        "rewards": list(rewards),
        "lengths": list(lengths),
        "num_eps": num_eps,
    }

    metrics.update(_compute_stats(rewards, "eval/reward"))
    metrics.update(_compute_stats(lengths, "eval/length"))

    return metrics