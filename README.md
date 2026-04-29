# DDPG + Prioritized Experience Replay

Deep Deterministic Policy Gradient (DDPG) is one of the most popular deep
reinforcement learning algorithms for continuous control. Inspired by the
Deep Q-network (DQN), DDPG uses a replay buffer to stabilize Q-learning.
This project integrates **Prioritized Experience Replay (PER)** with DDPG and
evaluates both variants on PyBullet continuous-control benchmarks.

Our experiments show that prioritized experience replay can improve the
performance of DDPG. For the full project report see [report.pdf](report.pdf).

## Repository Structure

```
ddpg_per/         # Installable Python package
  __init__.py     # Public API
  agent.py        # Actor, Critic, and DDPG classes
  buffers.py      # ReplayBuffer and PrioritizedReplayBuffer
  utils.py        # LinearSchedule and SegmentTree utilities
  plots.py        # Plotting helpers
tests/            # pytest unit tests
train.py          # CLI training script
pyproject.toml    # Package metadata and build config
requirements.txt  # Runtime dependencies
```

## Installation

```bash
# Clone the repo
git clone https://github.com/Jonathan-Pearce/DDPG_PER.git
cd DDPG_PER

# Install (editable mode is recommended for development)
pip install -e ".[dev]"
```

## Quick Start

**Train with Prioritized Experience Replay (default):**

```bash
python train.py
```

**Train with uniform replay on a different environment and seed:**

```bash
python train.py --env HopperBulletEnv-v0 --seed 1 --no-prioritized
```

**Full list of CLI options:**

```
python train.py --help

usage: train.py [-h] [--env ENV] [--seed SEED] [--prioritized] [--no-prioritized]
                [--train-steps N] [--test-freq N] [--batch-size N]
                [--gamma γ] [--tau τ] [--buffer-size N]
                [--per-alpha α] [--per-beta0 β] [--per-eps ε]
                [--output-dir DIR] [--verbose] [--quiet]
```

## Using the Package Programmatically

```python
import numpy as np
import gym
import pybullet_envs  # noqa: registers PyBullet envs

from ddpg_per import DDPG, PrioritizedReplayBuffer, LinearSchedule

env = gym.make("InvertedPendulumBulletEnv-v0")
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_max = float(env.action_space.high[0])

policy = DDPG(s_dim, a_dim, a_max)
buffer = PrioritizedReplayBuffer(size=100_000, alpha=0.6)
beta_schedule = LinearSchedule(100_000, initial_p=0.4, final_p=1.0)

s = env.reset()
for t in range(1000):
    a = policy.get_action(np.array(s))
    a_noisy = (a + np.random.normal(0, 0.1, size=a_dim)).clip(
        env.action_space.low, env.action_space.high
    )
    s_new, r, done, _ = env.step(a_noisy)
    buffer.add(s, a_noisy, r, s_new, float(done))
    s = s_new if not done else env.reset()
```

## Plotting Results

```python
from ddpg_per.plots import plot_returns, print_max_statistics

# Plot mean ± std for InvertedPendulum
plot_returns("data", env_name="InvertedPendulum")

# Print per-seed maximum-return statistics
print_max_statistics("data", env_names=["InvertedPendulum", "Hopper"])
```

## Running Tests

```bash
pytest
```

## References

- Lillicrap et al., 2015 — [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) (DDPG)
- Schaul et al., 2016 — [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) (PER)
