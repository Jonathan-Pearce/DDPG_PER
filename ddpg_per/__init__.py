"""
ddpg_per — Deep Deterministic Policy Gradient with Prioritized Experience Replay.

Public API
----------
DDPG          : The DDPG agent (actor-critic with soft target updates).
ReplayBuffer  : Uniform experience-replay buffer.
PrioritizedReplayBuffer : Proportional prioritized experience-replay buffer.
LinearSchedule: Linear annealing schedule (used for PER beta).
"""

from ddpg_per.agent import DDPG
from ddpg_per.buffers import ReplayBuffer, PrioritizedReplayBuffer
from ddpg_per.utils import LinearSchedule

__all__ = [
    "DDPG",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "LinearSchedule",
]
