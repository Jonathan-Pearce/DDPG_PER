"""
Experience-replay buffers.

Classes
-------
ReplayBuffer
    Simple uniform random replay buffer.
PrioritizedReplayBuffer
    Proportional prioritized experience-replay buffer as described in:
    "Prioritized Experience Replay", Schaul et al., 2016
    (https://arxiv.org/abs/1511.05952)
"""

import random

import numpy as np

from ddpg_per.utils import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    """Uniform experience-replay buffer with a fixed capacity.

    When the buffer is full the oldest experience is overwritten (FIFO).

    Parameters
    ----------
    size : int
        Maximum number of transitions to store.
    """

    def __init__(self, size: int) -> None:
        self._storage: list = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(
        self,
        obs_t: np.ndarray,
        action: np.ndarray,
        reward: float,
        obs_tp1: np.ndarray,
        done: float,
    ) -> None:
        """Add a single transition to the buffer.

        Parameters
        ----------
        obs_t : np.ndarray
            Observation at time *t*.
        action : np.ndarray
            Action taken at time *t*.
        reward : float
            Reward received after taking *action* in state *obs_t*.
        obs_tp1 : np.ndarray
            Observation at time *t + 1*.
        done : float
            ``1.0`` if the episode ended, ``0.0`` otherwise.
        """
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            obs_t, action, reward, obs_tp1, done = self._storage[i]
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones),
        )

    def sample(self, batch_size: int):
        """Sample a batch of transitions uniformly at random.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        obs_batch : np.ndarray of shape (batch_size, s_dim)
        act_batch : np.ndarray of shape (batch_size, a_dim)
        rew_batch : np.ndarray of shape (batch_size,)
        next_obs_batch : np.ndarray of shape (batch_size, s_dim)
        done_mask : np.ndarray of shape (batch_size,)
        """
        idxes = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized experience-replay buffer.

    Uses a sum-segment tree and a min-segment tree to efficiently sample
    transitions proportional to their priority and compute importance-sampling
    weights.

    Parameters
    ----------
    size : int
        Maximum number of transitions to store.
    alpha : float
        Degree of prioritization: ``0`` ⟹ uniform, ``1`` ⟹ full priority.
    """

    def __init__(self, size: int, alpha: float) -> None:
        super().__init__(size)
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self._alpha = alpha

        # Segment trees require capacity that is a power of two
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Add a transition and assign it the current maximum priority."""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size: int):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size: int, beta: float):
        """Sample a batch of transitions weighted by priority.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        beta : float
            Importance-sampling exponent: ``0`` ⟹ no correction,
            ``1`` ⟹ full correction.

        Returns
        -------
        obs_batch : np.ndarray of shape (batch_size, s_dim)
        act_batch : np.ndarray of shape (batch_size, a_dim)
        rew_batch : np.ndarray of shape (batch_size,)
        next_obs_batch : np.ndarray of shape (batch_size, s_dim)
        done_mask : np.ndarray of shape (batch_size,)
        weights : np.ndarray of shape (batch_size,)
            Importance-sampling weights (normalized so that the maximum
            weight is 1).
        idxes : list of int
            Buffer indices of the sampled transitions (needed to update
            priorities after learning).
        """
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")

        idxes = self._sample_proportional(batch_size)

        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        weights = []
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights_arr = np.array(weights, dtype=np.float32)

        encoded = self._encode_sample(idxes)
        return tuple(list(encoded) + [weights_arr, idxes])

    def update_priorities(self, idxes, priorities) -> None:
        """Update the priorities of previously sampled transitions.

        Parameters
        ----------
        idxes : list of int
            Buffer indices returned by :meth:`sample`.
        priorities : array-like of float
            New priority values (must be strictly positive).
        """
        if len(idxes) != len(priorities):
            raise ValueError("idxes and priorities must have the same length")
        for idx, priority in zip(idxes, priorities):
            if priority <= 0:
                raise ValueError(
                    f"priorities must be strictly positive, got {priority}"
                )
            if not (0 <= idx < len(self._storage)):
                raise IndexError(
                    f"index {idx} out of range [0, {len(self._storage)})"
                )
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)
