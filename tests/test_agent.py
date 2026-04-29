"""Tests for ddpg_per.agent (Actor, Critic, DDPG)."""

import numpy as np
import pytest
import torch

from ddpg_per.agent import Actor, Critic, DDPG
from ddpg_per.buffers import ReplayBuffer, PrioritizedReplayBuffer


S_DIM = 4
A_DIM = 2
A_MAX = 1.0


class TestActor:
    def test_output_shape(self):
        actor = Actor(S_DIM, A_DIM, A_MAX)
        s = torch.zeros(1, S_DIM)
        out = actor(s)
        assert out.shape == (1, A_DIM)

    def test_output_bounded(self):
        actor = Actor(S_DIM, A_DIM, A_MAX)
        s = torch.randn(32, S_DIM)
        out = actor(s)
        assert (out.abs() <= A_MAX + 1e-6).all()

    def test_batch_processing(self):
        actor = Actor(S_DIM, A_DIM, A_MAX)
        s = torch.randn(16, S_DIM)
        out = actor(s)
        assert out.shape == (16, A_DIM)


class TestCritic:
    def test_output_shape(self):
        critic = Critic(S_DIM, A_DIM)
        s = torch.zeros(1, S_DIM)
        a = torch.zeros(1, A_DIM)
        out = critic(s, a)
        assert out.shape == (1, 1)

    def test_batch_processing(self):
        critic = Critic(S_DIM, A_DIM)
        s = torch.randn(16, S_DIM)
        a = torch.randn(16, A_DIM)
        out = critic(s, a)
        assert out.shape == (16, 1)


class TestDDPG:
    def test_get_action_shape(self):
        policy = DDPG(S_DIM, A_DIM, A_MAX)
        s = np.zeros(S_DIM)
        a = policy.get_action(s)
        assert a.shape == (A_DIM,)

    def test_get_action_bounded(self):
        policy = DDPG(S_DIM, A_DIM, A_MAX)
        s = np.random.randn(S_DIM)
        a = policy.get_action(s)
        assert np.all(np.abs(a) <= A_MAX + 1e-6)

    def test_train_uniform(self):
        """Training with a uniform replay buffer should not raise errors."""
        policy = DDPG(S_DIM, A_DIM, A_MAX)
        buf = ReplayBuffer(1000)
        for i in range(100):
            s = np.random.randn(S_DIM).astype(np.float32)
            a = np.random.uniform(-A_MAX, A_MAX, A_DIM).astype(np.float32)
            r = float(np.random.randn())
            s_new = np.random.randn(S_DIM).astype(np.float32)
            done = 0.0
            buf.add(s, a, r, s_new, done)

        # Should complete without error
        policy.train(buf, prioritized=False, beta_value=0.0, epsilon=1e-6,
                     num_steps=5, batch_size=32)

    def test_train_prioritized(self):
        """Training with a PER buffer should not raise errors."""
        policy = DDPG(S_DIM, A_DIM, A_MAX)
        buf = PrioritizedReplayBuffer(1024, alpha=0.6)
        for i in range(100):
            s = np.random.randn(S_DIM).astype(np.float32)
            a = np.random.uniform(-A_MAX, A_MAX, A_DIM).astype(np.float32)
            r = float(np.random.randn())
            s_new = np.random.randn(S_DIM).astype(np.float32)
            done = 0.0
            buf.add(s, a, r, s_new, done)

        # Should complete without error
        policy.train(buf, prioritized=True, beta_value=0.4, epsilon=1e-6,
                     num_steps=5, batch_size=32)

    def test_target_networks_soft_update(self):
        """After training, target networks should not be identical to online networks."""
        torch.manual_seed(0)
        policy = DDPG(S_DIM, A_DIM, A_MAX)
        buf = ReplayBuffer(1000)
        for _ in range(100):
            s = np.random.randn(S_DIM).astype(np.float32)
            a = np.random.uniform(-A_MAX, A_MAX, A_DIM).astype(np.float32)
            r = float(np.random.randn())
            s_new = np.random.randn(S_DIM).astype(np.float32)
            buf.add(s, a, r, s_new, 0.0)

        # Record initial target weights
        initial_target = [
            p.clone() for p in policy.actor_target.parameters()
        ]

        policy.train(buf, prioritized=False, beta_value=0.0, epsilon=1e-6,
                     num_steps=10, batch_size=32)

        # Target weights should have changed (due to soft update)
        for p_before, p_after in zip(
            initial_target, policy.actor_target.parameters()
        ):
            assert not torch.equal(p_before, p_after)
