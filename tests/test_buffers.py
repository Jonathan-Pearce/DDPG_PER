"""Tests for ddpg_per.buffers (ReplayBuffer and PrioritizedReplayBuffer)."""

import numpy as np
import pytest

from ddpg_per.buffers import ReplayBuffer, PrioritizedReplayBuffer


def _make_transition(seed=0):
    """Return a deterministic (s, a, r, s', done) tuple for testing."""
    rng = np.random.default_rng(seed)
    s = rng.random(4).astype(np.float32)
    a = rng.random(2).astype(np.float32)
    r = float(rng.random())
    s_new = rng.random(4).astype(np.float32)
    done = 0.0
    return s, a, r, s_new, done


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(100)
        assert len(buf) == 0
        s, a, r, s_new, done = _make_transition()
        buf.add(s, a, r, s_new, done)
        assert len(buf) == 1

    def test_overflow_fifo(self):
        buf = ReplayBuffer(3)
        for i in range(5):
            s, a, r, s_new, done = _make_transition(i)
            buf.add(s, a, r, s_new, done)
        # Buffer should still hold only 3 transitions
        assert len(buf) == 3

    def test_sample_shapes(self):
        buf = ReplayBuffer(100)
        for i in range(20):
            buf.add(*_make_transition(i))
        obs, acts, rews, next_obs, dones = buf.sample(8)
        assert obs.shape == (8, 4)
        assert acts.shape == (8, 2)
        assert rews.shape == (8,)
        assert next_obs.shape == (8, 4)
        assert dones.shape == (8,)

    def test_sample_values_in_buffer(self):
        """All sampled observations should come from the stored transitions."""
        buf = ReplayBuffer(10)
        stored_obs = []
        for i in range(10):
            s, a, r, s_new, done = _make_transition(i)
            buf.add(s, a, r, s_new, done)
            stored_obs.append(s)

        obs_batch, _, _, _, _ = buf.sample(10)
        stored_arr = np.stack(stored_obs)
        for obs in obs_batch:
            # Each sampled obs must match at least one stored obs
            assert any(np.allclose(obs, stored) for stored in stored_arr)

    def test_sample_requires_data(self):
        """Sampling from an empty buffer should raise an error."""
        buf = ReplayBuffer(10)
        with pytest.raises((ValueError, IndexError)):
            buf.sample(1)


class TestPrioritizedReplayBuffer:
    def test_add_and_len(self):
        buf = PrioritizedReplayBuffer(128, alpha=0.6)
        buf.add(*_make_transition())
        assert len(buf) == 1

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            PrioritizedReplayBuffer(128, alpha=-0.1)

    def test_sample_shapes(self):
        buf = PrioritizedReplayBuffer(128, alpha=0.6)
        for i in range(32):
            buf.add(*_make_transition(i))
        obs, acts, rews, next_obs, dones, weights, idxes = buf.sample(8, beta=0.4)
        assert obs.shape == (8, 4)
        assert acts.shape == (8, 2)
        assert weights.shape == (8,)
        assert len(idxes) == 8

    def test_weights_normalized(self):
        """Maximum importance-sampling weight must be 1."""
        buf = PrioritizedReplayBuffer(128, alpha=0.6)
        for i in range(32):
            buf.add(*_make_transition(i))
        _, _, _, _, _, weights, _ = buf.sample(16, beta=0.4)
        assert np.max(weights) == pytest.approx(1.0, abs=1e-5)

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(128, alpha=0.6)
        for i in range(32):
            buf.add(*_make_transition(i))
        _, _, _, _, _, _, idxes = buf.sample(8, beta=0.4)
        new_priorities = np.ones(len(idxes)) * 2.0
        buf.update_priorities(idxes, new_priorities)
        # After updating, max priority should be 2.0
        assert buf._max_priority == pytest.approx(2.0)

    def test_update_priorities_bad_length(self):
        buf = PrioritizedReplayBuffer(128, alpha=0.6)
        for i in range(16):
            buf.add(*_make_transition(i))
        _, _, _, _, _, _, idxes = buf.sample(4, beta=0.4)
        with pytest.raises(ValueError):
            buf.update_priorities(idxes, [1.0])  # wrong length

    def test_update_priorities_zero_priority(self):
        buf = PrioritizedReplayBuffer(128, alpha=0.6)
        for i in range(16):
            buf.add(*_make_transition(i))
        _, _, _, _, _, _, idxes = buf.sample(4, beta=0.4)
        with pytest.raises(ValueError):
            buf.update_priorities(idxes[:1], [0.0])  # zero priority

    def test_invalid_beta(self):
        buf = PrioritizedReplayBuffer(128, alpha=0.6)
        for i in range(16):
            buf.add(*_make_transition(i))
        with pytest.raises(ValueError):
            buf.sample(4, beta=-0.1)  # negative beta is invalid
