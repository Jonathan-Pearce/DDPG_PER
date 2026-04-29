"""
DDPG agent: Actor, Critic, and the DDPG training algorithm.

Architecture follows the experimental-details section of:
    "Continuous control with deep reinforcement learning"
    Lillicrap et al., 2015 (https://arxiv.org/abs/1509.02971)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """Deterministic policy network: maps states to actions.

    Three fully-connected layers (400 → 300 → a_dim) with ReLU activations
    and a final tanh scaled by ``a_max`` so that actions stay within bounds.

    Parameters
    ----------
    s_dim : int
        Dimensionality of the observation space.
    a_dim : int
        Dimensionality of the action space.
    a_max : float
        Absolute maximum value of any action component (used to scale tanh).
    """

    def __init__(self, s_dim: int, a_dim: int, a_max: float) -> None:
        super().__init__()
        self.a_max = a_max

        self.l1 = nn.Linear(s_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, a_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → action."""
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x)) * self.a_max


class Critic(nn.Module):
    """Action-value (Q) network: maps (state, action) pairs to scalar Q-values.

    The action is concatenated with the first hidden layer's output before
    the second layer, matching the original DDPG paper architecture.

    Parameters
    ----------
    s_dim : int
        Dimensionality of the observation space.
    a_dim : int
        Dimensionality of the action space.
    """

    def __init__(self, s_dim: int, a_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(s_dim, 400)
        self.l2 = nn.Linear(400 + a_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Forward pass: (state, action) → Q-value."""
        x = F.relu(self.l1(s))
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.l2(x))
        return self.l3(x)


class DDPG:
    """Deep Deterministic Policy Gradient agent.

    Maintains actor/critic networks plus their slow-moving target copies.
    Supports both uniform and prioritized experience replay.

    Parameters
    ----------
    s_dim : int
        Dimensionality of the observation space.
    a_dim : int
        Dimensionality of the action space.
    a_max : float
        Absolute maximum value of any action component.
    actor_lr : float, optional
        Learning rate for the actor optimizer (default: 1e-4).
    critic_lr : float, optional
        Learning rate for the critic optimizer (default: 1e-3).
    critic_wd : float, optional
        L2 weight-decay for the critic optimizer (default: 1e-2).
    """

    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        a_max: float,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        critic_wd: float = 1e-2,
    ) -> None:
        # --- Actor ---
        self.actor = Actor(s_dim, a_dim, a_max).to(device)
        self.actor_target = Actor(s_dim, a_dim, a_max).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )

        # --- Critic ---
        self.critic = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=critic_wd
        )

    def get_action(self, s: np.ndarray) -> np.ndarray:
        """Select a deterministic action for the given state (no noise).

        Parameters
        ----------
        s : np.ndarray
            Current environment observation.

        Returns
        -------
        np.ndarray
            Action chosen by the actor network.
        """
        state = torch.FloatTensor(s.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        prioritized: bool,
        beta_value: float,
        epsilon: float,
        num_steps: int,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
    ) -> None:
        """Update actor, critic, and target networks using sampled experiences.

        Parameters
        ----------
        replay_buffer :
            Either a :class:`~ddpg_per.buffers.ReplayBuffer` (uniform) or a
            :class:`~ddpg_per.buffers.PrioritizedReplayBuffer` (prioritized).
        prioritized : bool
            Whether *replay_buffer* is a prioritized replay buffer.
        beta_value : float
            Importance-sampling exponent for PER (ignored when *prioritized*
            is ``False``).
        epsilon : float
            Small constant added to TD errors before updating priorities, to
            prevent zero priorities.
        num_steps : int
            Number of gradient update steps to perform (typically equal to the
            number of environment steps taken in the last episode).
        batch_size : int, optional
            Mini-batch size (default: 64).
        gamma : float, optional
            Discount factor (default: 0.99).
        tau : float, optional
            Soft target-update coefficient (default: 0.005).
        """
        for _ in range(num_steps):
            # ------------------------------------------------------------------
            # Sample from replay buffer
            # ------------------------------------------------------------------
            if prioritized:
                experience = replay_buffer.sample(batch_size, beta_value)
                s, a, r, s_new, done, weights, batch_idxes = experience
                # Importance-sampling weights are set to 1 (see project report
                # for the rationale behind this hyper-parameter choice).
                # The benefit of PER here comes from the non-uniform sampling
                # distribution rather than from IS correction.
                weights = np.ones_like(r)
            else:
                s, a, r, s_new, done = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(r), None

            # Ensure consistent (batch_size, 1) shapes for per-sample scalars
            r = r.reshape(-1, 1)
            done = done.reshape(-1, 1)

            # √w so that squaring inside MSE gives the correct weight
            weights = np.sqrt(weights).reshape(-1, 1)

            # Convert to tensors
            state = torch.FloatTensor(s).to(device)
            action = torch.FloatTensor(a).to(device)
            next_state = torch.FloatTensor(s_new).to(device)
            done_t = torch.FloatTensor(1 - done).to(device)
            reward = torch.FloatTensor(r).to(device)
            weights_t = torch.FloatTensor(weights).to(device)

            # ------------------------------------------------------------------
            # Critic update
            # ------------------------------------------------------------------
            with torch.no_grad():
                q_target = self.critic_target(
                    next_state, self.actor_target(next_state)
                )
                y = reward + done_t * gamma * q_target

            q = self.critic(state, action)
            td_errors = y - q
            weighted_td_errors = td_errors * weights_t
            zero_tensor = torch.zeros_like(weighted_td_errors)
            critic_loss = F.mse_loss(weighted_td_errors, zero_tensor)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ------------------------------------------------------------------
            # Actor update
            # ------------------------------------------------------------------
            actor_loss = -self.critic(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ------------------------------------------------------------------
            # Soft target updates
            # ------------------------------------------------------------------
            for p, p_tgt in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_tgt.data.copy_(tau * p.data + (1 - tau) * p_tgt.data)
            for p, p_tgt in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                p_tgt.data.copy_(tau * p.data + (1 - tau) * p_tgt.data)

            # ------------------------------------------------------------------
            # Update priorities (PER only)
            # ------------------------------------------------------------------
            if prioritized:
                td_np = td_errors.cpu().detach().numpy()
                new_priorities = np.abs(td_np) + epsilon
                replay_buffer.update_priorities(batch_idxes, new_priorities)
