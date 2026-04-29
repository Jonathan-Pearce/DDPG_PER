"""
train.py — Command-line entry point for training a DDPG (±PER) agent.

Usage examples
--------------
Train with prioritized experience replay (default):

    python train.py

Train with uniform replay on a different environment and seed:

    python train.py --env HopperBulletEnv-v0 --seed 1 --no-prioritized

Full list of options:

    python train.py --help
"""

import argparse
import os
import time

import gym
import numpy as np
import pybullet_envs  # noqa: F401 – registers PyBullet environments with gym
import torch

from ddpg_per import DDPG, ReplayBuffer, PrioritizedReplayBuffer, LinearSchedule


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a DDPG agent with optional Prioritized Experience Replay."
    )
    # Environment
    parser.add_argument(
        "--env",
        type=str,
        default="InvertedPendulumBulletEnv-v0",
        help="PyBullet gym environment ID (default: InvertedPendulumBulletEnv-v0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )

    # Algorithm selection
    parser.add_argument(
        "--prioritized",
        dest="prioritized",
        action="store_true",
        default=True,
        help="Use prioritized experience replay (default: True).",
    )
    parser.add_argument(
        "--no-prioritized",
        dest="prioritized",
        action="store_false",
        help="Use uniform experience replay instead of PER.",
    )

    # Training schedule
    parser.add_argument(
        "--train-steps",
        type=int,
        default=100_000,
        help="Total number of environment steps to train for (default: 100 000).",
    )
    parser.add_argument(
        "--test-freq",
        type=int,
        default=2_000,
        help="Evaluate the policy every this many steps (default: 2 000).",
    )

    # DDPG hyper-parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for gradient updates (default: 64).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.001,
        help="Soft target-update coefficient (default: 0.001).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="Replay buffer capacity (default: 1 000 000).",
    )

    # PER hyper-parameters
    parser.add_argument(
        "--per-alpha",
        type=float,
        default=0.6,
        help="PER priority exponent alpha (default: 0.6).",
    )
    parser.add_argument(
        "--per-beta0",
        type=float,
        default=0.4,
        help="Initial PER importance-sampling exponent beta (default: 0.4).",
    )
    parser.add_argument(
        "--per-eps",
        type=float,
        default=1e-6,
        help="Small constant added to TD errors to avoid zero priorities (default: 1e-6).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory where result .npy files are saved (default: data/).",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=True,
        help="Print training progress after each episode (default: True).",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Only print evaluation results.",
    )

    return parser.parse_args()


def test_policy(policy: DDPG, env: gym.Env, n_episodes: int = 10) -> float:
    """Evaluate the current policy over *n_episodes* episodes (no noise).

    Parameters
    ----------
    policy : DDPG
        The agent whose policy is to be evaluated.
    env : gym.Env
        The environment used for evaluation.
    n_episodes : int, optional
        Number of evaluation episodes (default: 10).

    Returns
    -------
    float
        Mean episode return across *n_episodes* episodes.
    """
    episode_rewards = np.zeros(n_episodes)
    for i in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = policy.get_action(np.array(s))
            s, r, done, _ = env.step(a)
            episode_rewards[i] += r
    mean_return = float(np.mean(episode_rewards))
    print(f"  Evaluation return: {mean_return:.2f}")
    return mean_return


def main() -> None:
    """Run the full training loop."""
    args = parse_args()

    # ------------------------------------------------------------------
    # File naming
    # ------------------------------------------------------------------
    prefix = "Priority_" if args.prioritized else ""
    env_short = args.env.split("BulletEnv")[0] if "BulletEnv" in args.env else args.env
    file_name = f"{prefix}DDPG_{env_short}_{args.seed}"
    print(f"Running: {file_name}")

    # ------------------------------------------------------------------
    # Environment and seeds
    # ------------------------------------------------------------------
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_max = float(env.action_space.high[0])

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    policy = DDPG(s_dim, a_dim, a_max)

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------
    if args.prioritized:
        replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.per_alpha)
        beta_iters = args.train_steps
        beta_schedule = LinearSchedule(
            beta_iters, initial_p=args.per_beta0, final_p=1.0
        )
    else:
        replay_buffer = ReplayBuffer(args.buffer_size)
        beta_schedule = None

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initial evaluation (random policy)
    # ------------------------------------------------------------------
    test_scores = [test_policy(policy, env)]
    np.save(os.path.join(args.output_dir, file_name), test_scores)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    total_time = 0
    test_time = 0
    done = True
    episode = 0
    episode_r = 0.0
    episode_t = 0

    t_0 = time.time()

    while total_time < args.train_steps:
        if done:
            if total_time != 0:
                if args.verbose:
                    elapsed = int(time.time() - t_0)
                    print(
                        f"Total Steps: {total_time:>7d} | "
                        f"Episode {episode:>4d} | "
                        f"Return: {episode_r:>8.2f} | "
                        f"Runtime: {elapsed}s"
                    )

                # Determine beta for importance sampling
                beta_value = 0.0
                if args.prioritized:
                    beta_value = beta_schedule.value(total_time)

                # Train DDPG for as many gradient steps as environment steps
                # taken in the last episode
                policy.train(
                    replay_buffer,
                    args.prioritized,
                    beta_value,
                    args.per_eps,
                    episode_t,
                    args.batch_size,
                    args.gamma,
                    args.tau,
                )

            # Periodic evaluation
            if test_time >= args.test_freq:
                test_time %= args.test_freq
                test_scores.append(test_policy(policy, env))
                np.save(os.path.join(args.output_dir, file_name), test_scores)

            # Reset episode
            s = env.reset()
            done = False
            episode_r = 0.0
            episode_t = 0
            episode += 1

        # ------------------------------------------------------------------
        # Environment interaction
        # ------------------------------------------------------------------
        a = policy.get_action(np.array(s))
        # Gaussian exploration noise
        a = (
            a + np.random.normal(0, 0.1, size=env.action_space.shape[0])
        ).clip(env.action_space.low, env.action_space.high)

        s_new, r, done, _ = env.step(a)
        # If the episode ended due to the time-limit (not a true terminal
        # state) we do not want to mask the bootstrap value.
        done_bool = (
            0.0
            if episode_t + 1 == env.spec.max_episode_steps
            else float(done)
        )
        episode_r += r

        # Store transition
        replay_buffer.add(s, a, r, s_new, done_bool)

        s = s_new
        episode_t += 1
        total_time += 1
        test_time += 1

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    test_scores.append(test_policy(policy, env))
    np.save(os.path.join(args.output_dir, file_name), test_scores)
    print(f"\nFinal scores: {test_scores}")


if __name__ == "__main__":
    main()
