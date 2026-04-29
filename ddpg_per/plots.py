"""
Plotting utilities for visualising DDPG / DDPG+PER training results.

Functions
---------
plot_returns
    Plot mean ± std of episode returns for one or more models.
plot_hyperparameter_sweep
    Plot a beta-annealing hyper-parameter comparison.
print_max_statistics
    Print mean and std of the per-seed maximum return for each model.
"""

import os
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches  # noqa: F401 – re-exported for callers
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for plotting. "
        "Install it with: pip install matplotlib"
    ) from exc


def plot_returns(
    data_dir: str,
    env_name: str,
    model_prefixes: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    x_scale: int = 2,
    smoothing: int = 3,
    ylim: Optional[tuple] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot the mean ± std of episode returns for one or more models.

    Looks for ``.npy`` files with the naming convention::

        {prefix}DDPG_{env_name}_{seed}.npy

    where ``seed`` runs over ``[0, 1, 2]``.

    Parameters
    ----------
    data_dir : str
        Directory that contains the ``.npy`` result files.
    env_name : str
        Short environment name embedded in the file names (e.g.
        ``"InvertedPendulum"`` or ``"Hopper"``).
    model_prefixes : list of str, optional
        List of filename prefixes, one per model.  Defaults to
        ``["", "Priority_"]`` (uniform vs. prioritized replay).
    labels : list of str, optional
        Legend labels corresponding to *model_prefixes*.  Defaults to
        ``["uniform replay sampling", "prioritized replay sampling"]``.
    x_scale : int, optional
        Spacing (in thousands of steps) between evaluation points
        (default: ``2``).
    smoothing : int, optional
        Window size for uniform smoothing applied to mean and std
        (default: ``3``).  Set to ``1`` to disable.
    ylim : tuple of (float, float) or None, optional
        Y-axis limits.  If ``None`` matplotlib's auto-scaling is used.
    save_path : str or None, optional
        If provided, the figure is saved to this path instead of displayed.
    """
    if model_prefixes is None:
        model_prefixes = ["", "Priority_"]
    if labels is None:
        labels = ["uniform replay sampling", "prioritized replay sampling"]

    num_seeds = 3
    num_evals = 51
    x = np.arange(0, num_evals * x_scale, x_scale)

    fig, ax = plt.subplots()
    for prefix, label in zip(model_prefixes, labels):
        data = np.zeros((num_seeds, num_evals))
        for seed in range(num_seeds):
            filename = os.path.join(
                data_dir, f"{prefix}DDPG_{env_name}_{seed}.npy"
            )
            data[seed, :] = np.load(filename)

        mean = ndimage.uniform_filter(np.mean(data, axis=0), size=smoothing)
        std = ndimage.uniform_filter(np.std(data, axis=0), size=smoothing)

        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)

    ax.legend(loc="lower right")
    ax.set_title(env_name, fontsize=18)
    ax.set_xlabel("time steps (1e3)", fontsize=14)
    ax.set_ylabel("average return", fontsize=14)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_hyperparameter_sweep(
    data_dir: str,
    env_name: str,
    file_label_map: Dict[str, str],
    x_scale: int = 2,
    ylim: Optional[tuple] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a beta-annealing hyper-parameter comparison.

    Parameters
    ----------
    data_dir : str
        Directory containing the ``.npy`` result files.
    env_name : str
        Short environment name (used as the plot title).
    file_label_map : dict of {str: str}
        Mapping from ``.npy`` file name (inside *data_dir*) to legend label.
    x_scale : int, optional
        Spacing (in thousands of steps) between evaluation points (default: 2).
    ylim : tuple of (float, float) or None, optional
        Y-axis limits.
    save_path : str or None, optional
        If provided, the figure is saved here instead of displayed.
    """
    num_evals = 51
    x = np.arange(0, num_evals * x_scale, x_scale)

    fig, ax = plt.subplots(figsize=(10, 5))
    for filename, label in file_label_map.items():
        data = np.load(os.path.join(data_dir, filename))
        ax.plot(x, data, label=label)

    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    ax.set_title("\u03B2 Annealing Schedule", fontsize=18)
    ax.set_xlabel("time steps (1e3)", fontsize=14)
    ax.set_ylabel("average return", fontsize=14)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def print_max_statistics(
    data_dir: str,
    env_names: List[str],
    model_prefixes: Optional[List[str]] = None,
) -> None:
    """Print mean and standard deviation of the per-seed maximum return.

    Parameters
    ----------
    data_dir : str
        Directory containing the ``.npy`` result files.
    env_names : list of str
        Short environment names (e.g. ``["InvertedPendulum", "Hopper"]``).
    model_prefixes : list of str, optional
        Filename prefixes for each model.  Defaults to
        ``["", "Priority_"]``.
    """
    if model_prefixes is None:
        model_prefixes = ["", "Priority_"]

    num_seeds = 3
    num_evals = 51
    for prefix in model_prefixes:
        for env in env_names:
            data = np.zeros((num_seeds, num_evals))
            for seed in range(num_seeds):
                filename = os.path.join(
                    data_dir, f"{prefix}DDPG_{env}_{seed}.npy"
                )
                data[seed, :] = np.load(filename)
            max_per_seed = np.max(data, axis=1)
            print(f"{prefix}DDPG_{env}")
            print(f"  mean: {np.mean(max_per_seed):.2f}")
            print(f"  std:  {np.std(max_per_seed):.2f}")
