import argparse
import os
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TWO_STEPS_DIR = os.path.join(ROOT_DIR, "two_steps_model_v1")
if TWO_STEPS_DIR not in sys.path:
    sys.path.insert(0, TWO_STEPS_DIR)

from two_steps_model_v1.CorrectorPlayEnv_v5 import CorrectorPlayEnv
from two_steps_model_v1.train_hierarchical_rppo import (
    ActionSubsetWrapper,
    C1A1RewardWrapper,
    build_action_indices,
    extract_c1a1_obs,
    make_low_env,
    _format_ceos_action_from_table,
)


OBS_NAMES = CorrectorPlayEnv.KEYS


def build_c1a1_action_labels(action_table: list, c1a1_indices: Sequence[int]) -> list[str]:
    return [_format_ceos_action_from_table(action_table, base_idx) for base_idx in c1a1_indices]


def build_obs_grid(
    c1_values: np.ndarray,
    fixed_a1: float,
    a2: float = 0.0,
    b2: float = 0.0,
    a3: float = 0.0,
    s3: float = 0.0,
    c3: float = 0.0,
) -> np.ndarray:
    obs_grid = np.zeros((len(c1_values), len(OBS_NAMES)), dtype=np.float32)
    obs_grid[:, 0] = c1_values
    obs_grid[:, 1] = fixed_a1
    obs_grid[:, 2] = a2
    obs_grid[:, 3] = b2
    obs_grid[:, 4] = a3
    obs_grid[:, 5] = s3
    obs_grid[:, 6] = c3
    return obs_grid


def make_legacy_c1a1_env(seed: int, action_idx: Sequence[int], env_kwargs: dict):
    def _init():
        env = CorrectorPlayEnv(explode_keys=["C1", "A1"], **env_kwargs)
        env.reset(seed=seed)
        env = ActionSubsetWrapper(env, action_idx)
        env = C1A1RewardWrapper(env)
        return env

    return _init


def zero_lstm_states(model: RecurrentPPO, batch_size: int) -> RNNStates:
    shape = model.policy.lstm_hidden_state_shape
    actor_state = th.zeros((shape[0], batch_size, shape[2]), dtype=th.float32, device=model.device)
    critic_state = th.zeros((shape[0], batch_size, shape[2]), dtype=th.float32, device=model.device)
    return RNNStates(pi=(actor_state, actor_state.clone()), vf=(critic_state, critic_state.clone()))


def get_action_probability_matrix(model: RecurrentPPO, obs_grid: np.ndarray) -> np.ndarray:
    expected_dim = int(model.observation_space.shape[0])
    if obs_grid.shape[1] != expected_dim:
        if expected_dim == 2:
            obs_grid = extract_c1a1_obs(obs_grid)
        else:
            raise ValueError(f"Model expects obs dim {expected_dim}, but got grid with dim {obs_grid.shape[1]}.")

    obs_tensor, _ = model.policy.obs_to_tensor(obs_grid)
    episode_starts = th.ones((obs_grid.shape[0],), dtype=th.float32, device=model.device)
    lstm_states = zero_lstm_states(model, batch_size=obs_grid.shape[0])

    with th.no_grad():
        distribution, _ = model.policy.get_distribution(
            obs_tensor,
            lstm_states=lstm_states.pi,
            episode_starts=episode_starts,
        )

    if hasattr(distribution.distribution, "probs"):
        probs = distribution.distribution.probs
    else:
        raise RuntimeError("Expected a categorical action distribution with `.distribution.probs`.")

    return probs.detach().cpu().numpy().T


def save_probability_csv(
    save_path: str,
    c1_values: np.ndarray,
    action_labels: Sequence[str],
    prob_matrix: np.ndarray,
) -> None:
    df = pd.DataFrame(prob_matrix, index=action_labels, columns=np.round(c1_values, 6))
    df.index.name = "action"
    df.columns.name = "C1"
    df.to_csv(save_path)


def plot_probability_heatmap(
    save_path: str,
    c1_values: np.ndarray,
    fixed_a1: float,
    action_labels: Sequence[str],
    prob_matrix: np.ndarray,
    title: str | None = None,
) -> None:
    fig_width = max(10, len(c1_values) * 0.12)
    fig_height = max(4.5, len(action_labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        prob_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        extent=[float(c1_values[0]), float(c1_values[-1]), -0.5, len(action_labels) - 0.5],
    )

    ax.set_yticks(np.arange(len(action_labels)))
    ax.set_yticklabels(action_labels)
    ax.set_xlabel("C1 value")
    ax.set_ylabel("C1A1 action")
    ax.set_title(title or f"C1A1 policy probabilities at fixed A1={fixed_a1}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Action probability")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_c1a1_policy_heatmap(
    model_path: str,
    env_kwargs: dict,
    fixed_a1: float,
    c1_min: float,
    c1_max: float,
    n_points: int,
    out_dir: str,
    seed: int = 0,
    a2: float = 0.0,
    b2: float = 0.0,
    a3: float = 0.0,
    s3: float = 0.0,
    c3: float = 0.0,
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    tmp_env = CorrectorPlayEnv(**env_kwargs)
    action_table = tmp_env.action_table
    idx = build_action_indices(tmp_env)
    idx_c1a1 = idx["C1A1"]
    action_labels = build_c1a1_action_labels(action_table, idx_c1a1)

    dummy_env = DummyVecEnv(
        [make_low_env("C1A1", seed=seed, action_idx=idx_c1a1, env_kwargs=dict(env_kwargs))]
    )
    try:
        model = RecurrentPPO.load(model_path, env=dummy_env, device="auto")
    except ValueError as exc:
        if "Observation spaces do not match" not in str(exc):
            raise
        legacy_dummy_env = DummyVecEnv(
            [make_legacy_c1a1_env(seed=seed, action_idx=idx_c1a1, env_kwargs=dict(env_kwargs))]
        )
        model = RecurrentPPO.load(model_path, env=legacy_dummy_env, device="auto")

    c1_values = np.linspace(c1_min, c1_max, n_points, dtype=np.float32)
    obs_grid = build_obs_grid(c1_values, fixed_a1, a2=a2, b2=b2, a3=a3, s3=s3, c3=c3)
    prob_matrix = get_action_probability_matrix(model, obs_grid)

    heatmap_path = os.path.join(out_dir, f"c1a1_prob_heatmap_A1_{fixed_a1:g}.png")
    csv_path = os.path.join(out_dir, f"c1a1_prob_heatmap_A1_{fixed_a1:g}.csv")

    extra_terms = f"A2={a2}, B2={b2}, A3={a3}, S3={s3}, C3={c3}"
    title = f"C1A1 action probabilities at fixed A1={fixed_a1} ({extra_terms})"
    plot_probability_heatmap(heatmap_path, c1_values, fixed_a1, action_labels, prob_matrix, title=title)
    save_probability_csv(csv_path, c1_values, action_labels, prob_matrix)

    return heatmap_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a C1-vs-action probability heatmap for a trained low-level C1A1 RecurrentPPO model."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained C1A1 .zip model.")
    parser.add_argument("--out-dir", type=str, default="eval_plots", help="Directory to save the heatmap and CSV.")
    parser.add_argument("--fixed-a1", type=float, required=True, help="Fixed A1 value for the sweep.")
    parser.add_argument("--c1-min", type=float, default=-20.0, help="Minimum C1 value.")
    parser.add_argument("--c1-max", type=float, default=20.0, help="Maximum C1 value.")
    parser.add_argument("--n-points", type=int, default=161, help="Number of C1 grid points.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for the dummy env when loading the model.")
    parser.add_argument("--a2", type=float, default=0.0, help="Fixed A2 value in the observation.")
    parser.add_argument("--b2", type=float, default=0.0, help="Fixed B2 value in the observation.")
    parser.add_argument("--a3", type=float, default=0.0, help="Fixed A3 value in the observation.")
    parser.add_argument("--s3", type=float, default=0.0, help="Fixed S3 value in the observation.")
    parser.add_argument("--c3", type=float, default=0.0, help="Fixed C3 value in the observation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_kwargs = dict(
        max_steps=500,
        couple_prob_pct=0.5,
        user_gamma={
            "C1-A1": 0.0005,
            "A1-C1": 0.24,
            "B2-A1": -0.912 / 10,
            "B2-C1": -1.82 / 10,
            "A2-A1": -1.244 / 10,
            "A2-C1": -0.637 / 10,
            "A2-B2": -1.18 / 5,
            "C3-C1": -0.72 / 100,
            "C3-A1": -0.44 / 100,
            "C3-A2": 0.967 / 10,
            "C3-B2": 0.882 / 10,
            "C3-S3": 0.345,
            "S3-A1": -0.325 / 100,
            "S3-C1": -1.332 / 100,
            "S3-A2": -0.777 / 20,
            "S3-B2": -0.577 / 20,
            "S3-A3": 0.2,
            "S3-C3": 0.23,
        },
        user_beta={},
        user_sigma={"C1": 1, "A1": 1, "A2": 100, "B2": 50, "C3": 200, "A3": 100, "S3": 100},
        init_ranges={
            "C1": (0, 20),
            "A1": (0, 50),
            "A2": (100, 500),
            "B2": (100, 500),
            "C3": (800, 3000),
            "S3": (1000, 3000),
            "A3": (1000, 3000),
        },
    )

    heatmap_path, csv_path = evaluate_c1a1_policy_heatmap(
        model_path=args.model_path,
        env_kwargs=env_kwargs,
        fixed_a1=args.fixed_a1,
        c1_min=args.c1_min,
        c1_max=args.c1_max,
        n_points=args.n_points,
        out_dir=args.out_dir,
        seed=args.seed,
        a2=args.a2,
        b2=args.b2,
        a3=args.a3,
        s3=args.s3,
        c3=args.c3,
    )

    print(f"Saved heatmap to: {heatmap_path}")
    print(f"Saved probabilities to: {csv_path}")


if __name__ == "__main__":
    main()
