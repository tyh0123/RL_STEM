import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from CorrectorPlayEnv_v4 import CorrectorPlayEnv


# Fixed environment configuration (same as train_ppo_v3.py)
USER_GAMMA = {
    'C1-A1': 0.0005,
    'A1-C1': 0.24,
    'B2-A1': -0.912/10, 'B2-C1': -1.82/10,
    'A2-A1': -1.244/10, 'A2-C1': -0.637/10, 'A2-B2': -1.18/5,
    'C3-C1': -0.72/100, 'C3-A1': -0.44/100, 'C3-A2': +0.967/10, 'C3-B2': +0.882/10, 'C3-S3': +0.345,
    'S3-A1': -0.325/100, 'S3-C1': -1.332/100, 'S3-A2': -0.777/20, 'S3-B2': -0.577/20, 'S3-A3': 0.2, 'S3-C3': 0.23,
}

USER_SIGMA = {
    'C1': 1,
    'A1': 1,
    'A2': 100,
    'B2': 20,
    'C3': 200,
    'A3': 100,
    'S3': 100,
}

INIT_RANGES = {
    'C1': (-20, 20),
    'A1': (0, 50),
    'A2': (0, 500),
    'B2': (0, 500),
    'C3': (-3000, 3000),
    'S3': (0, 3000),
    'A3': (0, 3000),
}


def load_model(model_path: str, vec_normalize_path: str = None, **env_kwargs):
    """Load trained model and optional VecNormalize stats."""
    model = PPO.load(model_path)
    print(f"Loaded model from: {model_path}")

    vec_normalize = None
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        # Create a dummy env with same kwargs for loading VecNormalize
        dummy_env = DummyVecEnv([lambda: CorrectorPlayEnv(**env_kwargs)])
        vec_normalize = VecNormalize.load(vec_normalize_path, dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        print(f"Loaded VecNormalize from: {vec_normalize_path}")

    return model, vec_normalize


def evaluate_episode(env, model, vec_normalize=None, deterministic=True, render=False):
    """Run a single evaluation episode and collect metrics."""
    obs, info = env.reset()

    if vec_normalize is not None:
        # Wrap obs for normalization - need to reshape for VecNormalize
        obs = vec_normalize.normalize_obs(obs.reshape(1, -1)).flatten()

    episode_reward = 0.0
    episode_length = 0
    deviations = [info.get("init_total_deviation", 0.0)]
    actions_taken = []
    action_details = []  # Store action details for logging
    rewards_per_step = []
    params_history = {k: [env.params[k]] for k in env.KEYS}

    done = False
    nan_encountered = False
    while not done:
        # Check for NaN/Inf in observation
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            nan_encountered = True
            break

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        if vec_normalize is not None:
            obs = vec_normalize.normalize_obs(obs.reshape(1, -1)).flatten()

        episode_reward += reward
        episode_length += 1
        deviations.append(info.get("total_deviation", 0.0))
        actions_taken.append(action)
        rewards_per_step.append(reward)

        # Store action details from info
        act_info = info.get("action", {})
        action_details.append(act_info)

        for k in env.KEYS:
            params_history[k].append(env.params[k])

        done = terminated or truncated

        if render:
            env.render(info)

    return {
        "reward": episode_reward,
        "length": episode_length,
        "goal_achieved": info.get("goal_achieved", False),
        "final_deviation": deviations[-1],
        "init_deviation": deviations[0],
        "deviations": deviations,
        "actions": actions_taken,
        "action_details": action_details,
        "rewards_per_step": rewards_per_step,
        "params_history": params_history,
        "nan_encountered": nan_encountered,
    }


def evaluate(model, n_episodes: int = 10, seed: int = 0, vec_normalize=None,
             deterministic: bool = True, render: bool = False, **env_kwargs):
    """Evaluate model over multiple episodes."""
    results = []
    nan_count = 0

    for ep in range(n_episodes):
        env = CorrectorPlayEnv(
            setting_seed=seed,  # Fixed settings
            init_seed=seed + ep * 1000,  # Different starting state per episode
            error_seed=seed + ep * 100 + 50,  # Different noise per episode
            render_mode="human" if render else None,
            **env_kwargs
        )

        result = evaluate_episode(env, model, vec_normalize, deterministic, render)

        # Skip episodes with NaN
        if result.get("nan_encountered", False):
            print(f"Episode {ep + 1:3d}: SKIPPED (NaN/Inf encountered at step {result['length']})")
            nan_count += 1
            env.close()
            continue

        results.append(result)

        status = "SUCCESS" if result["goal_achieved"] else "TIMEOUT"
        print(f"Episode {ep + 1:3d}: {status} | "
              f"Reward: {result['reward']:8.2f} | "
              f"Length: {result['length']:4d} | "
              f"Init Dev: {result['init_deviation']:8.2f} | "
              f"Final Dev: {result['final_deviation']:8.2f}")

        env.close()

    if nan_count > 0:
        print(f"\nWarning: {nan_count}/{n_episodes} episodes skipped due to NaN/Inf values")

    return results


def compare_with_random(model, n_episodes: int = 10, seed: int = 0,
                        vec_normalize=None, **env_kwargs):
    """Compare trained model with random policy."""
    print("\n" + "=" * 70)
    print("Evaluating TRAINED model (deterministic)")
    print("=" * 70)
    trained_results = evaluate(
        model, n_episodes, seed, vec_normalize,
        deterministic=True, render=False, **env_kwargs
    )

    print("\n" + "=" * 70)
    print("Evaluating RANDOM policy")
    print("=" * 70)
    random_results = []

    for ep in range(n_episodes):
        env = CorrectorPlayEnv(
            setting_seed=seed,
            init_seed=seed + ep * 1000,
            error_seed=seed + ep * 100 + 50,
            **env_kwargs
        )

        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        init_deviation = info.get("init_total_deviation", 0.0)
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        random_results.append({
            "reward": episode_reward,
            "length": episode_length,
            "goal_achieved": info.get("goal_achieved", False),
            "final_deviation": info.get("total_deviation", 0.0),
            "init_deviation": init_deviation,
        })

        status = "SUCCESS" if info.get("goal_achieved", False) else "TIMEOUT"
        print(f"Episode {ep + 1:3d}: {status} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Length: {episode_length:4d} | "
              f"Init Dev: {init_deviation:8.2f} | "
              f"Final Dev: {info.get('total_deviation', 0.0):8.2f}")

        env.close()

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    trained_rewards = [r["reward"] for r in trained_results]
    random_rewards = [r["reward"] for r in random_results]
    trained_success = sum(1 for r in trained_results if r["goal_achieved"])
    random_success = sum(1 for r in random_results if r["goal_achieved"])
    trained_lengths = [r["length"] for r in trained_results]
    random_lengths = [r["length"] for r in random_results]

    print(f"\n{'Metric':<25} {'Trained':>15} {'Random':>15}")
    print("-" * 55)
    print(f"{'Mean Reward':<25} {np.mean(trained_rewards):>15.2f} {np.mean(random_rewards):>15.2f}")
    print(f"{'Std Reward':<25} {np.std(trained_rewards):>15.2f} {np.std(random_rewards):>15.2f}")
    print(f"{'Success Rate':<25} {trained_success/n_episodes*100:>14.1f}% {random_success/n_episodes*100:>14.1f}%")
    print(f"{'Mean Episode Length':<25} {np.mean(trained_lengths):>15.1f} {np.mean(random_lengths):>15.1f}")
    print(f"{'Mean Final Deviation':<25} {np.mean([r['final_deviation'] for r in trained_results]):>15.2f} {np.mean([r['final_deviation'] for r in random_results]):>15.2f}")

    # Calculate improvement
    trained_final_dev = np.mean([r['final_deviation'] for r in trained_results])
    random_final_dev = np.mean([r['final_deviation'] for r in random_results])
    if random_final_dev > 0:
        improvement = (random_final_dev - trained_final_dev) / random_final_dev * 100
        print(f"\n{'Deviation Improvement':<25} {improvement:>14.1f}%")

    return trained_results, random_results


def format_action(act_info):
    """Format action info into readable string."""
    if not act_info:
        return "N/A"
    if act_info.get("type") == "pct_button":
        return f"{act_info['target']} {act_info['pct']:+d}%"
    elif act_info.get("type") == "step":
        direction = "+" if act_info.get("dir", 0) > 0 else "-"
        return f"{act_info['target']} {direction}{act_info['step']}"
    return str(act_info)


def print_action_log(result):
    """Print detailed action log to console."""
    print("\n" + "=" * 140)
    print("ACTION LOG (All KEYS)")
    print("=" * 140)
    print(f"{'Step':<6} {'Action':<15} {'C1':<8} {'A1':<8} {'A2':<8} {'B2':<8} {'A3':<8} {'S3':<8} {'C3':<8} {'Deviation':<12} {'Reward':<10}")
    print("-" * 140)

    cumulative_reward = 0.0
    deviations = result.get("deviations", [])
    action_details = result.get("action_details", [])
    rewards = result.get("rewards_per_step", [])
    params_history = result.get("params_history", {})

    # Get parameter histories for all KEYS
    c1_hist = params_history.get("C1", [])
    a1_hist = params_history.get("A1", [])
    a2_hist = params_history.get("A2", [])
    b2_hist = params_history.get("B2", [])
    a3_hist = params_history.get("A3", [])
    s3_hist = params_history.get("S3", [])
    c3_hist = params_history.get("C3", [])

    # Print initial state
    if c1_hist and a1_hist and a2_hist and b2_hist and a3_hist and s3_hist and c3_hist:
        print(f"{'Init':<6} {'':<15} {c1_hist[0]:<8.1f} {a1_hist[0]:<8.1f} {a2_hist[0]:<8.1f} {b2_hist[0]:<8.1f} {a3_hist[0]:<8.1f} {s3_hist[0]:<8.1f} {c3_hist[0]:<8.1f} {deviations[0]:<12.2f} {'':<10}")

    for i, (act, dev, rew) in enumerate(zip(action_details, deviations[1:], rewards)):
        cumulative_reward += rew
        action_str = format_action(act)
        c1 = c1_hist[i+1] if i+1 < len(c1_hist) else 0
        a1 = a1_hist[i+1] if i+1 < len(a1_hist) else 0
        a2 = a2_hist[i+1] if i+1 < len(a2_hist) else 0
        b2 = b2_hist[i+1] if i+1 < len(b2_hist) else 0
        a3 = a3_hist[i+1] if i+1 < len(a3_hist) else 0
        s3 = s3_hist[i+1] if i+1 < len(s3_hist) else 0
        c3 = c3_hist[i+1] if i+1 < len(c3_hist) else 0
        print(f"{i+1:<6} {action_str:<15} {c1:<8.1f} {a1:<8.1f} {a2:<8.1f} {b2:<8.1f} {a3:<8.1f} {s3:<8.1f} {c3:<8.1f} {dev:<12.2f} {rew:<10.4f}")

    print("-" * 140)
    print(f"{'FINAL':<6} {'':<15} {c1_hist[-1]:<8.1f} {a1_hist[-1]:<8.1f} {a2_hist[-1]:<8.1f} {b2_hist[-1]:<8.1f} {a3_hist[-1]:<8.1f} {s3_hist[-1]:<8.1f} {c3_hist[-1]:<8.1f} {deviations[-1]:<12.2f} {cumulative_reward:<10.4f}")
    print(f"\nGoal Achieved: {result.get('goal_achieved', False)}  |  Total Reward: {cumulative_reward:.4f}")
    print("=" * 140)


def plot_episode(result, save_path: str = None, goal_threshold: float = 50.0, truncate_at_best: bool = True):
    """Plot deviation and parameter trajectories for a single episode.

    Args:
        truncate_at_best: If True, only plot up to the minimum deviation point
    """
    # Print action log to console
    print_action_log(result)

    deviations = result["deviations"]

    # Find the best (minimum deviation) point
    if truncate_at_best and len(deviations) > 1:
        best_step = np.argmin(deviations)
        end_step = best_step + 1  # Include the best step
        print(f"\n[Plotting up to best step {best_step} with min deviation {deviations[best_step]:.2f}]")
    else:
        end_step = len(deviations)

    # Truncate data to best point
    deviations_plot = deviations[:end_step]
    params_history_plot = {k: v[:end_step] for k, v in result["params_history"].items()}
    rewards_plot = result.get("rewards_per_step", [])[:end_step-1] if end_step > 1 else []

    # Create figure with larger size and better spacing
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.92, bottom=0.08, left=0.08, right=0.95)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Font sizes
    TITLE_SIZE = 13
    LABEL_SIZE = 11
    TICK_SIZE = 10
    LEGEND_SIZE = 10

    # Plot 1: Total deviation over time
    steps = range(len(deviations_plot))
    ax1.plot(steps, deviations_plot, "b-", linewidth=2.5, label="Total Deviation")
    ax1.axhline(y=goal_threshold * 7, color="g", linestyle="--", linewidth=2, label=f"Goal (~{goal_threshold * 7:.0f})")
    ax1.set_xlabel("Step", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Total Deviation", fontsize=LABEL_SIZE)
    title_suffix = f" (truncated at step {best_step})" if truncate_at_best and end_step < len(deviations) else ""
    ax1.set_title(f"Total Deviation Over Time{title_suffix}", fontsize=TITLE_SIZE, fontweight='bold')
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.tick_params(axis='both', labelsize=TICK_SIZE)
    ax1.grid(True, alpha=0.3)
    if min(deviations_plot) > 0:
        ax1.set_yscale("log")

    # Plot 2: Main corrector parameters (A2, B2, A3, S3) over time
    main_keys = ["A2", "B2", "A3", "S3"]
    for key in main_keys:
        ax2.plot(steps, params_history_plot[key], label=key, linewidth=2)
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.axhline(y=goal_threshold, color="g", linestyle="--", alpha=0.5, linewidth=1.5)
    ax2.axhline(y=-goal_threshold, color="g", linestyle="--", alpha=0.5, linewidth=1.5)
    ax2.set_xlabel("Step", fontsize=LABEL_SIZE)
    ax2.set_ylabel("Parameter Value", fontsize=LABEL_SIZE)
    ax2.set_title("Main Correctors (A2, B2, A3, S3)", fontsize=TITLE_SIZE, fontweight='bold')
    ax2.legend(fontsize=LEGEND_SIZE, loc='best')
    ax2.tick_params(axis='both', labelsize=TICK_SIZE)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Secondary parameters (C1, A1, C3) over time
    secondary_keys = ["C1", "A1", "C3"]
    for key in secondary_keys:
        ax3.plot(steps, params_history_plot[key], label=key, linewidth=2)
    ax3.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax3.axhline(y=goal_threshold, color="g", linestyle="--", alpha=0.5, linewidth=1.5)
    ax3.axhline(y=-goal_threshold, color="g", linestyle="--", alpha=0.5, linewidth=1.5)
    ax3.set_xlabel("Step", fontsize=LABEL_SIZE)
    ax3.set_ylabel("Parameter Value", fontsize=LABEL_SIZE)
    ax3.set_title("Secondary Parameters (C1, A1, C3)", fontsize=TITLE_SIZE, fontweight='bold')
    ax3.legend(fontsize=LEGEND_SIZE, loc='best')
    ax3.tick_params(axis='both', labelsize=TICK_SIZE)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Per-step rewards
    if rewards_plot:
        ax4.plot(range(1, len(rewards_plot) + 1), rewards_plot, "r-", linewidth=2.5, label="Step Reward")
        ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax4.set_xlabel("Step", fontsize=LABEL_SIZE)
        ax4.set_ylabel("Reward", fontsize=LABEL_SIZE)
        ax4.set_title("Per-Step Rewards", fontsize=TITLE_SIZE, fontweight='bold')
        ax4.legend(fontsize=LEGEND_SIZE)
        ax4.tick_params(axis='both', labelsize=TICK_SIZE)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.set_visible(False)

    best_dev = min(deviations)
    fig.suptitle(f"Episode Analysis - Best Deviation: {best_dev:.2f} at step {np.argmin(deviations)}  |  Total Reward: {result['reward']:.4f}",
                 fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_action_table(result, save_path: str = None, goal_threshold: float = 50.0, truncate_at_best: bool = True):
    """Plot action log as a table figure, truncated at best deviation point."""
    deviations = result["deviations"]
    action_details = result.get("action_details", [])
    rewards = result.get("rewards_per_step", [])
    params_history = result.get("params_history", {})

    # Find the best (minimum deviation) point
    if truncate_at_best and len(deviations) > 1:
        best_step = np.argmin(deviations)
        end_step = best_step + 1
    else:
        end_step = len(deviations)

    # Get parameter histories
    c1_hist = params_history.get("C1", [])
    a1_hist = params_history.get("A1", [])
    a2_hist = params_history.get("A2", [])
    b2_hist = params_history.get("B2", [])
    a3_hist = params_history.get("A3", [])
    s3_hist = params_history.get("S3", [])
    c3_hist = params_history.get("C3", [])

    # Build table data
    table_data = []
    col_labels = ["Step", "Action", "C1", "A1", "A2", "B2", "A3", "S3", "C3", "Deviation", "Reward"]

    # Initial row
    if c1_hist and a1_hist:
        table_data.append([
            "Init", "",
            f"{c1_hist[0]:.1f}", f"{a1_hist[0]:.1f}", f"{a2_hist[0]:.1f}",
            f"{b2_hist[0]:.1f}", f"{a3_hist[0]:.1f}", f"{s3_hist[0]:.1f}",
            f"{c3_hist[0]:.1f}", f"{deviations[0]:.1f}", ""
        ])

    # Action rows (up to best step)
    cumulative_reward = 0.0
    for i in range(min(end_step - 1, len(action_details))):
        act = action_details[i]
        dev = deviations[i + 1] if i + 1 < len(deviations) else 0
        rew = rewards[i] if i < len(rewards) else 0
        cumulative_reward += rew

        action_str = format_action(act)
        c1 = c1_hist[i + 1] if i + 1 < len(c1_hist) else 0
        a1 = a1_hist[i + 1] if i + 1 < len(a1_hist) else 0
        a2 = a2_hist[i + 1] if i + 1 < len(a2_hist) else 0
        b2 = b2_hist[i + 1] if i + 1 < len(b2_hist) else 0
        a3 = a3_hist[i + 1] if i + 1 < len(a3_hist) else 0
        s3 = s3_hist[i + 1] if i + 1 < len(s3_hist) else 0
        c3 = c3_hist[i + 1] if i + 1 < len(c3_hist) else 0

        table_data.append([
            str(i + 1), action_str,
            f"{c1:.1f}", f"{a1:.1f}", f"{a2:.1f}",
            f"{b2:.1f}", f"{a3:.1f}", f"{s3:.1f}",
            f"{c3:.1f}", f"{dev:.1f}", f"{rew:.4f}"
        ])

    # Calculate figure height based on number of rows
    n_rows = len(table_data)
    fig_height = max(6, 1.5 + n_rows * 0.5)
    fig_width = 16

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color header row
    for j, label in enumerate(col_labels):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Color alternating rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    # Highlight best deviation row (last row before truncation)
    if truncate_at_best and len(table_data) > 1:
        for j in range(len(col_labels)):
            table[(len(table_data), j)].set_facecolor('#C6EFCE')

    best_dev = min(deviations)
    title = f"Action Log (Truncated at Best Step {np.argmin(deviations)}) - Best Deviation: {best_dev:.2f}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        # Save with different name for table
        table_save_path = save_path.replace('.png', '_table.png') if '.png' in save_path else save_path + '_table.png'
        plt.savefig(table_save_path, dpi=150, bbox_inches="tight")
        print(f"Action table saved to: {table_save_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison(trained_results, random_results, save_path: str = None):
    """Plot comparison between trained and random policies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Reward distribution
    ax1 = axes[0, 0]
    ax1.boxplot([
        [r["reward"] for r in trained_results],
        [r["reward"] for r in random_results]
    ], labels=["Trained", "Random"])
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Reward Distribution")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final deviation distribution
    ax2 = axes[0, 1]
    ax2.boxplot([
        [r["final_deviation"] for r in trained_results],
        [r["final_deviation"] for r in random_results]
    ], labels=["Trained", "Random"])
    ax2.set_ylabel("Final Deviation")
    ax2.set_title("Final Deviation Distribution")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Episode length distribution
    ax3 = axes[1, 0]
    ax3.boxplot([
        [r["length"] for r in trained_results],
        [r["length"] for r in random_results]
    ], labels=["Trained", "Random"])
    ax3.set_ylabel("Episode Length")
    ax3.set_title("Episode Length Distribution")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Success rate
    ax4 = axes[1, 1]
    trained_success = sum(1 for r in trained_results if r["goal_achieved"]) / len(trained_results) * 100
    random_success = sum(1 for r in random_results if r["goal_achieved"]) / len(random_results) * 100
    bars = ax4.bar(["Trained", "Random"], [trained_success, random_success], color=["blue", "orange"])
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_title("Goal Achievement Rate")
    ax4.set_ylim(0, 100)
    for bar, val in zip(bars, [trained_success, random_success]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.1f}%", ha="center", va="bottom")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model on CorrectorPlayEnv v3")

    # Model arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--vec-normalize-path", type=str, default=None, help="Path to VecNormalize stats")

    # Environment arguments
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--goal-threshold", type=float, default=50.0, help="Goal threshold for all params")
    parser.add_argument("--max-deviation", type=float, default=100000.0, help="Early stop if total deviation exceeds this")
    parser.add_argument("--obs-clip", type=float, default=10000.0, help="Clip observations to prevent overflow")

    # Evaluation arguments
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (setting_seed)")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Use stochastic actions")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--compare-random", action="store_true", help="Compare with random policy")
    parser.add_argument("--plot", action="store_true", help="Plot best episode trajectory")
    parser.add_argument("--plot-table", action="store_true", help="Plot action log as table")
    parser.add_argument("--plot-save", type=str, default=None, help="Save plot to file")
    parser.add_argument("--plot-comparison", action="store_true", help="Plot comparison charts")
    parser.add_argument("--plot-full", action="store_true", help="Plot full episode (don't truncate at best point)")

    args = parser.parse_args()

    env_kwargs = {
        "max_steps": args.max_steps,
        "goal_threshold": args.goal_threshold,
        "max_deviation": args.max_deviation,
        "obs_clip": args.obs_clip,
        "couple_prob_pct": 0.0,  # Not used since user_gamma is provided
        "fail_prob_range": (0.0, 0.0),  # No action failures
        "user_gamma": USER_GAMMA,
        "user_beta": {},  # Let env use default
        "user_sigma": USER_SIGMA,
        "init_ranges": INIT_RANGES,
    }

    # Load model (pass env_kwargs for VecNormalize compatibility)
    model, vec_normalize = load_model(args.model_path, args.vec_normalize_path, **env_kwargs)

    if args.compare_random:
        trained_results, random_results = compare_with_random(
            model, args.n_episodes, args.seed, vec_normalize, **env_kwargs
        )
        results = trained_results

        if args.plot_comparison:
            save_path = args.plot_save.replace(".png", "_comparison.png") if args.plot_save else None
            plot_comparison(trained_results, random_results, save_path)
    else:
        print("\n" + "=" * 70)
        print("Evaluating trained model (v3)")
        print("=" * 70)
        results = evaluate(
            model, args.n_episodes, args.seed, vec_normalize,
            args.deterministic, args.render, **env_kwargs
        )

        # Print summary
        if results:
            rewards = [r["reward"] for r in results]
            success_rate = sum(1 for r in results if r["goal_achieved"]) / len(results)
            print(f"\nMean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
            print(f"Success Rate: {success_rate * 100:.1f}%")
            print(f"Mean Final Deviation: {np.mean([r['final_deviation'] for r in results]):.2f}")
        else:
            print("\nNo valid episodes to summarize (all had NaN/Inf)")

    # Plot best episode (highest reward)
    if (args.plot or args.plot_save or args.plot_table) and results:
        # Find best episode by reward
        best_idx = max(range(len(results)), key=lambda i: results[i]["reward"])
        best_result = results[best_idx]

        if "deviations" in best_result:  # Only if we have full trajectory data
            print(f"\nPlotting best episode (Episode {best_idx + 1}, Reward: {best_result['reward']:.2f})")
            truncate = not args.plot_full  # Default: truncate at best point

            # Plot trajectory graphs
            if args.plot or args.plot_save:
                plot_episode(best_result, args.plot_save, args.goal_threshold, truncate_at_best=truncate)

            # Plot action table
            if args.plot_table:
                plot_action_table(best_result, args.plot_save, args.goal_threshold, truncate_at_best=truncate)

    elif (args.plot or args.plot_save or args.plot_table) and not results:
        print("\nNo valid episodes to plot")


if __name__ == "__main__":
    main()
