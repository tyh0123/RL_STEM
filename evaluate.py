import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from CorrectorPlayEnv import CorrectorPlayEnv


def load_model(model_path: str, vec_normalize_path: str = None):
    """Load trained model and optional VecNormalize stats."""
    model = PPO.load(model_path)
    print(f"Loaded model from: {model_path}")

    vec_normalize = None
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        vec_normalize = VecNormalize.load(vec_normalize_path, DummyVecEnv([lambda: CorrectorPlayEnv()]))
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        print(f"Loaded VecNormalize from: {vec_normalize_path}")

    return model, vec_normalize


def evaluate_episode(env, model, vec_normalize=None, deterministic=True, render=False):
    """Run a single evaluation episode and collect metrics."""
    obs, info = env.reset()

    if vec_normalize is not None:
        # Wrap obs for normalization
        obs = vec_normalize.normalize_obs(obs)

    episode_reward = 0.0
    episode_length = 0
    deviations = [info.get("init_main_deviation", 0.0)]
    actions_taken = []
    params_history = {k: [env.params[k]] for k in env.KEYS}

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        if vec_normalize is not None:
            obs = vec_normalize.normalize_obs(obs)

        episode_reward += reward
        episode_length += 1
        deviations.append(info.get("main_deviation", 0.0))
        actions_taken.append(action)

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
        "deviations": deviations,
        "actions": actions_taken,
        "params_history": params_history,
    }


def evaluate(model, n_episodes: int = 10, seed: int = 0, vec_normalize=None,
             deterministic: bool = True, render: bool = False, **env_kwargs):
    """Evaluate model over multiple episodes."""
    results = []

    for ep in range(n_episodes):
        env = CorrectorPlayEnv(
            static_seed=seed + ep * 100,
            dynamic_seed=seed + ep * 100 + 50,
            render_mode="human" if render else None,
            **env_kwargs
        )

        result = evaluate_episode(env, model, vec_normalize, deterministic, render)
        results.append(result)

        status = "SUCCESS" if result["goal_achieved"] else "TIMEOUT"
        print(f"Episode {ep + 1:3d}: {status} | "
              f"Reward: {result['reward']:8.2f} | "
              f"Length: {result['length']:4d} | "
              f"Final Dev: {result['final_deviation']:8.2f}")

        env.close()

    return results


def compare_with_random(model, n_episodes: int = 10, seed: int = 0,
                        vec_normalize=None, **env_kwargs):
    """Compare trained model with random policy."""
    print("\n" + "=" * 60)
    print("Evaluating TRAINED model (deterministic)")
    print("=" * 60)
    trained_results = evaluate(
        model, n_episodes, seed, vec_normalize,
        deterministic=True, render=False, **env_kwargs
    )

    print("\n" + "=" * 60)
    print("Evaluating RANDOM policy")
    print("=" * 60)
    random_results = []

    for ep in range(n_episodes):
        env = CorrectorPlayEnv(
            static_seed=seed + ep * 100,
            dynamic_seed=seed + ep * 100 + 50,
            **env_kwargs
        )

        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
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
            "final_deviation": info.get("main_deviation", 0.0),
        })

        status = "SUCCESS" if info.get("goal_achieved", False) else "TIMEOUT"
        print(f"Episode {ep + 1:3d}: {status} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Length: {episode_length:4d} | "
              f"Final Dev: {info.get('main_deviation', 0.0):8.2f}")

        env.close()

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    trained_rewards = [r["reward"] for r in trained_results]
    random_rewards = [r["reward"] for r in random_results]
    trained_success = sum(1 for r in trained_results if r["goal_achieved"])
    random_success = sum(1 for r in random_results if r["goal_achieved"])

    print(f"\n{'Metric':<25} {'Trained':>15} {'Random':>15}")
    print("-" * 55)
    print(f"{'Mean Reward':<25} {np.mean(trained_rewards):>15.2f} {np.mean(random_rewards):>15.2f}")
    print(f"{'Std Reward':<25} {np.std(trained_rewards):>15.2f} {np.std(random_rewards):>15.2f}")
    print(f"{'Success Rate':<25} {trained_success/n_episodes*100:>14.1f}% {random_success/n_episodes*100:>14.1f}%")
    print(f"{'Mean Final Deviation':<25} {np.mean([r['final_deviation'] for r in trained_results]):>15.2f} {np.mean([r['final_deviation'] for r in random_results]):>15.2f}")

    return trained_results, random_results


def plot_episode(result, save_path: str = None):
    """Plot deviation and parameter trajectories for a single episode."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Main deviation over time
    ax1 = axes[0]
    steps = range(len(result["deviations"]))
    ax1.plot(steps, result["deviations"], "b-", linewidth=2, label="Main Deviation")
    ax1.axhline(y=10.0, color="g", linestyle="--", label="Goal Threshold")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Main Deviation (|A2| + |B2| + |A3| + |S3|)")
    ax1.set_title(f"Episode: Reward={result['reward']:.2f}, Length={result['length']}, Goal={'Achieved' if result['goal_achieved'] else 'Not Achieved'}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Main corrector parameters over time
    ax2 = axes[1]
    main_keys = ["A2", "B2", "A3", "S3"]
    for key in main_keys:
        ax2.plot(steps, result["params_history"][key], label=key, linewidth=1.5)
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.axhline(y=10, color="g", linestyle="--", alpha=0.5)
    ax2.axhline(y=-10, color="g", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Parameter Value")
    ax2.set_title("Main Corrector Parameters Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model on CorrectorPlayEnv")

    # Model arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--vec-normalize-path", type=str, default=None, help="Path to VecNormalize stats")

    # Environment arguments
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--noise-sigma", type=float, default=50.0, help="Gaussian noise sigma")
    parser.add_argument("--couple-prob", type=float, default=0.5, help="Coupling probability")
    parser.add_argument("--goal-threshold", type=float, default=10.0, help="Goal threshold")

    # Evaluation arguments
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for evaluation")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Use stochastic actions")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--compare-random", action="store_true", help="Compare with random policy")
    parser.add_argument("--plot", action="store_true", help="Plot first episode trajectory")
    parser.add_argument("--plot-save", type=str, default=None, help="Save plot to file")

    args = parser.parse_args()

    # Load model
    model, vec_normalize = load_model(args.model_path, args.vec_normalize_path)

    env_kwargs = {
        "max_steps": args.max_steps,
        "noise_sigma": args.noise_sigma,
        "couple_prob_pct": args.couple_prob,
        "goal_threshold": args.goal_threshold,
    }

    if args.compare_random:
        trained_results, random_results = compare_with_random(
            model, args.n_episodes, args.seed, vec_normalize, **env_kwargs
        )
        results = trained_results
    else:
        print("\n" + "=" * 60)
        print("Evaluating trained model")
        print("=" * 60)
        results = evaluate(
            model, args.n_episodes, args.seed, vec_normalize,
            args.deterministic, args.render, **env_kwargs
        )

        # Print summary
        rewards = [r["reward"] for r in results]
        success_rate = sum(1 for r in results if r["goal_achieved"]) / len(results)
        print(f"\nMean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"Success Rate: {success_rate * 100:.1f}%")

    # Plot first episode
    if args.plot or args.plot_save:
        plot_episode(results[0], args.plot_save)


if __name__ == "__main__":
    main()
