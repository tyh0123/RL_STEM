import os
import argparse
from datetime import datetime

import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.utils import set_random_seed

from CorrectorPlayEnv import CorrectorPlayEnv


def make_env(rank: int, seed: int = 0, **env_kwargs):
    """Factory function to create a single environment instance."""
    def _init():
        env = CorrectorPlayEnv(
            static_seed=seed + rank,
            dynamic_seed=seed + rank + 1000,
            **env_kwargs
        )
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init


def create_vec_env(n_envs: int, seed: int = 0, use_subproc: bool = False, **env_kwargs):
    """Create vectorized environment with optional normalization."""
    env_fns = [make_env(rank=i, seed=seed, **env_kwargs) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    return vec_env


def train(args):
    """Main training function."""
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"ppo_corrector_{timestamp}"
    exp_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")

    # Environment kwargs
    env_kwargs = {
        "max_steps": args.max_steps,
        "noise_sigma": args.noise_sigma,
        "couple_prob_pct": args.couple_prob,
        "goal_threshold": args.goal_threshold,
        "render_mode": None,
    }

    # Build config dict for wandb
    config = {
        # Environment
        "max_steps": args.max_steps,
        "noise_sigma": args.noise_sigma,
        "couple_prob": args.couple_prob,
        "goal_threshold": args.goal_threshold,
        # Training
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "normalize": args.normalize,
        "seed": args.seed,
        # PPO hyperparameters
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        # Network architecture
        "policy_arch": args.policy_arch,
        "value_arch": args.value_arch,
    }

    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=exp_name,
        config=config,
        sync_tensorboard=False,
        monitor_gym=True,
        save_code=True,
    )

    print(f"Wandb run: {run.url}")

    # Create training environment
    print(f"Creating {args.n_envs} parallel environments...")
    train_env = create_vec_env(
        n_envs=args.env_kwargs,
        seed=args.seed,
        use_subproc=args.use_subproc,
        **env_kwargs
    )

    # Optionally wrap with VecNormalize for observation/reward normalization
    if args.normalize:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        print("Using VecNormalize for obs/reward normalization")

    # Create evaluation environment (single env, different seed)
    eval_env = create_vec_env(
        n_envs=1,
        seed=args.seed + 10000,
        use_subproc=False,
        **env_kwargs
    )
    if args.normalize:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,  # Don't normalize reward for eval (see true rewards)
            clip_obs=10.0,
            training=False,  # Don't update stats during eval
        )

    # PPO hyperparameters
    ppo_kwargs = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "verbose": 1,
    }

    # Policy network architecture
    policy_kwargs = {
        "net_arch": dict(pi=args.policy_arch, vf=args.value_arch),
    }

    print("\nPPO Hyperparameters:")
    for k, v in ppo_kwargs.items():
        print(f"  {k}: {v}")
    print(f"\nPolicy architecture: pi={args.policy_arch}, vf={args.value_arch}")

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        **ppo_kwargs
    )

    print(f"\nModel created with {sum(p.numel() for p in model.policy.parameters())} parameters")

    # Callbacks
    callbacks = []

    # Wandb callback
    wandb_callback = WandbCallback(
        gradient_save_freq=args.gradient_save_freq,
        model_save_path=os.path.join(exp_dir, "wandb_models"),
        model_save_freq=args.checkpoint_freq,
        verbose=2,
    )
    callbacks.append(wandb_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(exp_dir, "best_model"),
        log_path=os.path.join(exp_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(exp_dir, "checkpoints"),
        name_prefix="ppo_corrector",
        save_replay_buffer=False,
        save_vecnormalize=args.normalize,
    )
    callbacks.append(checkpoint_callback)

    callback_list = CallbackList(callbacks)

    # Train
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    print("=" * 60)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback_list,
        progress_bar=True,
    )

    print("=" * 60)
    print("Training complete!")

    # Save final model
    final_model_path = os.path.join(exp_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Save VecNormalize statistics if used
    if args.normalize:
        vec_norm_path = os.path.join(exp_dir, "vec_normalize.pkl")
        train_env.save(vec_norm_path)
        print(f"VecNormalize stats saved to: {vec_norm_path}")

    # Log final model as artifact to wandb
    artifact = wandb.Artifact(
        name=f"model-{run.id}",
        type="model",
        description="Final trained PPO model",
    )
    artifact.add_file(final_model_path + ".zip")
    if args.normalize:
        artifact.add_file(vec_norm_path)
    run.log_artifact(artifact)

    # Cleanup
    train_env.close()
    eval_env.close()

    # Finish wandb run
    wandb.finish()

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Train PPO on CorrectorPlayEnv")

    # Environment arguments
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--noise-sigma", type=float, default=50.0, help="Gaussian noise sigma")
    parser.add_argument("--couple-prob", type=float, default=0.5, help="Coupling probability")
    parser.add_argument("--goal-threshold", type=float, default=10.0, help="Goal threshold")

    # Training arguments
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--use-subproc", action="store_true", help="Use SubprocVecEnv instead of DummyVecEnv")
    parser.add_argument("--normalize", action="store_true", help="Use VecNormalize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")

    # Network architecture
    parser.add_argument("--policy-arch", type=int, nargs="+", default=[64, 64], help="Policy network architecture")
    parser.add_argument("--value-arch", type=int, nargs="+", default=[64, 64], help="Value network architecture")

    # Logging and saving
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Episodes per evaluation")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Checkpoint frequency")
    parser.add_argument("--gradient-save-freq", type=int, default=0, help="Gradient save frequency for wandb (0=disabled)")

    # Wandb arguments
    parser.add_argument("--wandb-project", type=str, default="corrector-ppo", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity (team/user)")

    args = parser.parse_args()

    # Print configuration
    print("\n" + "=" * 60)
    print("PPO Training for CorrectorPlayEnv")
    print("=" * 60)

    train(args)


if __name__ == "__main__":
    main()
