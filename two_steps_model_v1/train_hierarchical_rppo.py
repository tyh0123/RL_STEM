from datetime import datetime
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Tuple, Optional, Any
import csv, os
import torch as th
from collections import Counter
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

# -----------------------------
# 1) Import your simulator env
# -----------------------------
SIM_PATH = "CorrectorPlayEnv_v5.py"
# You can also replace with a normal import if it's in your PYTHONPATH.

import importlib.util

spec = importlib.util.spec_from_file_location("corrector_sim", SIM_PATH)
corrector_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corrector_sim)
CorrectorPlayEnv = corrector_sim.CorrectorPlayEnv


def time_tag():
    return datetime.now().strftime("%d-%m-%y-%H%M")


# -----------------------------
# 2) Utilities: action filtering
# -----------------------------
def build_action_indices(env: CorrectorPlayEnv) -> Dict[str, List[int]]:
    """
    Build action index sets for different modes by inspecting env.action_table.
    Your simulator action_table contains dicts like:
      {"type": "pct_button", "target": "B2", "pct": 50, "key": ("B2", 50)}
      {"type": "step", "target": "C1", "step": 10.0, "dir": +1, ...}
    """
    c1a1_idx = []
    high_idx = []

    for i, act in enumerate(env.action_table):
        tgt = act["target"]

        # C1A1 mode: only C1 step +/- and A1 pct buttons
        if tgt in ["C1", "A1"]:
            c1a1_idx.append(i)

        # HighOrder mode: MAIN_KEYS + C3 (and you can decide whether to include A1 here)
        if tgt in ["A2", "B2", "A3", "S3", "C3"]:
            high_idx.append(i)

    # Safety check
    assert len(c1a1_idx) > 0, "No actions found for C1A1_mode"
    assert len(high_idx) > 0, "No actions found for HighOrder_mode"

    return {"C1A1": c1a1_idx, "HIGH": high_idx}


# -----------------------------
# 3) Mode-specific env wrappers
# -----------------------------
class ActionSubsetWrapper(gym.Wrapper):
    """
    Restrict action space to a subset of indices of the base env.action_space.
    action a' in [0..n_sub-1] is mapped to base action a = subset[a'].
    """

    def __init__(self, env: CorrectorPlayEnv, subset: List[int]):
        super().__init__(env)
        self.subset = list(subset)
        self.action_space = spaces.Discrete(len(self.subset))
        self.observation_space = env.observation_space

    def step(self, action):
        a_sub = int(action)
        base_action = self.subset[a_sub]
        obs, r, terminated, truncated, info = self.env.step(base_action)
        info["base_action_idx"] = int(base_action)
        # 记录：子动作索引 + 原始动作索引（用于翻译到 CEOS action_table）
        info["base_action_idx"] = base_action

        return obs, r, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class C1A1RewardWrapper(gym.Wrapper):
    """
    Shape reward to strongly encourage minimizing |C1| and |A1| quickly and stably.
    Adds:
      - potential-based shaping on (|C1|, |A1|)
      - small step penalty
      - terminal bonus if gate achieved
    """

    def __init__(self, env: gym.Env, c1_gate: float = 1.0, a1_gate: float = 2.5,
                 c1_weight: float = 2.0, a1_weight: float = 2.0,
                 step_penalty: float = 0.01, terminal_bonus: float = 5.0, gamma: float = 0.99):
        super().__init__(env)
        self.c1_gate = float(c1_gate)
        self.a1_gate = float(a1_gate)
        self.c1_weight = float(c1_weight)
        self.a1_weight = float(a1_weight)
        self.step_penalty = float(step_penalty)
        self.terminal_bonus = float(terminal_bonus)
        self.gamma = float(gamma)
        self._last_phi = None

    def _phi(self, obs: np.ndarray) -> float:
        # obs order: env.KEYS = ["C1","A1","A2","B2","A3","S3","C3"]
        c1 = abs(float(obs[0]))
        a1 = abs(float(obs[1]))
        return -(self.c1_weight * c1 + self.a1_weight * a1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)

        # ---- 物理量 ----
        abs_c1 = abs(float(obs[0]))
        abs_a1 = abs(float(obs[1]))

        phi = self._phi(obs)
        shaped = (self.gamma * phi - self._last_phi) - self.step_penalty
        self._last_phi = phi

        # Gate achieved?
        gate_ok = (abs_c1 < self.c1_gate) and (abs_a1 < self.a1_gate)
        if gate_ok:
            shaped += self.terminal_bonus

        # Keep explosion penalty from base env if it happened
        # Base env uses reward=-1000 for explosion; keep that dominant:
        if info.get("deviation_exploded", False):
            shaped = -1000.0
        if gate_ok:
            terminated = True

        # ---- 写入 info（关键）----
        info["abs_C1"] = abs_c1
        info["abs_A1"] = abs_a1
        info["gate_ok"] = gate_ok
        # We do not force terminate here; we let training see more transitions,
        # but you can optionally terminate when gate_ok to make it a pure subtask.
        return obs, shaped, terminated, truncated, info


class HighOrderRewardWrapper(gym.Wrapper):
    """
    High-order mode:
      - potential-based shaping on high-order terms
      - soft penalty if C1/A1 drift away
      - NEW: gate + terminal_bonus + force terminated when gate achieved
    """

    def __init__(
            self,
            env: gym.Env,
            c1a1_soft: float = 120.0,  # 软约束用（你原来就有）
            step_penalty: float = 0.01,
            gamma: float = 0.99,

            # ---- NEW: gate thresholds (你可以按 CEOS 经验改) ----
            a2_gate: float = 30.0,
            b2_gate: float = 20.0,
            a3_gate: float = 350.0,
            s3_gate: float = 350.0,
            c3_gate: float = 500.0,

            # 可选：要求 C1/A1 也在一个可测范围（建议先 False）
            require_c1a1_gate: bool = False,
            c1_gate: float = 1.0,
            a1_gate: float = 2.5,

            terminal_bonus: float = 5.0,
    ):
        super().__init__(env)
        self.c1a1_soft = float(c1a1_soft)
        self.step_penalty = float(step_penalty)
        self.gamma = float(gamma)
        self._last_phi = None

        self.a2_gate = float(a2_gate)
        self.b2_gate = float(b2_gate)
        self.a3_gate = float(a3_gate)
        self.s3_gate = float(s3_gate)
        self.c3_gate = float(c3_gate)

        self.require_c1a1_gate = bool(require_c1a1_gate)
        self.c1_gate = float(c1_gate)
        self.a1_gate = float(a1_gate)

        self.terminal_bonus = float(terminal_bonus)

    def _phi(self, obs: np.ndarray) -> float:
        c1 = abs(float(obs[0]))
        a1 = abs(float(obs[1]))
        a2 = abs(float(obs[2]))
        b2 = abs(float(obs[3]))
        a3 = abs(float(obs[4]))
        s3 = abs(float(obs[5]))
        c3 = abs(float(obs[6]))

        def soft_bound(x, bound):
            return max(0.0, x - bound)

        c1_pen = soft_bound(c1, self.c1a1_soft)
        a1_pen = soft_bound(a1, self.c1a1_soft)

        cost = (1.0 * a2 + 1.0 * b2 + 0.7 * a3 + 0.7 * s3 + 0.5 * c3) + (2.0 * c1_pen + 2.0 * a1_pen)
        return -cost

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)

        # ----- shaping reward (same as before) -----
        phi = self._phi(obs)
        shaped = (self.gamma * phi - self._last_phi) - self.step_penalty
        self._last_phi = phi

        if info.get("deviation_exploded", False):
            shaped = -1000.0

        # ----- write abs metrics into info (你原来就有) -----
        abs_c1 = abs(float(obs[0]))
        abs_a1 = abs(float(obs[1]))
        abs_a2 = abs(float(obs[2]))
        abs_b2 = abs(float(obs[3]))
        abs_a3 = abs(float(obs[4]))
        abs_s3 = abs(float(obs[5]))
        abs_c3 = abs(float(obs[6]))

        info["abs_C1"] = abs_c1
        info["abs_A1"] = abs_a1
        info["abs_A2"] = abs_a2
        info["abs_B2"] = abs_b2
        info["abs_A3"] = abs_a3
        info["abs_S3"] = abs_s3
        info["abs_C3"] = abs_c3

        # ----- NEW: high-order gate -----
        high_gate_ok = (
                (abs_a2 < self.a2_gate) and
                (abs_b2 < self.b2_gate) and
                (abs_a3 < self.a3_gate) and
                (abs_s3 < self.s3_gate) and
                (abs_c3 < self.c3_gate)
        )

        if self.require_c1a1_gate:
            gate_ok = high_gate_ok and (abs_c1 < self.c1_gate) and (abs_a1 < self.a1_gate)
        else:
            gate_ok = high_gate_ok

        info["gate_ok"] = gate_ok
        info["high_gate_ok"] = high_gate_ok

        # 达标：给 bonus + 强制终止（像 C1A1 一样）
        if gate_ok and (not info.get("deviation_exploded", False)):
            shaped += self.terminal_bonus
            terminated = True

        return obs, shaped, terminated, truncated, info


# -----------------------------
# 5) VecEnv factories
# -----------------------------
def make_base_env(seed: int, env_kwargs):
    def _init():
        env = CorrectorPlayEnv(**env_kwargs)
        env.reset(seed=seed)
        return env

    return _init


def make_low_env(mode: str, seed: int, action_idx: List[int],
                 env_kwargs,
                 c1_gate=1.0, a1_gate=2.5,
                 c1_weight=2.0, a1_weight=2.0,

                 a2_gate=30.0,
                 b2_gate=20.0,
                 a3_gate=350.0,
                 s3_gate=350.0,
                 c3_gate=500.0):
    def _init():
        if mode == "C1A1":
            env = CorrectorPlayEnv(explode_keys=["C1", "A1"], **env_kwargs)
        else:
            env = CorrectorPlayEnv(**env_kwargs)

        env.reset(seed=seed)
        env = ActionSubsetWrapper(env, action_idx)
        if mode == "C1A1":
            env = C1A1RewardWrapper(env, c1_gate=c1_gate, a1_gate=a1_gate, c1_weight=c1_weight, a1_weight=a1_weight)
        else:
            env = HighOrderRewardWrapper(env,
                                         a2_gate=a2_gate,
                                         b2_gate=b2_gate,
                                         a3_gate=a3_gate,
                                         s3_gate=s3_gate,
                                         c3_gate=c3_gate,
                                         require_c1a1_gate=False)

        return env

    return _init


# -----------------------------
# 6) Training
# -----------------------------
def train_low_level(out_dir,
                    env_kwargs,
                    total_steps_c1a1=100_000,
                    total_steps_high=200_000,
                    c1_gate=1.0, a1_gate=2.5,
                    c1_weight=2.0,
                    a1_weight=2.0,
                    n_envs=8,
                    seed=0):
    os.makedirs(out_dir, exist_ok=True)
    set_random_seed(seed)

    # Build a temporary env to read action_table and define subsets
    env_kwargs['setting_seed'] = seed
    tmp = CorrectorPlayEnv(**env_kwargs)
    idx = build_action_indices(tmp)

    idx_c1a1 = idx["C1A1"]
    idx_high = idx["HIGH"]

    action_table = tmp.action_table
    trace_cb_c1a1 = EpisodeTraceToCSVCallback(
        out_dir=os.path.join(out_dir, "traces"),
        action_table=action_table,
        prefix="c1a1_trace",
        sample_every_episodes=10,  # record a whole episode every 200 episodes
    )

    # action_table = tmp.action_table
    # high_trace_cb = EpisodeTraceToCSVCallback(
    #     out_dir=os.path.join(out_dir, "traces_high"),
    #     action_table=action_table,
    #     prefix="high_trace",
    #     sample_every_episodes=10,  # 你要的 N：每 200 个 episode 记录 1 个
    #     record_dims=7,  # 记录全部像差
    # )

    # Vectorized envs
    c1a1_vec = SubprocVecEnv([
        make_low_env("C1A1", seed + i, idx_c1a1, c1_gate=c1_gate, a1_gate=a1_gate,
                     c1_weight=c1_weight, a1_weight=a1_weight, env_kwargs=env_kwargs)
        for i in range(n_envs)
    ])
    # high_vec = SubprocVecEnv([
    #     make_low_env("HIGH", seed + 100 + i, idx_high,
    #                  a2_gate=30.0,
    #                  b2_gate=20.0,
    #                  a3_gate=500.0,
    #                  s3_gate=500.0,
    #                  c3_gate=500.0,
    #                  **env_kwargs)
    #     for i in range(n_envs)
    # ])

    c1a1_vec = VecMonitor(c1a1_vec, filename=os.path.join(out_dir, "monitor_c1a1.csv"))
    # high_vec = VecMonitor(high_vec, filename=os.path.join(out_dir, "monitor_high.csv"))

    # Recurrent PPO configs (reasonable defaults for your noisy env)
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        lstm_hidden_size=128,
        n_lstm_layers=1,
        shared_lstm=False,
    )

    c1a1_model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=c1a1_vec,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=os.path.join(out_dir, "tb"),
        device='cuda:0'
    )

    # high_model = RecurrentPPO(
    #     policy="MlpLstmPolicy",
    #     env=high_vec,
    #     learning_rate=3e-4,
    #     n_steps=256,
    #     batch_size=256,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.01,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,
    #     seed=seed + 1,
    #     tensorboard_log=os.path.join(out_dir, "tb"),
    #     device='cuda:0'
    # )

    # checkpoints
    ckpt_c1a1 = CheckpointCallback(save_freq=50_000, save_path=out_dir, name_prefix="c1a1_rppo")
    # ckpt_high = CheckpointCallback(save_freq=50_000, save_path=out_dir, name_prefix="high_rppo")
    c1a1_metrics_cb = C1A1MetricsCallback()
    tmp = CorrectorPlayEnv()

    print("\n=== Train low-level C1A1 policy ===")
    c1a1_model.learn(
        total_timesteps=total_steps_c1a1,
        callback=[ckpt_c1a1, c1a1_metrics_cb, trace_cb_c1a1]
    )

    tag = time_tag()
    c1a1_path = os.path.join(out_dir, f"c1a1_rppo_{tag}.zip")
    c1a1_model.save(c1a1_path)

    # print("\n=== Train low-level HighOrder policy ===")
    # high_model.learn(total_timesteps=total_steps_high, callback=[ckpt_high, high_trace_cb])
    # high_path = os.path.join(out_dir, "high_rppo_final.zip")
    # high_model.save(high_path)

    # c1a1_vec.close()
    # high_vec.close()

    # return c1a1_path, high_path, idx_c1a1, idx_high
    return c1a1_path, idx_c1a1


class AlternatingHighEnv(gym.Env):
    """
    Scheme B:
      Each env.step(action_seq) does:
        1) Run C1A1 policy until gate (or n_c1a1_max)
        2) Execute a SEQUENCE of up to n_high HIGH actions (each can be different)
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            base_env,
            idx_c1a1: list,
            idx_high: list,
            c1a1_policy,
            c1_gate: float = 10.0,
            a1_gate: float = 20.0,
            # NEW: high-order gates
            a2_gate: float = 30.0,
            b2_gate: float = 30.0,
            a3_gate: float = 800.0,
            s3_gate: float = 800.0,
            c3_gate: float = 500.0,
            # NEW: optional terminal bonus when high gates reached
            high_gate_bonus: float = 0.0,

            n_c1a1_max: int = 60,
            n_high: int = 3,
            meta_max_steps: int = 200,
            deterministic_c1a1: bool = True,
    ):
        super().__init__()
        self.env = base_env
        self.idx_c1a1 = list(idx_c1a1)
        self.idx_high = list(idx_high)
        self.c1a1_policy = c1a1_policy

        self.c1_gate = float(c1_gate)
        self.a1_gate = float(a1_gate)
        self.a2_gate = float(a2_gate)
        self.b2_gate = float(b2_gate)
        self.a3_gate = float(a3_gate)
        self.s3_gate = float(s3_gate)
        self.c3_gate = float(c3_gate)
        self.high_gate_bonus = float(high_gate_bonus)

        self.n_c1a1_max = int(n_c1a1_max)
        self.n_high = int(n_high)
        self.meta_max_steps = int(meta_max_steps)
        self.deterministic_c1a1 = bool(deterministic_c1a1)

        # ✅ Scheme B: one action = sequence of n_high HIGH sub-actions
        # Each element is in [0, len(idx_high)-1]
        self.action_space = spaces.MultiDiscrete([len(self.idx_high)] * self.n_high)

        self.observation_space = self.env.observation_space

        self._c1a1_state = None
        self._c1a1_episode_start = np.array([True], dtype=bool)
        self._meta_t = 0

    def _high_gate_ok(self, obs: np.ndarray) -> bool:
        return (
                abs(float(obs[2])) < self.a2_gate and
                abs(float(obs[3])) < self.b2_gate and
                abs(float(obs[4])) < self.a3_gate and
                abs(float(obs[5])) < self.s3_gate and
                abs(float(obs[6])) < self.c3_gate
        )

    def _obs(self):
        return self.env._obs().astype(np.float32)

    def _c1a1_gate_ok(self, obs):
        return (abs(float(obs[0])) < self.c1_gate) and (abs(float(obs[1])) < self.a1_gate)

    def _run_c1a1_until_gate(self, obs):
        exploded_any = False
        steps = 0
        last_info = {}
        c1a1_trace = []  # NEW

        while (not self._c1a1_gate_ok(obs)) and (steps < self.n_c1a1_max):
            obs_before = obs.copy()

            obs_batch = obs[None, :]  # (1,7)
            a_sub, self._c1a1_state = self.c1a1_policy.predict(
                obs_batch,
                state=self._c1a1_state,
                episode_start=self._c1a1_episode_start,
                deterministic=self.deterministic_c1a1,
            )
            self._c1a1_episode_start[:] = False

            a_sub = int(np.array(a_sub).ravel()[0])
            base_action_idx = int(self.idx_c1a1[a_sub])

            obs2, base_r, terminated, truncated, info = self.env.step(base_action_idx)
            info = dict(info) if info is not None else {}
            last_info = info

            obs = obs2.astype(np.float32)
            steps += 1

            exploded = bool(info.get("deviation_exploded", False))
            c1a1_trace.append({
                "c1a1_j": steps - 1,
                "action_sub_idx": a_sub,
                "base_action_idx": base_action_idx,
                "C1_before": float(obs_before[0]), "C1_after": float(obs[0]),
                "A1_before": float(obs_before[1]), "A1_after": float(obs[1]),
                "A2_before": float(obs_before[2]), "A2_after": float(obs[2]),
                "B2_before": float(obs_before[3]), "B2_after": float(obs[3]),
                "A3_before": float(obs_before[4]), "A3_after": float(obs[4]),
                "S3_before": float(obs_before[5]), "S3_after": float(obs[5]),
                "C3_before": float(obs_before[6]), "C3_after": float(obs[6]),
                "deviation_exploded": exploded,
                "base_done": bool(terminated),
                "base_truncated": bool(truncated),
            })

            if exploded or terminated or truncated:
                exploded_any = exploded_any or exploded
                break

        return obs, exploded_any, steps, last_info, c1a1_trace

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)

        self._meta_t = 0
        self._c1a1_state = None
        self._c1a1_episode_start = np.array([True], dtype=bool)

        obs = obs.astype(np.float32)
        obs, exploded, steps_used, _, c1a1_trace = self._run_c1a1_until_gate(obs)

        info = dict(info) if info is not None else {}
        info.update({
            "c1a1_gate_ok": bool(self._c1a1_gate_ok(obs)),
            "c1a1_steps_used": int(steps_used),
            "deviation_exploded": bool(exploded),
            "c1a1_trace": c1a1_trace
        })

        return obs, info

    def step(self, action_seq_high):
        """
        action_seq_high: array-like of shape (n_high,), each in [0, len(idx_high)-1]
        """
        self.env.t = 0  # reset base env step counter
        self._meta_t += 1
        obs = self._obs()

        # 1) C1A1 until gate
        # obs, exploded_c1a1, steps_c1a1, _ = self._run_c1a1_until_gate(obs)
        obs, exploded_c1a1, steps_c1a1, _, c1a1_trace = self._run_c1a1_until_gate(obs)

        if exploded_c1a1:
            info = {
                "stage": "C1A1",
                "c1a1_steps_used": int(steps_c1a1),
                "c1a1_gate_ok": bool(self._c1a1_gate_ok(obs)),
                "deviation_exploded": True,
                "meta_t": int(self._meta_t),
            }
            return obs, -1000.0, True, False, info

        # meta truncation cap (high-level budget)
        if self._meta_t >= self.meta_max_steps:
            info = {
                "stage": "C1A1",
                "c1a1_steps_used": int(steps_c1a1),
                "c1a1_gate_ok": bool(self._c1a1_gate_ok(obs)),
                "deviation_exploded": False,
                "meta_t": int(self._meta_t),
            }
            return obs, -0.01, False, True, info

        # 2) Execute HIGH sequence (each step can use a different high action)
        action_seq_high = np.asarray(action_seq_high, dtype=int).reshape(-1)
        if action_seq_high.shape[0] != self.n_high:
            raise ValueError(f"Expected action_seq_high length {self.n_high}, got {action_seq_high.shape[0]}")

        total_r = 0.0
        exploded_any = False
        base_done_any = False
        base_trunc_any = False

        high_trace = []
        high_gate_ok = False
        for j in range(self.n_high):
            a_sub = int(action_seq_high[j])
            base_action_idx = int(self.idx_high[a_sub])

            obs_before = obs.copy()
            obs2, base_r, terminated, truncated, info = self.env.step(base_action_idx)
            info = dict(info) if info is not None else {}

            obs = obs2.astype(np.float32)
            exploded = bool(info.get("deviation_exploded", False))
            if exploded:
                exploded_any = True
            if bool(terminated):
                base_done_any = True
            if bool(truncated):
                base_trunc_any = True

            high_trace.append({
                "high_j": j,
                "action_sub_idx": a_sub,
                "base_action_idx": base_action_idx,
                "C1_before": float(obs_before[0]), "C1_after": float(obs[0]),
                "A1_before": float(obs_before[1]), "A1_after": float(obs[1]),
                "A2_before": float(obs_before[2]), "A2_after": float(obs[2]),
                "B2_before": float(obs_before[3]), "B2_after": float(obs[3]),
                "A3_before": float(obs_before[4]), "A3_after": float(obs[4]),
                "S3_before": float(obs_before[5]), "S3_after": float(obs[5]),
                "C3_before": float(obs_before[6]), "C3_after": float(obs[6]),
                "deviation_exploded": exploded,
                "base_done": bool(terminated),
                "base_truncated": bool(truncated),
            })

            # reward shaping: encourage reducing high-order magnitudes
            a2 = abs(float(obs[2]));
            b2 = abs(float(obs[3]))
            a3 = abs(float(obs[4]));
            s3 = abs(float(obs[5]));
            c3 = abs(float(obs[6]))
            total_r += -(a2 + b2 + 0.7 * a3 + 0.7 * s3 + 0.5 * c3) * 1e-3

            high_gate_ok = (not exploded_any) and self._high_gate_ok(obs)
            if high_gate_ok:
                # optional terminal bonus (keep modest)
                total_r += self.high_gate_bonus

            if exploded_any or base_done_any or base_trunc_any:
                break

        if exploded_any:
            total_r = -1000.0

        total_r -= 0.01  # time penalty

        info_out = {
            "stage": "HIGH",
            "meta_t": int(self._meta_t),
            "c1a1_steps_used": int(steps_c1a1),
            "c1a1_gate_ok": bool(self._c1a1_gate_ok(obs)),
            "c1a1_trace": c1a1_trace,
            "high_trace": high_trace,
            "deviation_exploded": bool(exploded_any),
            "c1a1_trace": c1a1_trace,
            # NEW: high gate status + abs metrics
            "high_gate_ok": bool(high_gate_ok),
            "abs_A2": abs(float(obs[2])),
            "abs_B2": abs(float(obs[3])),
            "abs_A3": abs(float(obs[4])),
            "abs_S3": abs(float(obs[5])),
            "abs_C3": abs(float(obs[6])),
        }

        terminated = bool(exploded_any or base_done_any or high_gate_ok)
        truncated = bool(base_trunc_any)

        return obs, float(total_r), terminated, truncated, info_out


def train_high_order_with_c1a1_guard(
        out_dir: str,
        c1a1_model_path: str,
        env_kwargs: dict,
        n_envs: int = 8,
        total_timesteps: int = 500_000,
        seed: int = 0,

        # workflow params
        n_c1a1_max: int = 60,
        n_high: int = 3,
        c1_gate: float = 10.0,
        a1_gate: float = 20.0,
        highorder_gate: dict | None = None,
        high_gate_bonus: float = 0.0,
        meta_max_steps: int = 200,
        deterministic_c1a1: bool = True,
        # PPO params
        device: str = "auto",
        n_steps: int = 256,
        batch_size: int = 256,
        n_epochs: int = 10,
        lr: float = 3e-4,
):
    os.makedirs(out_dir, exist_ok=True)

    # Build action indices from a tmp env
    tmp = CorrectorPlayEnv(**env_kwargs)
    idx = build_action_indices(tmp)
    idx_c1a1 = idx["C1A1"]
    idx_high = idx["HIGH"]

    if highorder_gate is None:
        highorder_gate = {
            'A2': 30.0,
            'B2': 30.0,
            'A3': 800.0,
            'S3': 800.0,
            'C3': 500.0,
        }
    # Dummy env to load pretrained C1A1 correctly (same wrappers/space as training)
    dummy_c1 = DummyVecEnv([make_low_env("C1A1", seed=seed, action_idx=idx_c1a1, env_kwargs=env_kwargs)])
    c1a1_policy = RecurrentPPO.load(c1a1_model_path, env=dummy_c1, device="cpu")

    def make_env(rank: int):
        def _init():
            base = CorrectorPlayEnv(**env_kwargs)
            base.reset(seed=seed + rank)

            # each subprocess loads its own c1a1 policy on CPU
            local_dummy_c1 = DummyVecEnv(
                [make_low_env("C1A1", seed=seed + rank, action_idx=idx_c1a1, env_kwargs=env_kwargs)])
            local_c1 = RecurrentPPO.load(c1a1_model_path, env=local_dummy_c1, device="cpu")

            env = AlternatingHighEnv(
                base_env=base,
                idx_c1a1=idx_c1a1,
                idx_high=idx_high,
                c1a1_policy=local_c1,
                c1_gate=c1_gate,
                a1_gate=a1_gate,
                a2_gate=highorder_gate['A2'],
                b2_gate=highorder_gate['B2'],
                a3_gate=highorder_gate['A3'],
                s3_gate=highorder_gate['S3'],
                c3_gate=highorder_gate['C3'],
                high_gate_bonus=high_gate_bonus,
                n_c1a1_max=n_c1a1_max,
                n_high=n_high,
                meta_max_steps=meta_max_steps,
                deterministic_c1a1=deterministic_c1a1,
            )
            return env

        return _init

    vec = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec = VecMonitor(vec)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec,
        verbose=1,
        seed=seed,
        device=device,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128], lstm_hidden_size=128),
    )

    action_table = tmp.action_table
    trace_cb = HighAndC1A1TraceCSVCallback(
        save_dir=os.path.join(out_dir, "high_seq_traces"),
        action_table=action_table,
        every_n_episodes=10,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, callback=trace_cb)

    tag = time_tag()
    save_path = os.path.join(out_dir, f"high_with_c1a1_guard_rppo_{tag}.zip")
    model.save(save_path)
    return save_path


# 你 obs 的固定顺序
OBS_NAMES = ["C1", "A1", "A2", "B2", "A3", "S3", "C3"]


def evaluate_low_level_model(
        model_path: str,
        mode: str,  # "C1A1" or "HIGH"
        out_dir: str = "eval_trace.csv",
        deterministic: bool = True,
        max_steps: int = 500,
        seed: int = 0,
        # ---- 需要你按训练时一致地传入 env_kwargs ----
        env_kwargs: dict | None = None,
):
    """
    Rollout one episode with a trained RecurrentPPO low-level model (C1A1 or HIGH),
    printing step-by-step actions and aberration changes, and saving a CSV trace.

    Output per step:
      - action_sub_idx, base_action_idx, decoded CEOS meaning (target/type/pct/step/dir)
      - aberrations before/after (C1,A1,A2,B2,A3,S3,C3)
      - reward, done, gate_ok, deviation_exploded
    """
    assert mode in ("C1A1", "HIGH"), "mode must be 'C1A1' or 'HIGH'"

    # --- import from your codebase ---
    # If these are in train_hierarchical_rppo.py, import accordingly:

    if env_kwargs is None:
        env_kwargs = {}

    # Build a tmp env to get action_table and indices (this does not affect training)
    tmp = CorrectorPlayEnv(**env_kwargs)
    action_table = tmp.action_table
    idx = build_action_indices(tmp)
    action_idx = idx["C1A1"] if mode == "C1A1" else idx["HIGH"]

    # Use a single-env DummyVecEnv for evaluation (easier for printing)
    vec_env = DummyVecEnv([make_low_env(mode, seed=seed, action_idx=action_idx, env_kwargs=env_kwargs)])

    # Load model
    model = RecurrentPPO.load(model_path, device="auto")

    # reset
    obs = vec_env.reset()
    lstm_state = None
    episode_start = np.array([True])

    # helper: decode subset->base mapping
    # vec_env.envs[0] is outer wrapper; walk down to find ActionSubsetWrapper.subset
    def get_subset_mapping():
        w = vec_env.envs[0]
        while hasattr(w, "env"):
            if hasattr(w, "subset"):
                return w.subset
            w = w.env
        return None

    subset = get_subset_mapping()
    if subset is None:
        raise RuntimeError("Cannot find ActionSubsetWrapper.subset mapping in env wrappers.")

    rows = []

    # Write init row
    o0 = obs[0]
    init_row = {"t": -1, "mode": mode, "action_sub": "", "base_action_idx": "",
                "target": "", "type": "", "pct": "", "step": "", "dir": "",
                "reward": "", "done": False, "gate_ok": "", "deviation_exploded": ""}
    for i, name in enumerate(OBS_NAMES):
        # init_row[f"{name}_before"] = float(o0[i])
        # init_row[f"{name}_after"] = float(o0[i])
        init_row[f"{name}"] = np.around(float(o0[i]), decimals=1)
    rows.append(init_row)

    print(f"\n=== EVAL START: mode={mode}, deterministic={deterministic} ===")
    print("Init:", " ".join([f"{OBS_NAMES[i]}={o0[i]:.3f}" for i in range(len(OBS_NAMES))]))

    for t in range(max_steps):
        o_before = obs[0].copy()

        action_sub, lstm_state = model.predict(
            obs, state=lstm_state, episode_start=episode_start, deterministic=deterministic
        )
        a_sub = int(np.array(action_sub).ravel()[0])
        base_action_idx = int(subset[a_sub])

        # decode action meaning
        act = action_table[base_action_idx]
        target = act.get("target", "")
        act_type = act.get("type", "")
        pct = act.get("pct", "")
        step = act.get("step", "")
        direction = act.get("dir", "")

        obs2, reward, done, infos = vec_env.step([a_sub])
        r = float(np.array(reward).ravel()[0])
        d = bool(np.array(done).ravel()[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        o_after = obs2[0].copy()

        gate_ok = info.get("gate_ok", "")
        exploded = bool(info.get("deviation_exploded", False))

        # print one step
        delta = o_after - o_before
        print(
            f"[t={t:03d}] a_sub={a_sub:3d} base={base_action_idx:3d} "
            f"{target} {act_type} {pct}{step}{direction} | "
            + " ".join([f"{OBS_NAMES[i]}:{o_before[i]:8.2f}->{o_after[i]:8.2f} (Δ{delta[i]:+7.2f})"
                        for i in range(len(OBS_NAMES))])
            + f" | r={r:+8.3f} done={d} gate_ok={gate_ok} exploded={exploded}"
        )

        # record row
        row = {
            "t": t, "mode": mode,
            "action_sub": a_sub,
            "base_action_idx": base_action_idx,
            "target": target,
            "type": act_type,
            "pct": pct,
            "step": step,
            "dir": direction,
            "reward": r,
            "done": d,
            "gate_ok": gate_ok,
            "deviation_exploded": exploded,
        }
        for i, name in enumerate(OBS_NAMES):
            # row[f"{name}_before"] = float(o_before[i])
            # row[f"{name}_after"] = float(o_after[i])
            row[f"{name}"] = np.around(float(o_after[i]), decimals=1)
        rows.append(row)

        obs = obs2
        episode_start = np.array([d])

        if d:
            print("=== EPISODE DONE ===")
            break

    # save csv
    os.makedirs(out_dir or ".", exist_ok=True)
    tag = time_tag()
    csv_path = os.path.join(out_dir, f"eval_alt_c1a1_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved trace to: {csv_path}\n")


def decode_action_to_string(action_dict: dict) -> str:
    """
    Robust decoder for your action_table entry.
    Tries to format like: 'A2 +30% pct_button (base=40)' or 'C1 -10 step_button'
    Adjust keys if your action_table uses different field names.
    """
    if not isinstance(action_dict, dict):
        return str(action_dict)

    target = action_dict.get("target", "?")
    typ = action_dict.get("type", action_dict.get("kind", "?"))

    # sign / direction
    sign = action_dict.get("sign", None)
    direction = action_dict.get("direction", action_dict.get("dir", None))

    # magnitude fields
    pct = action_dict.get("pct", action_dict.get("percent", None))
    step = action_dict.get("step", action_dict.get("delta", None))
    base = action_dict.get("base", action_dict.get("base_step", None))

    # build string
    sgn = ""
    if sign in (+1, 1, "plus", "+"):
        sgn = "+"
    elif sign in (-1, -1, "minus", "-"):
        sgn = "-"
    elif isinstance(direction, str):
        if direction.lower().startswith(("p", "+", "inc")):
            sgn = "+"
        elif direction.lower().startswith(("n", "-", "dec")):
            sgn = "-"

    mag = None
    unit = ""
    if pct is not None:
        mag = pct
        unit = "%"
    elif step is not None:
        mag = step
        unit = ""

    if mag is None:
        core = f"{target} {typ}"
    else:
        core = f"{target} {sgn}{mag}{unit} {typ}"

    if base is not None:
        core += f" (base={base})"
    return core


def gate_ok(obs: np.ndarray, gates: dict) -> bool:
    """
    obs order: [C1, A1, A2, B2, A3, S3, C3]
    gates: dict like {"C1":10, "A1":20, ...}. Only checks provided keys.
    """
    name2idx = {"C1": 0, "A1": 1, "A2": 2, "B2": 3, "A3": 4, "S3": 5, "C3": 6}
    for k, thr in gates.items():
        if k not in name2idx:
            continue
        if abs(float(obs[name2idx[k]])) >= float(thr):
            return False
    return True


def evaluate_alternating_high_model(
        *,
        out_csv_dir: str,
        high_model_path: str,
        c1a1_model_path: str,
        env_kwargs: dict,
        n_episodes: int = 5,
        seed: int = 0,

        # must match training (Scheme B)
        n_high: int = 3,
        n_c1a1_max: int = 60,
        c1_gate: float = 10.0,
        a1_gate: float = 20.0,
        meta_max_steps: int = 200,

        # NEW: gates for stopping condition
        stop_gates: dict | None = None,

        deterministic_high: bool = True,
        deterministic_c1a1: bool = True,
):
    """
    Evaluate AlternatingHighEnv (Scheme B: MultiDiscrete action sequence).

    NEW behavior:
      - If stop_gates is provided, evaluation ends EARLY when all requested aberrations
        satisfy |aberration| < gate (for all keys in stop_gates).

    CSV:
      - One CSV per episode
      - Rows expanded: C1A1 trace + HIGH trace, each action as one row
    """
    os.makedirs(out_csv_dir, exist_ok=True)

    tmp = CorrectorPlayEnv(**env_kwargs)
    action_table = tmp.action_table
    idx = build_action_indices(tmp)
    idx_c1a1 = idx["C1A1"]
    idx_high = idx["HIGH"]

    dummy_c1 = DummyVecEnv([make_low_env("C1A1", seed=seed, action_idx=idx_c1a1, env_kwargs=env_kwargs)])
    c1a1_policy = RecurrentPPO.load(c1a1_model_path, env=dummy_c1, device="cpu")

    base = CorrectorPlayEnv(**env_kwargs)
    base.reset(seed=seed)

    env = AlternatingHighEnv(
        base_env=base,
        idx_c1a1=idx_c1a1,
        idx_high=idx_high,
        c1a1_policy=c1a1_policy,
        c1_gate=c1_gate,
        a1_gate=a1_gate,
        n_c1a1_max=n_c1a1_max,
        n_high=n_high,
        meta_max_steps=meta_max_steps,
        deterministic_c1a1=deterministic_c1a1,
    )

    high_model = RecurrentPPO.load(high_model_path, env=None, device="cpu")

    # If user doesn't pass stop_gates, default to “all aberrations under gates”
    if stop_gates is None:
        stop_gates = {
            "C1": c1_gate,
            "A1": a1_gate,
            "A2": 30.0,
            "B2": 30.0,
            "A3": 800.0,
            "S3": 800.0,
            "C3": 500.0,
        }

    # --- helpers for labels ---
    def _safe_float(x, default=np.nan):
        try:
            return float(x)
        except Exception:
            return default

    def c1a1_ok_from_vals(vals: dict) -> bool:
        c1 = abs(_safe_float(vals.get("C1", np.nan)))
        a1 = abs(_safe_float(vals.get("A1", np.nan)))
        return (c1 < c1_gate) and (a1 < a1_gate)

    # gates dict only for high-order terms (A2,B2,A3,S3,C3)
    high_gates = {
        "A2": float(stop_gates.get("A2", 30.0)),
        "B2": float(stop_gates.get("B2", 30.0)),
        "A3": float(stop_gates.get("A3", 800.0)),
        "S3": float(stop_gates.get("S3", 800.0)),
        "C3": float(stop_gates.get("C3", 500.0)),
    }

    def high_ok_from_vals(vals: dict) -> bool:
        for k, g in high_gates.items():
            if abs(_safe_float(vals.get(k, np.nan))) >= g:
                return False
        return True

    # ensure consistent CSV header even if first row is C1A1 (reward="")
    fieldnames = [
        "episode", "meta_step", "phase", "phase_j",
        "action_sub_idx", "base_action_idx", "action_str",
        "reward", "deviation_exploded", "stop_reason",
        "c1a1_ok", "high_order_ok",
        "C1", "A1", "A2", "B2", "A3", "S3", "C3",
    ]

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False

        state = None
        episode_start = np.array([True], dtype=bool)
        rows = []
        meta_step = 0

        # NOTE: csv_path is defined later, so don't print it here
        if gate_ok(obs, stop_gates):
            print(f"[EVAL] episode {ep}: already below gates right after reset. (skip)")
            continue

        while (not done) and (not truncated):
            obs_batch = obs[None, :].astype(np.float32)

            action_seq, state = high_model.predict(
                obs_batch,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic_high,
            )
            episode_start[:] = False

            obs2, reward, done, truncated, info = env.step(action_seq[0])

            # ---- record C1A1 trace if present ----
            for tr in info.get("c1a1_trace", []):
                base_idx = int(tr.get("base_action_idx", -1))
                act_str = decode_action_to_string(action_table[base_idx]) if (
                        0 <= base_idx < len(action_table)) else "UNKNOWN"

                vals = {n: tr.get(f"{n}_after", "") for n in ["C1", "A1", "A2", "B2", "A3", "S3", "C3"]}
                row = {
                    "episode": ep,
                    "meta_step": meta_step,
                    "phase": "C1A1",
                    "phase_j": int(tr.get("c1a1_j", -1)),
                    "action_sub_idx": int(tr.get("action_sub_idx", -1)),
                    "base_action_idx": base_idx,
                    "action_str": act_str,
                    "reward": "",
                    "deviation_exploded": bool(tr.get("deviation_exploded", False)),
                    "stop_reason": "",
                    # NEW labels:
                    "c1a1_ok": c1a1_ok_from_vals(vals),
                    "high_order_ok": high_ok_from_vals(vals),
                }
                row.update(vals)
                rows.append(row)

            # ---- record HIGH trace ----
            for tr in info.get("high_trace", []):
                base_idx = int(tr.get("base_action_idx", -1))
                act_str = decode_action_to_string(action_table[base_idx]) if (
                        0 <= base_idx < len(action_table)) else "UNKNOWN"

                vals = {n: tr.get(f"{n}_after", "") for n in ["C1", "A1", "A2", "B2", "A3", "S3", "C3"]}
                row = {
                    "episode": ep,
                    "meta_step": meta_step,
                    "phase": "HIGH",
                    "phase_j": int(tr.get("high_j", -1)),
                    "action_sub_idx": int(tr.get("action_sub_idx", -1)),
                    "base_action_idx": base_idx,
                    "action_str": act_str,
                    "reward": float(reward),
                    "deviation_exploded": bool(tr.get("deviation_exploded", False)),
                    "stop_reason": "",
                    # NEW labels:
                    "c1a1_ok": c1a1_ok_from_vals(vals),
                    "high_order_ok": high_ok_from_vals(vals),
                }
                row.update(vals)
                rows.append(row)

            obs = obs2
            meta_step += 1

            # NEW: stop as soon as all requested aberrations are under gates
            if gate_ok(obs, stop_gates):
                obs_vals = {n: float(obs[i]) for i, n in enumerate(["C1", "A1", "A2", "B2", "A3", "S3", "C3"])}
                rows.append({
                    "episode": ep,
                    "meta_step": meta_step,
                    "phase": "STOP",
                    "phase_j": -1,
                    "action_sub_idx": "",
                    "base_action_idx": "",
                    "action_str": "all_below_gates",
                    "reward": float(reward),
                    "deviation_exploded": bool(info.get("deviation_exploded", False)),
                    "stop_reason": "all_below_gates",
                    # NEW labels:
                    "c1a1_ok": c1a1_ok_from_vals(obs_vals),
                    "high_order_ok": high_ok_from_vals(obs_vals),
                    **obs_vals,
                })
                print(f"[EVAL] episode {ep}: all_below_gates -> stop.")
                break

            if meta_step >= meta_max_steps:
                break

        tag = time_tag()
        csv_path = os.path.join(out_csv_dir, f"eval_alt_high_ep{ep}_{tag}.csv")
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        print(f"[EVAL] episode {ep}: saved {csv_path}")


class C1A1MetricsCallback(BaseCallback):
    """
    Log physically meaningful C1A1 metrics:
      - mean |C1|
      - mean |A1|
      - gate_ok frequency (episode-level)
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_abs_c1 = []
        self.ep_abs_a1 = []
        self.ep_gate_ok = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if "abs_C1" in info:
                self.ep_abs_c1.append(info["abs_C1"])
                self.ep_abs_a1.append(info["abs_A1"])
                self.ep_gate_ok.append(1.0 if info["gate_ok"] else 0.0)

            if done and len(self.ep_abs_c1) > 0:
                # episode-level means
                self.logger.record("C1A1/abs_C1_mean", np.mean(self.ep_abs_c1))
                self.logger.record("C1A1/abs_A1_mean", np.mean(self.ep_abs_a1))
                self.logger.record("C1A1/gate_ok_rate", np.mean(self.ep_gate_ok))

                # reset buffers
                self.ep_abs_c1.clear()
                self.ep_abs_a1.clear()
                self.ep_gate_ok.clear()

        return True


class ActionCSVCallback(BaseCallback):
    def __init__(self, csv_path, action_table, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.action_table = action_table
        self.f = None
        self.w = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=[
            "t", "env_id", "base_action_idx", "target", "type", "pct", "step", "dir",
            "C1", "A1", "A2", "B2", "A3", "S3", "C3"
        ])
        self.w.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for env_id, info in enumerate(infos):
            if not info:
                continue
            if "base_action_idx" not in info:
                continue

            idx = int(info["base_action_idx"])
            act = self.action_table[idx]
            row = {
                "t": int(self.num_timesteps),
                "env_id": env_id,
                "base_action_idx": idx,
                "target": act.get("target"),
                "type": act.get("type"),
                "pct": act.get("pct", ""),
                "step": act.get("step", ""),
                "dir": act.get("dir", ""),
            }

            # If your env puts current params in info, use those; otherwise read obs
            obs = self.locals.get("new_obs", None)
            if obs is not None:
                o = np.array(obs)[env_id]
                row.update({"C1": o[0], "A1": o[1], "A2": o[2], "B2": o[3], "A3": o[4], "S3": o[5], "C3": o[6]})

            self.w.writerow(row)

        return True

    def _on_training_end(self):
        if self.f is not None:
            self.f.close()


# class C1A1PrintCallback(BaseCallback):
#     """
#     训练时记录完整 episode 轨迹（按采样频率）。
#     支持 VecEnv（infos/actions/dones 都是 list/array）。
#
#     每一行记录：
#       t, env_id, episode_id, step_in_ep,
#       action_sub_idx, base_action_idx, target, type, pct, step, dir,
#       C1_before, A1_before, C1_after, A1_after, reward, done, gate_ok
#     """
#     def __init__(
#         self,
#         out_dir: str,
#         action_table: list,
#         prefix: str = "c1a1_trace",
#         sample_every_episodes: int = 200,  # 每 200 个 episode 记录一次
#         verbose: int = 0,
#     ):
#         super().__init__(verbose)
#         self.out_dir = out_dir
#         self.action_table = action_table
#         self.prefix = prefix
#         self.sample_every_episodes = int(sample_every_episodes)
#
#         # per-env buffers
#         self.last_obs = None                 # shape (n_envs, obs_dim)
#         self.episode_count = None            # per-env episode counter
#         self.step_in_ep = None               # per-env step index
#         self.rows_buffer = None              # per-env list of rows
#         self.recording_on = None             # per-env bool
#
#     def _on_training_start(self) -> None:
#         os.makedirs(self.out_dir, exist_ok=True)
#         n_envs = self.training_env.num_envs
#         self.last_obs = None
#         self.episode_count = np.zeros(n_envs, dtype=np.int64)
#         self.step_in_ep = np.zeros(n_envs, dtype=np.int64)
#         self.rows_buffer = [[] for _ in range(n_envs)]
#         self.recording_on = np.zeros(n_envs, dtype=bool)
#
#     def _decode_action(self, base_action_idx: int):
#         if base_action_idx is None or base_action_idx < 0:
#             return ("", "", "", "", "")
#         act = self.action_table[int(base_action_idx)]
#         return (
#             act.get("target", ""),
#             act.get("type", ""),
#             act.get("pct", ""),
#             act.get("step", ""),
#             act.get("dir", ""),
#         )
#
#     def _flush_episode(self, env_id: int):
#         """Write buffered rows of one env's episode into a CSV file."""
#         rows = self.rows_buffer[env_id]
#         if not rows:
#             return
#
#         ep_id = int(self.episode_count[env_id])
#         csv_path = os.path.join(self.out_dir, f"{self.prefix}_env{env_id}_ep{ep_id}_t{self.num_timesteps}.csv")
#
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#             writer.writeheader()
#             writer.writerows(rows)
#
#         # clear buffer
#         self.rows_buffer[env_id] = []
#
#     def _on_step(self) -> bool:
#         infos = self.locals.get("infos", None)
#         actions = self.locals.get("actions", None)
#         obs_after = self.locals.get("new_obs", None)
#
#         if infos is None or actions is None or obs_after is None:
#             return True
#
#         obs_after = np.array(obs_after)
#         if obs_after.ndim == 1:
#             obs_after = obs_after[None, :]
#
#         obs_before = self.locals.get("obs", None)
#         if obs_before is not None:
#             obs_before = np.array(obs_before)
#             if obs_before.ndim == 1:
#                 obs_before = obs_before[None, :]
#         else:
#             obs_before = self.last_obs
#             if obs_before is None:
#                 obs_before = obs_after.copy()
#
#         # dones (兼容)
#         n_envs = obs_after.shape[0]
#         dones = self.locals.get("dones", None)
#         if dones is None:
#             terminated = self.locals.get("terminated", None)
#             truncated = self.locals.get("truncated", None)
#             if terminated is not None and truncated is not None:
#                 dones = np.array(terminated) | np.array(truncated)
#             else:
#                 dones = np.zeros(n_envs, dtype=bool)
#         else:
#             dones = np.array(dones, dtype=bool)
#
#         rewards = self.locals.get("rewards", None)
#         if rewards is None:
#             rewards = np.zeros(n_envs, dtype=np.float32)
#         rewards = np.array(rewards).reshape(-1)
#
#         # 初始化 last_obs：必须用 obs_after（obs_before 可能 None）
#         if self.last_obs is None:
#             self.last_obs = obs_after.copy()
#
#         # iterate envs
#         for env_id in range(n_envs):
#             info = infos[env_id] if isinstance(infos, (list, tuple)) else infos
#             if info is None:
#                 info = {}
#
#             # decide whether to record this episode
#             # If step_in_ep==0, a new episode has started for this env_id
#             if self.step_in_ep[env_id] == 0:
#                 # increment episode counter at the START (so ep_id is meaningful)
#                 self.episode_count[env_id] += 1
#                 ep_id = int(self.episode_count[env_id])
#
#                 # sample rule: record if ep_id % sample_every == 0
#                 self.recording_on[env_id] = (ep_id % self.sample_every_episodes == 0)
#                 # reset buffer if starting a recorded episode
#                 if self.recording_on[env_id]:
#                     self.rows_buffer[env_id] = []
#
#                     # also write an "init row" (t=-1) using obs_before
#                     c1_init = float(obs_before[env_id][0])
#                     a1_init = float(obs_before[env_id][1])
#                     self.rows_buffer[env_id].append({
#                         "t": int(self.num_timesteps),
#                         "env_id": env_id,
#                         "episode_id": ep_id,
#                         "step_in_ep": -1,
#                         "action_sub_idx": "",
#                         "base_action_idx": "",
#                         "target": "",
#                         "type": "",
#                         "pct": "",
#                         "step": "",
#                         "dir": "",
#                         "C1_before": c1_init,
#                         "A1_before": a1_init,
#                         "C1_after": c1_init,
#                         "A1_after": a1_init,
#                         "reward": "",
#                         "done": False,
#                         "gate_ok": "",
#                     })
#
#             # record step row if recording is on for this env
#             if self.recording_on[env_id]:
#                 a_sub = int(np.array(actions).reshape(-1)[env_id])
#                 base_action_idx = info.get("base_action_idx", "")
#
#                 target, act_type, pct, step, direction = ("", "", "", "", "")
#                 if base_action_idx != "":
#                     target, act_type, pct, step, direction = self._decode_action(int(base_action_idx))
#
#                 c1_b = float(obs_before[env_id][0])
#                 a1_b = float(obs_before[env_id][1])
#                 c1_a = float(obs_after[env_id][0])
#                 a1_a = float(obs_after[env_id][1])
#
#                 gate_ok = info.get("gate_ok", "")
#
#                 self.rows_buffer[env_id].append({
#                     "t": int(self.num_timesteps),
#                     "env_id": env_id,
#                     "episode_id": int(self.episode_count[env_id]),
#                     "step_in_ep": int(self.step_in_ep[env_id]),
#                     "action_sub_idx": a_sub,
#                     "base_action_idx": base_action_idx,
#                     "target": target,
#                     "type": act_type,
#                     "pct": pct,
#                     "step": step,
#                     "dir": direction,
#                     "C1_before": c1_b,
#                     "A1_before": a1_b,
#                     "C1_after": c1_a,
#                     "A1_after": a1_a,
#                     "reward": float(rewards[env_id]),
#                     "done": bool(dones[env_id]),
#                     "gate_ok": gate_ok,
#                 })
#
#             # advance step counter or reset at episode end
#             if dones[env_id]:
#                 # episode ended -> flush if recorded
#                 if self.recording_on[env_id]:
#                     self._flush_episode(env_id)
#                 self.step_in_ep[env_id] = 0
#                 self.recording_on[env_id] = False
#             else:
#                 self.step_in_ep[env_id] += 1
#
#         # update last obs
#         self.last_obs = obs_after.copy()
#         return True


class EpisodeTraceToCSVCallback(BaseCallback):
    """
    Record full episode traces to CSV (sampled every N episodes) for VecEnv/SubprocVecEnv.

    Key fix:
      - Use pending_reset_obs[env_id] captured immediately after dones[env_id]==True
        (VecEnv auto-reset) as the TRUE initial obs for next episode.
      - This prevents the "init row" (step_in_ep=-1) from being polluted by previous
        episode's terminal state.

    Columns include:
      - action_sub_idx / base_action_idx and decoded action meaning
      - obs before/after for each dimension (default 7 dims: C1,A1,A2,B2,A3,S3,C3)
      - reward/done/gate_ok/deviation_exploded
    """

    def __init__(
            self,
            out_dir: str,
            action_table: list,
            prefix: str,
            sample_every_episodes: int = 200,
            record_dims: int = 7,
            verbose: int = 0,
    ):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.action_table = action_table
        self.prefix = prefix
        self.sample_every_episodes = int(sample_every_episodes)
        self.record_dims = int(record_dims)

        self.obs_names = ["C1", "A1", "A2", "B2", "A3", "S3", "C3"][: self.record_dims]

        # per-env state
        self.ep_count = None
        self.step_in_ep = None
        self.recording_on = None
        self.rows_buffer = None

        self.last_obs = None
        self.pending_reset_obs = None  # <-- 핵심: store true init obs for next episode

    def _on_training_start(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        n_envs = self.training_env.num_envs

        self.ep_count = np.zeros(n_envs, dtype=np.int64)
        self.step_in_ep = np.zeros(n_envs, dtype=np.int64)
        self.recording_on = np.zeros(n_envs, dtype=bool)
        self.rows_buffer = [[] for _ in range(n_envs)]

        self.last_obs = None
        self.pending_reset_obs = [None for _ in range(n_envs)]

    def _decode_action(self, base_action_idx):
        if base_action_idx is None or base_action_idx == "":
            return ("", "", "", "", "")
        act = self.action_table[int(base_action_idx)]
        return (
            act.get("target", ""),
            act.get("type", ""),
            act.get("pct", ""),
            act.get("step", ""),
            act.get("dir", ""),
        )

    def _flush(self, env_id: int):
        rows = self.rows_buffer[env_id]
        if not rows:
            return

        ep_id = int(self.ep_count[env_id])
        csv_path = os.path.join(
            self.out_dir,
            f"{self.prefix}_env{env_id}_ep{ep_id}_t{self.num_timesteps}.csv",
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        self.rows_buffer[env_id] = []

    def _get_dones(self, n_envs: int):
        # SB3 classic
        dones = self.locals.get("dones", None)
        if dones is not None:
            return np.array(dones, dtype=bool).reshape(-1)

        # gymnasium style
        terminated = self.locals.get("terminated", None)
        truncated = self.locals.get("truncated", None)
        if terminated is not None and truncated is not None:
            return (np.array(terminated, dtype=bool) | np.array(truncated, dtype=bool)).reshape(-1)

        return None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        actions = self.locals.get("actions", None)
        obs_after = self.locals.get("new_obs", None)

        if infos is None or actions is None or obs_after is None:
            return True

        obs_after = np.array(obs_after)
        if obs_after.ndim == 1:
            obs_after = obs_after[None, :]

        n_envs = obs_after.shape[0]

        # Determine dones robustly
        dones = self._get_dones(n_envs)
        if dones is None:
            return True

        # Rewards (VecEnv)
        rewards = self.locals.get("rewards", None)
        if rewards is None:
            rewards = np.zeros(n_envs, dtype=np.float32)
        rewards = np.array(rewards).reshape(-1)

        # obs_before: prefer locals["obs"], else last_obs, else obs_after (fallback)
        obs_before = self.locals.get("obs", None)
        if obs_before is not None:
            obs_before = np.array(obs_before)
            if obs_before.ndim == 1:
                obs_before = obs_before[None, :]
        else:
            obs_before = self.last_obs
            if obs_before is None:
                obs_before = obs_after.copy()

        # init last_obs at first call
        if self.last_obs is None:
            self.last_obs = obs_after.copy()

        actions = np.array(actions).reshape(-1)

        for env_id in range(n_envs):
            info = infos[env_id] if isinstance(infos, (list, tuple)) else infos
            if info is None:
                info = {}

            # ---------- episode start ----------
            if self.step_in_ep[env_id] == 0:
                # count this new episode
                self.ep_count[env_id] += 1
                ep_id = int(self.ep_count[env_id])

                # decide sampling
                self.recording_on[env_id] = (ep_id % self.sample_every_episodes == 0)

                if self.recording_on[env_id]:
                    self.rows_buffer[env_id] = []

                    # TRUE init obs: use pending_reset_obs if available
                    init_obs = self.pending_reset_obs[env_id]
                    if init_obs is None:
                        # fallback (first ever episode): use current obs_before
                        init_obs = obs_before[env_id].copy()
                    else:
                        init_obs = np.array(init_obs, dtype=float).copy()

                    # clear pending init once consumed
                    self.pending_reset_obs[env_id] = None

                    row = {
                        "t": int(self.num_timesteps),
                        "env_id": env_id,
                        "episode_id": ep_id,
                        "step_in_ep": -1,
                        "action_sub_idx": "",
                        "base_action_idx": "",
                        "target": "",
                        "type": "",
                        "pct": "",
                        "step": "",
                        "dir": "",
                        "reward": "",
                        "done": False,
                        "gate_ok": info.get("gate_ok", ""),
                        "deviation_exploded": bool(info.get("deviation_exploded", False)),
                    }

                    for i, name in enumerate(self.obs_names):
                        v = float(init_obs[i])
                        row[f"{name}_before"] = v
                        row[f"{name}_after"] = v

                    self.rows_buffer[env_id].append(row)

            # ---------- record step row ----------
            if self.recording_on[env_id]:
                a_sub = int(actions[env_id])
                base_action_idx = info.get("base_action_idx", "")
                target, act_type, pct, step, direction = self._decode_action(base_action_idx)

                row = {
                    "t": int(self.num_timesteps),
                    "env_id": env_id,
                    "episode_id": int(self.ep_count[env_id]),
                    "step_in_ep": int(self.step_in_ep[env_id]),
                    "action_sub_idx": int(info.get("action_sub_idx", a_sub)),
                    "base_action_idx": base_action_idx,
                    "target": target,
                    "type": act_type,
                    "pct": pct,
                    "step": step,
                    "dir": direction,
                    "reward": float(rewards[env_id]),
                    "done": bool(dones[env_id]),
                    "gate_ok": info.get("gate_ok", ""),
                    "deviation_exploded": bool(info.get("deviation_exploded", False)),
                }

                for i, name in enumerate(self.obs_names):
                    row[f"{name}_before"] = float(obs_before[env_id][i])
                    row[f"{name}_after"] = float(obs_after[env_id][i])

                self.rows_buffer[env_id].append(row)

            # ---------- episode end ----------
            if dones[env_id]:
                # IMPORTANT: VecEnv auto-resets; obs_after[env_id] at this step is typically the RESET obs.
                # Store it as pending init for next episode.
                self.pending_reset_obs[env_id] = obs_after[env_id].copy()

                if self.recording_on[env_id]:
                    self._flush(env_id)

                self.step_in_ep[env_id] = 0
                self.recording_on[env_id] = False
            else:
                self.step_in_ep[env_id] += 1

        # update last_obs to the latest obs_after (post-step)
        self.last_obs = obs_after.copy()
        return True


def format_ceos_action(act: dict) -> str:
    if not isinstance(act, dict):
        return str(act)
    tgt = act.get("target", "?")
    typ = act.get("type", act.get("kind", "?"))
    d = act.get("dir", act.get("direction", act.get("sign", None)))
    sign = "+" if d in (+1, 1, "plus", "+") else "-" if d in (-1, -1, "minus", "-") else ""
    pct = act.get("pct", act.get("percent", None))
    step = act.get("step", act.get("delta", None))
    if pct is not None:
        return f"{tgt} {sign}{pct}%".strip()
    if step is not None:
        return f"{tgt} {sign}{step} step".strip()
    return f"{tgt} {typ}"


class HighAndC1A1TraceCSVCallback(BaseCallback):
    """
    Expand BOTH c1a1_trace and high_trace into CSV rows.
    Each row = one executed low-level action.
    """

    def __init__(self, save_dir: str, action_table: list, every_n_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.every_n_episodes = int(every_n_episodes)
        self.action_table = action_table
        self._episode_buffers = {}
        self._episode_counts = {}
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        for env_id in range(n_envs):
            self._episode_buffers[env_id] = []
            self._episode_counts[env_id] = 0

    def _decode(self, base_action_idx: int) -> str:
        try:
            i = int(base_action_idx)
            if 0 <= i < len(self.action_table):
                return format_ceos_action(self.action_table[i])
        except Exception:
            pass
        return "UNKNOWN"

    def _append_trace_rows(self, env_id, meta_step, phase, trace_list, reward_val, c1a1_gate_ok=None, high_gate_ok=None):
        for tr in trace_list:
            base_idx = int(tr.get("base_action_idx", -1))
            action_str = self._decode(base_idx)

            row = {
                "env_id": env_id,
                "episode": self._episode_counts[env_id],
                "meta_step": int(meta_step),
                "phase": phase,  # NEW
                "phase_j": int(tr.get("c1a1_j", tr.get("high_j", -1))),
                "action_sub": int(tr.get("action_sub_idx", -1)),
                "base_action_idx": base_idx,
                "action_str": action_str,
                "reward": float(reward_val) if reward_val is not None else "",
                "deviation_exploded": bool(tr.get("deviation_exploded", False)),
                "C1A1_gate_ok": "" if c1a1_gate_ok is None else bool(c1a1_gate_ok),
                "High_gate_ok": "" if high_gate_ok is None else bool(high_gate_ok),
            }
            for name in ["C1", "A1", "A2", "B2", "A3", "S3", "C3"]:
                # row[f"{name}_before"] = tr.get(f"{name}_before", "")
                row[f"{name}"] = np.around(tr.get(f"{name}_after", ""), decimals=1)
            self._episode_buffers[env_id].append(row)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        for env_id, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            # 只有在 AlternatingHighEnv.step() 返回的那一步（stage="HIGH"）才会带 trace
            if info.get("stage", "") == "HIGH":
                meta_step = info.get("meta_t", -1)
                c1_ok = info.get("c1a1_gate_ok", None)
                h_ok = info.get("high_gate_ok", None)
                # 1) C1A1 trace (conditioning)
                c1a1_trace = info.get("c1a1_trace", [])
                self._append_trace_rows(env_id, meta_step, "C1A1", c1a1_trace, c1a1_gate_ok=c1_ok, high_gate_ok=h_ok, reward_val=None)

                # 2) HIGH trace
                high_trace = info.get("high_trace", [])
                self._append_trace_rows(env_id, meta_step, "HIGH", high_trace, c1a1_gate_ok=c1_ok, high_gate_ok=h_ok, reward_val=rewards[env_id])

            # episode end -> flush
            if dones[env_id]:
                self._episode_counts[env_id] += 1
                ep = self._episode_counts[env_id]
                if ep % self.every_n_episodes == 0:
                    self._flush_episode(env_id, ep)
                self._episode_buffers[env_id] = []

        return True

    def _flush_episode(self, env_id: int, ep: int):
        rows = self._episode_buffers[env_id]
        if not rows:
            return
        path = os.path.join(self.save_dir, f"trace_env{env_id}_ep{ep}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        if self.verbose:
            print(f"[Trace] saved {path}")


def _format_ceos_action_from_table(action_table, base_action_idx: int) -> str:
    """把 action_table[base_action_idx] 格式化成 'A1 30%' 这种"""
    try:
        act = action_table[int(base_action_idx)]
    except Exception:
        return f"UNKNOWN({base_action_idx})"
    if not isinstance(act, dict):
        return str(act)

    tgt = act.get("target", "?")
    typ = act.get("type", act.get("kind", ""))
    d = act.get("dir", act.get("direction", act.get("sign", None)))
    sign = "+" if d in (+1, 1, "plus", "+") else "-" if d in (-1, -1, "minus", "-") else ""

    pct = act.get("pct", act.get("percent", None))
    step = act.get("step", act.get("delta", None))
    if pct is not None:
        return f"{tgt} {sign}{pct}%".strip()
    if step is not None:
        return f"{tgt} {sign}{step} step".strip()
    return f"{tgt} {typ}".strip()


def debug_c1a1_action_distribution(
    c1a1_model_path: str,
    env_kwargs: dict,
    seed: int = 0,
    n_samples: int = 200,
):
    """
    在同一个初始 obs 下：
      1) deterministic=True 预测一次
      2) deterministic=False 采样 n_samples 次
    统计 action_sub 分布，并翻译成动作文本
    """

    # 用 tmp env 拿 action_table + idx_c1a1
    tmp = CorrectorPlayEnv(**env_kwargs)
    action_table = tmp.action_table
    idx = build_action_indices(tmp)
    idx_c1a1 = idx["C1A1"]

    # DummyVecEnv 必须跟 low-level 训练 wrapper 顺序一致
    dummy = DummyVecEnv([make_low_env("C1A1", seed=seed, action_idx=idx_c1a1, env_kwargs=env_kwargs)])

    model = RecurrentPPO.load(c1a1_model_path, env=dummy, device="cpu")

    # 固定初始状态：reset 后拿 obs（注意 DummyVecEnv reset 返回 (obs,) 这种）
    obs = dummy.reset()
    # obs shape: (1, obs_dim)
    obs0 = obs.copy()

    # --- 1) deterministic=True ---
    state_det = None
    ep_start = np.array([True], dtype=bool)
    a_det, _ = model.predict(obs0, state=state_det, episode_start=ep_start, deterministic=True)
    a_det = int(np.array(a_det).ravel()[0])
    base_det = int(idx_c1a1[a_det])
    det_str = _format_ceos_action_from_table(action_table, base_det)

    print("\n=== Deterministic=True (single choice) ===")
    print(f"obs0: C1={obs0[0,0]:.3f}, A1={obs0[0,1]:.3f}")
    print(f"action_sub={a_det} -> base_action_idx={base_det} -> {det_str}")

    # --- 2) deterministic=False 采样多次（同一个 obs0，每次都当 episode_start=True，避免 LSTM 状态影响）---
    counts = Counter()
    base_counts = Counter()
    text_counts = Counter()

    for _ in range(n_samples):
        ep_start = np.array([True], dtype=bool)
        a, _ = model.predict(obs0, state=None, episode_start=ep_start, deterministic=False)
        a = int(np.array(a).ravel()[0])
        base = int(idx_c1a1[a])
        txt = _format_ceos_action_from_table(action_table, base)

        counts[a] += 1
        base_counts[base] += 1
        text_counts[txt] += 1

    print(f"\n=== Deterministic=False sampling (n={n_samples}) ===")
    print("Top actions (human-readable):")
    for txt, c in text_counts.most_common(15):
        print(f"  {txt:20s}  {c:4d}  ({c/n_samples*100:5.1f}%)")

    # 额外：看看 C1 vs A1 动作总体占比
    c1_total = sum(c for txt, c in text_counts.items() if txt.strip().startswith("C1"))
    a1_total = sum(c for txt, c in text_counts.items() if txt.strip().startswith("A1"))
    print(f"\nC1 actions total: {c1_total} ({c1_total/n_samples*100:.1f}%)")
    print(f"A1 actions total: {a1_total} ({a1_total/n_samples*100:.1f}%)")

    print("\nIf deterministic always picks A1 but stochastic shows many C1,")
    print("then your policy distribution is close and argmax locks it into A1-only.\n")


if __name__ == "__main__":
    out_dir = "runs_hier_rppo"

    env_kwargs = dict(
        max_steps=500,
        couple_prob_pct=0.5,
        user_gamma={'C1-A1': 0.0005,
                    'A1-C1': 0.24,
                    'B2-A1': -0.912 / 10, 'B2-C1': -1.82 / 10,
                    'A2-A1': -1.244 / 10, 'A2-C1': -0.637 / 10, 'A2-B2': -1.18 / 5,
                    'C3-C1': -0.72 / 100, 'C3-A1': -0.44 / 100, 'C3-A2': +0.967 / 10, 'C3-B2': +0.882 / 10,
                    'C3-S3': +0.345,
                    'S3-A1': -0.325 / 100, 'S3-C1': -1.332 / 100, 'S3-A2': -0.777 / 20, 'S3-B2': -0.577 / 20,
                    'S3-A3': 0.2, 'S3-C3': 0.23},  # leave empty to enable random sparse coupling via couple_prob_pct,
        user_beta={},
        user_sigma={'C1': 1, 'A1': 1, 'A2': 100, 'B2': 50, 'C3': 200, 'A3': 100, 'S3': 100},
        init_ranges={'C1': (0, 20), 'A1': (0, 50), 'A2': (100, 500), 'B2': (100, 500),
                     'C3': (800, 3000), 'S3': (1000, 3000), 'A3': (1000, 3000)}  # nm
    )

    # c1a1_path, idx_c1a1 = train_low_level(
    #     out_dir=out_dir,
    #     env_kwargs=env_kwargs,
    #     a1_gate=2.5,
    #     c1_gate=1.2,
    #     a1_weight=2.0,
    #     c1_weight=2.0,
    #     total_steps_c1a1=50_000,
    #     total_steps_high=100_000,
    #     n_envs=8,
    #     seed=0,
    # )
    #
    # evaluate_low_level_model(
    #     model_path=c1a1_path,
    #     mode="C1A1",
    #     out_dir=out_dir,
    #     deterministic=False,
    #     max_steps=500,
    #     seed=0,
    #     env_kwargs=env_kwargs  # nm
    #     )

    # debug_c1a1_action_distribution(
    #     c1a1_model_path=c1a1_model_path,
    #     env_kwargs=env_kwargs,
    #     seed=0,
    #     n_samples=300,
    # )

    gates = {
        "C1": 1.2,
        "A1": 2.5,
        "A2": 30.0,
        "B2": 30.0,
        "A3": 800.0,
        "S3": 800.0,
        "C3": 500.0,
    }

    c1a1_path = r"D:\0_Codes\RL_STEM-main\RL_STEM-main\runs_hier_rppo\c1a1_rppo_03-01-26-1048.zip"
    seed = 0

    save_path = train_high_order_with_c1a1_guard(
        out_dir=out_dir,
        c1a1_model_path=c1a1_path,
        env_kwargs=env_kwargs,
        n_envs=8,
        total_timesteps=200_000,
        seed=seed,

        n_c1a1_max=60,  # 每次 conditioning 最多跑多少个 C1A1 step
        n_high=3,  # 每次 high-order 决策执行几次同样的 high 动作
        c1_gate=1.2,
        a1_gate=2.5,
        highorder_gate=gates,
        high_gate_bonus=5.0,
        meta_max_steps=200,  # the max number of high-level step sets per episode
        deterministic_c1a1=False,
        # PPO 超参数
        device="cuda",  # 有 GPU 就用
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        lr=3e-4,
    )

    evaluate_alternating_high_model(
        out_csv_dir=r'D:\0_Codes\RL_STEM-main\RL_STEM-main\runs_hier_rppo',
        high_model_path=r"D:\0_Codes\RL_STEM-main\RL_STEM-main\runs_hier_rppo\high_with_c1a1_guard_rppo_03-01-26-1428.zip",
        c1a1_model_path=c1a1_path,
        env_kwargs=env_kwargs,
        n_episodes=3,
        seed=0,
        n_high=3,
        n_c1a1_max=60,
        c1_gate=1.2,
        a1_gate=2.5,
        stop_gates=gates,
        meta_max_steps=200,
        deterministic_high=True,    # eval 推荐 True
        deterministic_c1a1=False,    # conditioning 推荐 True
    )

