import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CorrectorPlayEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    KEYS = ["C1", "A1", "A2", "B2", "A3", "S3", "C3"]
    MAIN_KEYS = ["A2", "B2", "A3", "S3"]
    C1A1_KEYS = ["C1", "A1"]

    PCT_CHOICES = [-200, -100, -50, 10, 20, 50, 100, 200]
    C3_STEPS = [0.01, 0.02, 0.05, 0.1]

    def __init__(
        self,
        render_mode=None,
        static_seed: int = 0,
        dynamic_seed: int | None = None,
        max_steps: int = 200,
        couple_prob_pct: float = 0.5,
        noise_sigma: float = 50.0,
        base_step_range: tuple[float, float] = (0.0, 500.0),
        fail_prob_range: tuple[float, float] = (0.0, 0.2),  # Per-action fail prob range
        user_couplings: dict | None = None,
        goal_threshold: float = 10.0,  # Threshold for considering goal achieved
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.noise_sigma = float(noise_sigma)
        self.couple_prob_pct = float(couple_prob_pct)
        self.user_couplings = user_couplings
        self.goal_threshold = float(goal_threshold)

        # RNG for random variables that stay fixed during play
        self.static_rng = np.random.default_rng(static_seed)

        # RNG for per-step stochasticity (noise + per-step failure draws)
        if dynamic_seed is None:
            dynamic_seed = int(self.static_rng.integers(0, 2**31 - 1))
        self.dynamic_rng = np.random.default_rng(dynamic_seed)

        # Base step per MAIN_KEY sampled once in [0, 500]
        lo, hi = base_step_range
        keys_for_steps = sorted(list(set(self.MAIN_KEYS + self.C1A1_KEYS)))
        self.base_steps = {p: float(self.static_rng.uniform(lo, hi)) for p in keys_for_steps}

        # Initial parameter ranges: ±2000 for all parameters
        self.init_ranges = {k: (-2000.0, 2000.0) for k in self.KEYS}

        # Initial parameters sampled once and kept fixed
        self.init_params = self._sample_init_params_fixed()
        self.params = dict(self.init_params)

        # Build action table and spaces
        self.action_table = self._build_action_table()
        self.action_space = spaces.Discrete(len(self.action_table))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.KEYS),), dtype=np.float32
        )

        # Initialize fixed couplings
        self.couplings = self._init_couplings_fixed()

        # ✅ Per-action fail probabilities, fixed for the whole run
        f_lo, f_hi = fail_prob_range
        self.fail_probs = self._init_fail_probs_fixed(f_lo, f_hi)

        self.t = 0

    def _sample_init_params_fixed(self):
        init_params = {}
        for k in self.KEYS:
            lo, hi = self.init_ranges[k]
            init_params[k] = float(self.static_rng.uniform(lo, hi))
        return init_params

    def _build_action_table(self):
        table = []
        keys_for_buttons = sorted(list(set(self.MAIN_KEYS + self.C1A1_KEYS)))
        for p in keys_for_buttons:
            for pct in self.PCT_CHOICES:
                table.append({
                    "type": "pct_button",
                    "target": p,
                    "pct": int(pct),
                    "key": (p, int(pct)),
                })
        for step in self.C3_STEPS:
            for direction in (-1, +1):
                table.append({
                    "type": "c3_step",
                    "target": "C3",
                    "step": float(step),
                    "dir": int(direction),
                    "key": ("C3", float(step), int(direction)),
                })
        return table

    def _init_couplings_fixed(self):
        couplings = {}
        # Define coupling factor between each pair of parameters
        # Keys are strings "Source-Target", values are floats (the coupling factor)
        for source in self.KEYS:
            for target in self.KEYS:
                if source == target:
                    continue

                key = f"{source}-{target}"

                # Check if user provided a specific coupling factor for this pair
                if self.user_couplings is not None and key in self.user_couplings:
                    couplings[key] = float(self.user_couplings[key])
                    continue

                if self.static_rng.random() < self.couple_prob_pct:
                    # Generate a coupling factor, e.g. between -0.5 and 0.5
                    factor = float(self.static_rng.uniform(-0.5, 0.5))
                    couplings[key] = factor
                else:
                    couplings[key] = 0.0
        print(couplings)
        return couplings

    def _init_fail_probs_fixed(self, lo: float, hi: float):
        """Sample a fixed fail probability for each action key."""
        fail_probs = {}
        for act in self.action_table:
            key = act["key"]
            fail_probs[key] = float(self.static_rng.uniform(lo, hi))
        return fail_probs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.params = dict(self.init_params)

        # Calculate initial deviation for monitoring
        init_main_deviation = sum(abs(self.init_params[k]) for k in self.MAIN_KEYS)

        info = {
            "base_steps": dict(self.base_steps),
            "init_params": dict(self.init_params),
            # Optional: expose per-action fail probs if you want to debug
            "fail_probs": dict(self.fail_probs),
            "init_main_deviation": init_main_deviation,
        }
        return self._obs(), info

    def step(self, action):
        self.t += 1
        act = self.action_table[int(action)]
        key = act["key"]
        target = act["target"]

        delta = {k: 0.0 for k in self.KEYS}

        # ✅ Use per-action fail probability (fixed) for the failure draw
        fail_p = self.fail_probs[key]
        target_failed = (self.dynamic_rng.random() < fail_p)

        # Calculate the change applied to the target parameter
        target_change = 0.0

        # Main update
        if act["type"] == "pct_button":
            if not target_failed:
                pct = act["pct"]
                # delta[target] += - self.base_steps[target] * (pct / 100.0)
                target_change = - self.params[target] * (pct / 100.0)
                delta[target] += target_change
        elif act["type"] == "c3_step":
            if not target_failed:
                target_change = act["dir"] * act["step"]
                delta["C3"] += target_change
        else:
            raise ValueError("Unknown action type")

        # Coupling updates
        # Only apply coupling if the main action succeeded
        if not target_failed:
            for other in self.KEYS:
                if other == target:
                    continue

                # Retrieve coupling factor for this pair
                # Key format must match _init_couplings_fixed (string "Source-Target")
                coupling_key = f"{target}-{other}"
                factor = self.couplings.get(coupling_key, 0.0)

                if factor == 0.0:
                    continue

                # Apply change to 'other' based on the change of the current parameter
                delta[other] += target_change * factor

        # Add Gaussian noise to all non-zero deltas
        for k in self.KEYS:
            if delta[k] != 0.0:
                delta[k] += float(self.dynamic_rng.normal(0.0, self.noise_sigma))

        # Apply deltas
        for k in self.KEYS:
            self.params[k] += delta[k]

        obs = self._obs()

        # Reward: negative sum of absolute values of main correctors (normalized)
        # Goal is to minimize A2, B2, A3, S3 to zero
        main_deviation = sum(abs(self.params[k]) for k in self.MAIN_KEYS)
        reward = -main_deviation / 8000.0  # Normalize by max possible (4 × 2000)

        # Terminal condition: all main correctors within goal threshold
        terminated = all(abs(self.params[k]) < self.goal_threshold for k in self.MAIN_KEYS)
        truncated = (self.t >= self.max_steps) and (not terminated)

        info = {
            "action": act,
            "target_failed": target_failed,
            "fail_prob": fail_p,   # optional but useful for debugging/render
            "delta": delta,
            "main_deviation": main_deviation,  # useful for monitoring training
            "goal_achieved": terminated,
        }

        if self.render_mode == "human":
            self.render(info)

        return obs, reward, terminated, truncated, info

    def _obs(self):
        return np.array([self.params[k] for k in self.KEYS], dtype=np.float32)

    def render(self, info=None):
        if self.render_mode != "human":
            return

        state_str = "  ".join([f"{k}={self.params[k]:.4g}" for k in self.KEYS])

        if info is None:
            print(f"t={self.t}  {state_str}")
            return

        act = info["action"]
        # Show deviation and goal status
        dev = info.get("main_deviation", 0.0)
        goal = info.get("goal_achieved", False)
        dev_str = f" dev={dev:.2f} goal={goal}"

        if act["type"] == "pct_button":
            print(
                f"t={self.t}  "
                f"act={act['target']} pct={act['pct']}{dev_str}  "
                f"{state_str}"
            )
        else:
            print(
                f"t={self.t}  "
                f"act=C3 step={act['step']} dir={act['dir']}{dev_str}  "
                f"{state_str}"
            )
