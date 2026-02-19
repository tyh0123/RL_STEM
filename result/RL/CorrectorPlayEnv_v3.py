import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any


class CorrectorPlayEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    KEYS = ["C1", "A1", "A2", "B2", "A3", "S3", "C3"]
    MAIN_KEYS = ["A2", "B2", "A3", "S3"]
    C3_KEYs = ["C3"]
    C1_KEYS = ["C1"]
    A1_KEYS = ["A1"]

    PCT_CHOICES = [-200, -100, -50, 10, 20, 50, 100, 200]
    C3_STEPS = [10, 20, 50, 100, 200]
    C1_STEPS = [1, 5, 10, 50, 100]

    def __init__(
            self,
            render_mode: str | None = None,
            # seeds
            error_seed: int | None = None,  # controls step-wise noise sampling
            setting_seed: int = 0,  # controls beta/gamma/sigma sampling
            init_seed: int | None = None,  # controls initial values sampling
            # episode
            max_steps: int = 200,
            # coupling / failure
            couple_prob_pct: float = 0.5,
            fail_prob_range: tuple[float, float] = (0.0, 0.2),
            # user overrides
            user_beta: Dict[str, float] | None = None,  # per-key beta
            user_gamma: Dict[str, float] | None = None,  # pair coupling "Src-Tgt" -> factor
            user_sigma: Dict[str, float] | None = None,  # per-key sigma for OTHER params noise
            # misc
            init_ranges: Dict[str, tuple[float, float]] | None = None,
            # goal
            goal_threshold: float = 20.0,  # All params must be < this for success
            max_deviation: float = 100000.0,  # Early stop if total deviation exceeds this
            obs_clip: float = 10000.0,  # Clip observations to prevent overflow
    ):
        super().__init__()
        self.render_mode = render_mode
        self.goal_threshold = float(goal_threshold)
        self.max_deviation = float(max_deviation)
        self.obs_clip = float(obs_clip)

        self.error_seed = error_seed
        self.setting_seed = int(setting_seed)
        self.init_seed = init_seed

        self.max_steps = int(max_steps)
        self.couple_prob_pct = float(couple_prob_pct)
        self.fail_prob_range = (float(fail_prob_range[0]), float(fail_prob_range[1]))

        # initial range
        self.init_ranges = init_ranges or {k: (-2000.0, 2000.0) for k in self.KEYS}

        # user-provided overrides
        self.user_beta = user_beta or {}
        self.user_gamma = user_gamma or {}
        self.user_sigma = user_sigma or {}

        # RNG that only depends on setting_seed (controls sampling of settings each reset)
        self.setting_rng = np.random.default_rng(self.setting_seed)

        # Build action table & spaces
        self.action_table = self._build_action_table()
        self.action_space = spaces.Discrete(len(self.action_table))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.KEYS),), dtype=np.float32
        )

        # stateful vars (reset() will fill them)
        self.t = 0
        self.params: Dict[str, float] = {}
        self.init_params: Dict[str, float] = {}

        self.beta: Dict[str, float] = {}  # exponent for pct button
        self.couplings: Dict[str, float] = {}  # "Src-Tgt" -> factor
        self.sigma: Dict[str, float] = {}  # noise sigma for OTHER params
        self.fail_probs: Dict[Any, float] = {}  # per-action fixed fail prob for this episode

        # per-episode RNGs (created in reset)
        self.init_rng = None
        self.error_rng = None

    # ----------------------------
    # Action table (interfaces separated)
    # ----------------------------
    def _build_action_table(self):
        table = []

        # pct buttons: MAIN_KEYS + A1
        for p in self.MAIN_KEYS + self.A1_KEYS:
            for pct in self.PCT_CHOICES:
                table.append({
                    "type": "pct_button",
                    "target": p,
                    "pct": int(pct),
                    "key": (p, int(pct)),
                })

        # C3 step +/- dir
        for step in self.C3_STEPS:
            for direction in (-1, +1):
                table.append({
                    "type": "step",
                    "target": "C3",
                    "step": float(step),
                    "dir": int(direction),
                    "key": ("C3", float(step), int(direction)),
                })

        # C1 step +/- dir
        for step in self.C1_STEPS:
            for direction in (-1, +1):
                table.append({
                    "type": "step",
                    "target": "C1",
                    "step": float(step),
                    "dir": int(direction),
                    "key": ("C1", float(step), int(direction)),
                })

        return table

    # ----------------------------
    # Sampling at reset (all re-randomized)
    # ----------------------------
    def _ensure_rngs(self):
        # init_rng: seeded once, then advances across resets
        if self.init_rng is None:
            if self.init_seed is None:
                self.init_rng = np.random.default_rng()  # truly random
            else:
                self.init_rng = np.random.default_rng(int(self.init_seed))

        # error_rng: seeded once, then advances across steps
        if self.error_rng is None:
            if self.error_seed is None:
                self.error_rng = np.random.default_rng()
            else:
                self.error_rng = np.random.default_rng(int(self.error_seed))

    def _sample_init_params(self):
        init_params = {}
        for k in self.KEYS:
            lo, hi = self.init_ranges[k]
            v = float(self.init_rng.uniform(lo, hi))
            # keep your original constraint: non-(C1,C3) must be positive
            if k not in ["C1", "C3"]:
                v = abs(v)
            init_params[k] = v
        return init_params

    def _sample_beta(self):
        beta = {}

        if len(self.user_beta) == 0:
            # no user beta: sample all
            for k in self.KEYS:
                beta[k] = float(self.setting_rng.uniform(0.8, 1.2))
        else:
            # user beta provided: defined keys use user value, others = 0.5
            for k in self.KEYS:
                if k in self.user_beta:
                    beta[k] = float(self.user_beta[k])
                else:
                    beta[k] = 1

        return beta

    def _sample_sigma(self):
        sigma = {}

        if len(self.user_sigma) == 0:
            # no user sigma: sample all
            for k in self.KEYS:
                sigma[k] = float(self.setting_rng.uniform(0.0, 100.0))
        else:
            # user sigma provided: defined keys use user value, others = 0.0
            for k in self.KEYS:
                if k in self.user_sigma:
                    sigma[k] = float(self.user_sigma[k])
                else:
                    sigma[k] = 0.0

        return sigma

    def _sample_couplings(self):
        couplings = {}
        user_defined = (len(self.user_gamma) > 0)

        for src in self.KEYS:
            for tgt in self.KEYS:
                if src == tgt:
                    continue

                key = f"{src}-{tgt}"

                if user_defined:
                    # strictly follow user definition
                    if key in self.user_gamma:
                        couplings[key] = float(self.user_gamma[key])
                    else:
                        couplings[key] = 0.0
                else:
                    # random sparse coupling
                    if self.setting_rng.random() < self.couple_prob_pct:
                        couplings[key] = float(self.setting_rng.uniform(-0.2, 0.2))
                    else:
                        couplings[key] = 0.0

        return couplings

    def _sample_fail_probs(self):
        lo, hi = self.fail_prob_range
        fp = {}
        for act in self.action_table:
            fp[act["key"]] = float(self.setting_rng.uniform(lo, hi))
        return fp

    # ----------------------------
    # Gym API
    # ----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        self._ensure_rngs()

        # settings 每次 reset 重新 sample（由 setting_rng 控制）
        self.beta = self._sample_beta()
        self.sigma = self._sample_sigma()
        self.couplings = self._sample_couplings()
        self.fail_probs = self._sample_fail_probs()

        self.init_params = self._sample_init_params()
        self.params = dict(self.init_params)

        # Track initial deviation for monitoring
        self.init_max_deviation = max(abs(self.init_params[k]) for k in self.KEYS)
        self.init_total_deviation = sum(abs(self.init_params[k]) for k in self.KEYS)

        info = {
            "beta": dict(self.beta),
            "sigma": dict(self.sigma),
            "init_params": dict(self.init_params),
            "init_max_deviation": self.init_max_deviation,
            "init_total_deviation": self.init_total_deviation,
        }
        return self._obs(), info

    def step(self, action):
        self.t += 1
        act = self.action_table[int(action)]
        target = act["target"]
        act_key = act["key"]

        delta = {k: 0.0 for k in self.KEYS}

        # per-action fixed fail prob (for this episode)
        fail_p = self.fail_probs[act_key]
        failed = (self.error_rng.random() < fail_p)

        target_change = 0.0

        if act["type"] == "pct_button":
            if not failed:
                pct = act["pct"]
                val = self.params[target]
                b = float(self.beta.get(target))
                base_mag = abs(val) ** b
                base_sig = float(self.sigma.get(target))
                target_change = -(base_mag + float(self.error_rng.normal(0.0, base_sig))) * (pct / 100.0)
                delta[target] += target_change

                for other in self.KEYS:
                    if other == target:
                        continue
                    ck = f"{target}-{other}"
                    factor = float(self.couplings.get(ck))
                    if factor == 0.0:
                        continue

                    # add noise to OTHER params (sigma[other])
                    sig = float(self.sigma.get(other))

                    if sig > 0:
                        coupled = (pct / 100.0) * (-factor * base_mag + float(self.error_rng.normal(0.0, sig)))
                    else:
                        coupled = -(pct / 100.0) * factor * base_mag

                    delta[other] += coupled

        elif act["type"] == "step":
            if not failed:
                step_val = float(act["step"])
                base_sig = float(self.sigma.get(target))
                target_change = int(act["dir"]) * (step_val + float(self.error_rng.normal(0.0, base_sig)))
                delta[target] += target_change

                for other in self.KEYS:
                    if other == target:
                        continue
                    ck = f"{target}-{other}"
                    factor = float(self.couplings.get(ck))
                    if factor == 0.0:
                        continue

                    # add noise to OTHER params (sigma[other])
                    sig = float(self.sigma.get(other))

                    if sig > 0:
                        coupled = factor * (target_change + float(self.error_rng.normal(0.0, sig)))
                    else:
                        coupled = factor * target_change

                    delta[other] += coupled

        else:
            raise ValueError(f"Unknown action type: {act['type']}")

        # apply deltas + positivity constraint
        for k in self.KEYS:
            self.params[k] += delta[k]
            if k not in ["C1", "C3"]:
                self.params[k] = abs(self.params[k])

        obs = self._obs()

        # Reward: combined max + sum (penalizes outliers while reducing all params)
        # Goal is to minimize all KEYS to zero
        max_dev = max(abs(self.params[k]) for k in self.KEYS)
        total_dev = sum(abs(self.params[k]) for k in self.KEYS)
        reward = -(0.7 * max_dev / 3000.0 + 0.3 * total_dev / 14000.0)

        # Alternative: L2 reward (heavily penalizes outliers)
        # sum_sq = sum(self.params[k]**2 for k in self.KEYS)
        # reward = -np.sqrt(sum_sq) / 5000.0

        # Terminal condition: all params within goal threshold
        terminated = all(abs(self.params[k]) < self.goal_threshold for k in self.KEYS)

        # Early stop if deviation explodes
        deviation_exploded = total_dev > self.max_deviation

        truncated = ((self.t >= self.max_steps) or deviation_exploded) and (not terminated)

        info = {
            "action": act,
            "target_failed": failed,
            "fail_prob": fail_p,
            "delta": delta,
            "max_deviation": max_dev,
            "total_deviation": total_dev,
            "goal_achieved": terminated,
            "deviation_exploded": deviation_exploded,
        }

        if self.render_mode == "human":
            self.render(info)

        return obs, reward, terminated, truncated, info

    def _obs(self):
        obs = np.array([self.params[k] for k in self.KEYS], dtype=np.float64)
        # Clip to prevent overflow when casting to float32
        obs = np.clip(obs, -self.obs_clip, self.obs_clip)
        return obs.astype(np.float32)

    def render(self, info=None):
        if self.render_mode != "human":
            return
        state_str = "  ".join([f"{k}={self.params[k]:.4g}" for k in self.KEYS])
        if info is None:
            print(f"t={self.t}  {state_str}")
            return

        act = info["action"]
        fp = info.get("fail_prob", None)
        fp_str = f" fail_p={fp:.3f}" if fp is not None else ""

        if act["type"] == "pct_button":
            print(f"t={self.t} act={act['target']} pct={act['pct']}{fp_str}  {state_str}")
        else:
            print(f"t={self.t} act={act['target']} step={act['step']} dir={act['dir']}{fp_str}  {state_str}")