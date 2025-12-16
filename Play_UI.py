import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from CorrectorPlayEnv import CorrectorPlayEnv


class PlayUI:
    """Tkinter UI that appends a 3-line log entry per action (history preserved)."""

    def __init__(self, root, env: CorrectorPlayEnv):
        self.root = root
        self.env = env
        self.root.title("CorrectorPlayEnv - Play UI")

        # Build action_id lookup tables from env.action_table
        self.main_action_id = {}
        self.c3_action_id = {}
        for i, act in enumerate(self.env.action_table):
            if act["type"] == "pct_button":
                self.main_action_id[(act["target"], int(act["pct"]))] = i
            elif act["type"] == "c3_step":
                self.c3_action_id[("C3", float(act["step"]), int(act["dir"]))] = i

        # Pull these lists from env if present, otherwise use safe defaults
        self.keys = getattr(self.env, "KEYS", ["A1", "A2", "B2", "A3", "S3", "C3"])
        self.main_keys = getattr(self.env, "MAIN_KEYS", ["A2", "B2", "A3", "S3"])
        self.c1a1_keys = getattr(self.env, "C1A1_KEYS", ["C1", "A1"])
        self.pct_choices = getattr(self.env, "PCT_CHOICES", [-200, -100, -50, 10, 20, 50, 100, 200])
        self.c3_steps = getattr(self.env, "C3_STEPS", [0.01, 0.02, 0.05, 0.1])

        # ---- Layout ----
        self.main = ttk.Frame(root, padding=12)
        self.main.grid(row=0, column=0, sticky="nsew")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(2, weight=1)

        # Controls (top)
        self.ctrl_frame = ttk.LabelFrame(self.main, text="Controls", padding=10)
        self.ctrl_frame.grid(row=0, column=0, sticky="ew")
        for c in range(3):
            self.ctrl_frame.columnconfigure(c, weight=1)

        # MAIN controls
        self.main_ctrl = ttk.LabelFrame(self.ctrl_frame, text="MAIN button", padding=10)
        self.main_ctrl.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.main_ctrl.columnconfigure(1, weight=1)

        ttk.Label(self.main_ctrl, text="Button:").grid(row=0, column=0, sticky="w")
        self.target_var = tk.StringVar(value=self.main_keys[0])
        ttk.Combobox(
            self.main_ctrl, textvariable=self.target_var, values=self.main_keys, state="readonly"
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(self.main_ctrl, text="Value (%):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.pct_var = tk.IntVar(value=50)
        ttk.Combobox(
            self.main_ctrl, textvariable=self.pct_var, values=self.pct_choices, state="readonly"
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        ttk.Button(self.main_ctrl, text="Apply", command=self.apply_main_action).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )

        # C3 controls
        self.c3_ctrl = ttk.LabelFrame(self.ctrl_frame, text="C3", padding=10)
        self.c3_ctrl.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.c3_ctrl.columnconfigure(1, weight=1)

        ttk.Label(self.c3_ctrl, text="Step:").grid(row=0, column=0, sticky="w")
        self.step_var = tk.DoubleVar(value=float(self.c3_steps[0]))
        ttk.Combobox(
            self.c3_ctrl, textvariable=self.step_var, values=self.c3_steps, state="readonly"
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(self.c3_ctrl, text="Direction:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.dir_var = tk.IntVar(value=+1)
        dir_frame = ttk.Frame(self.c3_ctrl)
        dir_frame.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Radiobutton(dir_frame, text="+", variable=self.dir_var, value=+1).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(dir_frame, text="-", variable=self.dir_var, value=-1).pack(side="left")

        ttk.Button(self.c3_ctrl, text="Apply", command=self.apply_c3_action).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )

        # Utilities
        self.util_ctrl = ttk.LabelFrame(self.ctrl_frame, text="Utilities", padding=10)
        self.util_ctrl.grid(row=0, column=2, sticky="ew")
        self.util_ctrl.columnconfigure(0, weight=1)

        ttk.Button(self.util_ctrl, text="Reset", command=self.reset_env).grid(row=0, column=0, sticky="ew")
        ttk.Button(self.util_ctrl, text="Random action", command=self.apply_random_action).grid(
            row=1, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(self.util_ctrl, text="Clear log", command=self.clear_log).grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )

        # C1A1 controls
        self.c1a1_ctrl = ttk.LabelFrame(self.main, text="C1A1", padding=10)
        self.c1a1_ctrl.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.c1a1_ctrl.columnconfigure(1, weight=1)

        ttk.Label(self.c1a1_ctrl, text="Button:").grid(row=0, column=0, sticky="w")
        self.c1a1_target_var = tk.StringVar(value=self.c1a1_keys[0])
        ttk.Combobox(
            self.c1a1_ctrl, textvariable=self.c1a1_target_var, values=self.c1a1_keys, state="readonly"
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(self.c1a1_ctrl, text="Value (%):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.c1a1_pct_var = tk.IntVar(value=50)
        ttk.Combobox(
            self.c1a1_ctrl, textvariable=self.c1a1_pct_var, values=self.pct_choices, state="readonly"
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        ttk.Button(self.c1a1_ctrl, text="Apply", command=self.apply_c1a1_action).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )

        # Log area (bottom) - keeps full history
        self.log_frame = ttk.LabelFrame(self.main, text="Log (history preserved)", padding=10)
        self.log_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self.log_frame.rowconfigure(0, weight=1)
        self.log_frame.columnconfigure(0, weight=1)

        self.log = ScrolledText(self.log_frame, height=18, wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")
        self.log.configure(state="disabled")

        # Initialize
        self.reset_env()

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")

    def append_block(self, lines):
        """Append multiple lines as a single log block (history preserved)."""
        self.log.configure(state="normal")
        for line in lines:
            self.log.insert(tk.END, line + "\n")
        self.log.insert(tk.END, "\n")  # blank line between blocks
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def reset_env(self):
        self.env.reset()
        self.append_block([
            f"t = {getattr(self.env, 't', '?')}",
            "action: RESET",
            self._format_values_line(),
        ])

    def apply_main_action(self):
        target = self.target_var.get()
        pct = int(self.pct_var.get())
        action_id = self.main_action_id[(target, pct)]
        self._step_and_log(action_id)

    def apply_c1a1_action(self):
        target = self.c1a1_target_var.get()
        pct = int(self.c1a1_pct_var.get())
        action_id = self.main_action_id[(target, pct)]
        self._step_and_log(action_id)

    def apply_c3_action(self):
        step = float(self.step_var.get())
        direction = int(self.dir_var.get())
        action_id = self.c3_action_id[("C3", step, direction)]
        self._step_and_log(action_id)

    def apply_random_action(self):
        action_id = int(self.env.action_space.sample())
        self._step_and_log(action_id)

    def _step_and_log(self, action_id: int):
        obs, reward, terminated, truncated, info = self.env.step(action_id)
        self.append_block(self._format_3line_log(info))

    def _format_values_line(self) -> str:
        """Format the third line: new parameter values after the action."""
        # Uses env.params to avoid any mismatch with obs ordering
        parts = [f"{k}={self.env.params[k]:.6g}" for k in self.keys]
        return "new values: " + ", ".join(parts)

    def _format_3line_log(self, info: dict):
        """Return exactly 3 lines: t, action (button/value/dir), new values."""
        t = getattr(self.env, "t", "?")
        act = info.get("action", None)

        # Line 1: time step
        line1 = f"t = {t}"

        # Line 2: action summary (no fail_prob)
        if act is None:
            line2 = "action: (none)"
        elif act.get("type") == "pct_button":
            # "value" is the correction strength percentage
            line2 = f"action: button={act['target']}, value={act['pct']}%"
        else:
            # C3 action: "value" is step size, plus direction
            d = act.get("dir", 0)
            dir_str = "+" if int(d) > 0 else "-"
            line2 = f"action: button=C3, value={act['step']}, direction={dir_str}"

        # Line 3: new values
        line3 = self._format_values_line()

        return [line1, line2, line3]


if __name__ == "__main__":
    # Use render_mode=None to prevent terminal printing; UI log is the only output.
    env = CorrectorPlayEnv(
        render_mode=None,
        static_seed=42,
        dynamic_seed=123,
        max_steps=500,
        couple_prob_pct=0.5,
        noise_sigma=50.0,
        user_couplings={"A2-B2": 10, "A2-A1": 10, "B2-A1": 10, "S3-A3": -0.1},
    )

    root = tk.Tk()
    app = PlayUI(root, env)
    root.mainloop()
