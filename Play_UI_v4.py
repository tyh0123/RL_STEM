import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from CorrectorPlayEnv_v4 import CorrectorPlayEnv


class PlayUI:
    """Tkinter UI for the refactored CorrectorPlayEnv.
    - pct_button: MAIN_KEYS + A1_KEYS
    - step: C3 (C3_STEPS) and C1 (C1_STEPS), with +/- direction
    """

    def __init__(self, root, env: CorrectorPlayEnv):
        self.root = root
        self.env = env
        self.root.title("CorrectorPlayEnv - Play UI")

        # Pull lists from env (new API)
        self.keys = getattr(self.env, "KEYS", ["C1", "A1", "A2", "B2", "A3", "S3", "C3"])
        self.main_keys = getattr(self.env, "MAIN_KEYS", ["A2", "B2", "A3", "S3"])
        self.a1_keys = getattr(self.env, "A1_KEYS", ["A1"])
        self.pct_choices = getattr(self.env, "PCT_CHOICES", [-200, -100, -50, 10, 20, 50, 100, 200])
        self.c3_steps = getattr(self.env, "C3_STEPS", [0.01, 0.02, 0.05, 0.1])
        self.c1_steps = getattr(self.env, "C1_STEPS", [1, 5, 10, 50, 100])

        # Build action_id lookup tables from env.action_table
        self.pct_action_id = {}   # (target, pct) -> action_id
        self.step_action_id = {}  # (target, step, dir) -> action_id

        for i, act in enumerate(self.env.action_table):
            if act["type"] == "pct_button":
                self.pct_action_id[(act["target"], int(act["pct"]))] = i
            elif act["type"] == "step":
                self.step_action_id[(act["target"], float(act["step"]), int(act["dir"]))] = i

        # ---- Layout ----
        self.main = ttk.Frame(root, padding=12)
        self.main.grid(row=0, column=0, sticky="nsew")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(3, weight=1)

        # Controls (top row)
        self.ctrl_frame = ttk.LabelFrame(self.main, text="Controls", padding=10)
        self.ctrl_frame.grid(row=0, column=0, sticky="ew")
        for c in range(4):
            self.ctrl_frame.columnconfigure(c, weight=1)

        # MAIN pct controls
        self.main_ctrl = ttk.LabelFrame(self.ctrl_frame, text="MAIN (pct)", padding=10)
        self.main_ctrl.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.main_ctrl.columnconfigure(1, weight=1)

        ttk.Label(self.main_ctrl, text="Button:").grid(row=0, column=0, sticky="w")
        self.main_target_var = tk.StringVar(value=self.main_keys[0])
        ttk.Combobox(
            self.main_ctrl, textvariable=self.main_target_var, values=self.main_keys, state="readonly"
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(self.main_ctrl, text="Value (%):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.main_pct_var = tk.IntVar(value=50)
        ttk.Combobox(
            self.main_ctrl, textvariable=self.main_pct_var, values=self.pct_choices, state="readonly"
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        ttk.Button(self.main_ctrl, text="Apply", command=self.apply_main_action).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )

        # A1 pct controls
        self.a1_ctrl = ttk.LabelFrame(self.ctrl_frame, text="A1 (pct)", padding=10)
        self.a1_ctrl.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.a1_ctrl.columnconfigure(1, weight=1)

        ttk.Label(self.a1_ctrl, text="Button:").grid(row=0, column=0, sticky="w")
        self.a1_target_var = tk.StringVar(value=self.a1_keys[0] if self.a1_keys else "A1")
        ttk.Combobox(
            self.a1_ctrl, textvariable=self.a1_target_var, values=self.a1_keys or ["A1"], state="readonly"
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(self.a1_ctrl, text="Value (%):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.a1_pct_var = tk.IntVar(value=50)
        ttk.Combobox(
            self.a1_ctrl, textvariable=self.a1_pct_var, values=self.pct_choices, state="readonly"
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        ttk.Button(self.a1_ctrl, text="Apply", command=self.apply_a1_action).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )

        # C1 step controls
        self.c1_ctrl = ttk.LabelFrame(self.ctrl_frame, text="C1 (step)", padding=10)
        self.c1_ctrl.grid(row=0, column=2, sticky="ew", padx=(0, 8))
        self.c1_ctrl.columnconfigure(1, weight=1)

        ttk.Label(self.c1_ctrl, text="Step:").grid(row=0, column=0, sticky="w")
        self.c1_step_var = tk.DoubleVar(value=float(self.c1_steps[0]))
        ttk.Combobox(
            self.c1_ctrl, textvariable=self.c1_step_var, values=self.c1_steps, state="readonly"
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(self.c1_ctrl, text="Direction:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.c1_dir_var = tk.IntVar(value=+1)
        c1_dir_frame = ttk.Frame(self.c1_ctrl)
        c1_dir_frame.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Radiobutton(c1_dir_frame, text="+", variable=self.c1_dir_var, value=+1).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(c1_dir_frame, text="-", variable=self.c1_dir_var, value=-1).pack(side="left")

        ttk.Button(self.c1_ctrl, text="Apply", command=self.apply_c1_action).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )

        # C3 step controls
        self.c3_ctrl = ttk.LabelFrame(self.ctrl_frame, text="C3 (step)", padding=10)
        self.c3_ctrl.grid(row=0, column=3, sticky="ew")
        self.c3_ctrl.columnconfigure(1, weight=1)

        ttk.Label(self.c3_ctrl, text="Step:").grid(row=0, column=0, sticky="w")
        self.c3_step_var = tk.DoubleVar(value=float(self.c3_steps[0]))
        ttk.Combobox(
            self.c3_ctrl, textvariable=self.c3_step_var, values=self.c3_steps, state="readonly"
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(self.c3_ctrl, text="Direction:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.c3_dir_var = tk.IntVar(value=+1)
        c3_dir_frame = ttk.Frame(self.c3_ctrl)
        c3_dir_frame.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Radiobutton(c3_dir_frame, text="+", variable=self.c3_dir_var, value=+1).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(c3_dir_frame, text="-", variable=self.c3_dir_var, value=-1).pack(side="left")

        ttk.Button(self.c3_ctrl, text="Apply", command=self.apply_c3_action).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )

        # Utilities row
        self.util_frame = ttk.LabelFrame(self.main, text="Utilities", padding=10)
        self.util_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for c in range(3):
            self.util_frame.columnconfigure(c, weight=1)

        ttk.Button(self.util_frame, text="Reset", command=self.reset_env).grid(row=0, column=0, sticky="ew")
        ttk.Button(self.util_frame, text="Random action", command=self.apply_random_action).grid(
            row=0, column=1, sticky="ew", padx=(8, 8)
        )
        ttk.Button(self.util_frame, text="Clear log", command=self.clear_log).grid(row=0, column=2, sticky="ew")

        # Log area (bottom) - keeps full history
        self.log_frame = ttk.LabelFrame(self.main, text="Log (history preserved)", padding=10)
        self.log_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
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
        self.log.insert(tk.END, "\n")
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
        target = self.main_target_var.get()
        pct = int(self.main_pct_var.get())
        action_id = self.pct_action_id[(target, pct)]
        self._step_and_log(action_id)

    def apply_a1_action(self):
        target = self.a1_target_var.get()
        pct = int(self.a1_pct_var.get())
        action_id = self.pct_action_id[(target, pct)]
        self._step_and_log(action_id)

    def apply_c1_action(self):
        step = float(self.c1_step_var.get())
        direction = int(self.c1_dir_var.get())
        action_id = self.step_action_id[("C1", step, direction)]
        self._step_and_log(action_id)

    def apply_c3_action(self):
        step = float(self.c3_step_var.get())
        direction = int(self.c3_dir_var.get())
        action_id = self.step_action_id[("C3", step, direction)]
        self._step_and_log(action_id)

    def apply_random_action(self):
        action_id = int(self.env.action_space.sample())
        self._step_and_log(action_id)

    def _step_and_log(self, action_id: int):
        obs, reward, terminated, truncated, info = self.env.step(action_id)
        self.append_block(self._format_3line_log(info))

    def _format_values_line(self) -> str:
        parts = [f"{k}={self.env.params[k]:.6g}" for k in self.keys]
        return "new values: " + ", ".join(parts)

    def _format_3line_log(self, info: dict):
        t = getattr(self.env, "t", "?")
        act = info.get("action", None)

        line1 = f"t = {t}"

        if act is None:
            line2 = "action: (none)"
        elif act.get("type") == "pct_button":
            line2 = f"action: button={act['target']}, value={act['pct']}%"
        elif act.get("type") == "step":
            d = int(act.get("dir", 0))
            dir_str = "+" if d > 0 else "-"
            line2 = f"action: button={act['target']}, value={act['step']}, direction={dir_str}"
        else:
            line2 = f"action: (unknown type={act.get('type')})"

        line3 = self._format_values_line()
        return [line1, line2, line3]


if __name__ == "__main__":
    # Use render_mode=None so the UI log is the only output.
    env = CorrectorPlayEnv(
        render_mode=None,
        # seeds (new API)
        setting_seed=0,
        init_seed=123,
        error_seed=999,
        max_steps=500,
        couple_prob_pct=0.5,
        # If you define user_gamma (even partially), missing pairs become 0 (no random coupling)
        user_gamma={'C1-A1':0.0005,
                    'A1-C1':0.24,
                    'B2-A1':-0.912/10, 'B2-C1':-1.82/10,
                    'A2-A1':-1.244/10, 'A2-C1':-0.637/10, 'A2-B2':-1.18/5,
                    'C3-C1':-0.72/100, 'C3-A1':-0.44/100, 'C3-A2':+0.967/10, 'C3-B2':+0.882/10, 'C3-S3':+0.345,
                    'S3-A1':-0.325/100,'S3-C1':-1.332/100,'S3-A2':-0.777/20,'S3-B2':-0.577/20,'S3-A3':0.2,'S3-C3':0.23},  # leave empty to enable random sparse coupling via couple_prob_pct
        user_beta={},   # leave empty to let env sample beta (if you implemented that behavior)
        user_sigma={'C1':1,'A1':1,'A2':100,'B2':20,'C3':200,'A3':100,'S3':100},  # leave empty to let env sample sigma (if you implemented that behavior)
        init_ranges={'C1':(-20,20),'A1':(0, 50), 'A2':(0,500), 'B2':(0,500),
                     'C3':(-3000,3000), 'S3':(0,3000), 'A3':(0, 3000)}
    )

    root = tk.Tk()
    app = PlayUI(root, env)
    root.mainloop()