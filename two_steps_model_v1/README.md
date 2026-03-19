# Two-Step Hierarchical Reinforcement Learning for STEM Aberration Correction

This repository implements a **two-step hierarchical reinforcement learning (RL) framework** for automated aberration correction in scanning transmission electron microscopy (STEM), inspired by practical CEOS corrector operation workflows.

The framework decomposes aberration correction into **C1A1 conditioning** and **high-order aberration optimization**, enabling stable training, interpretability, and direct extensibility to real microscopes.

------

## 2. Overview of the Two-Step Model

### Step 1: C1A1 Conditioning Policy (Low-Level)

- A dedicated **Recurrent PPO (R-PPO)** policy trained **only** on C1 and A1 actions.
- Action space limited to **C1/A1 percentage buttons**, mimicking CEOS conditioning.
- Terminates when both `|C1| < c1_gate` and `|A1| < a1_gate`.
- Purpose:
  - Rapidly bring the system into a *measurable and stable region*.
  - Prevent high-order optimization from starting in invalid states.

### Step 2: High-Order Aberration Policy (Low-Level)

- A second R-PPO policy trained to optimize **A2, B2, A3, S3, C3**.
- Guarded by the C1A1 policy:
  - If C1/A1 drift beyond gate values, control is returned to C1A1 conditioning.
- Supports configurable **gate thresholds** for each high-order aberration.

### High-Level Alternating Controller

- A deterministic meta-controller that **alternates** between:
  - Running the C1A1 policy for up to `n_c1a1_max` steps.
  - Running the high-order policy for `n_high` steps.
- Ensures:
  - C1A1 and high-order corrections never conflict.
  - Physical alignment logic matches real CEOS operation.

------

## 3. Key Design Features

### 3.1 Explicit Gate-Based Termination

Each correction stage terminates based on **physical gate conditions**, not reward heuristics:

- `C1A1_gate_ok`:
   `|C1| < c1_gate AND |A1| < a1_gate`
- `High_gate_ok`:
   All high-order aberrations below their respective gates.

This makes training **interpretable, debuggable, and physically meaningful**.

------

### 3.2 CEOS-Like Action Representation

- All aberrations (including C1 and C3 in v5) are controlled via **percentage buttons**. (During the training, I found using percentage is easier to train than absolute value. Besides, it's highly possible to calculate the absolute value based on the predicted percentage.)
- Actions correspond directly to CEOS UI operations (e.g. *A3 30%*, *B2 10%*).
- No abstract or non-physical actions are used.

------

### 3.3 Recurrent Policies for Sequential Alignment

- Both C1A1 and high-order policies use **Recurrent PPO (LSTM)**.
- Enables:
  - Memory of past correction steps.
  - Learning iterative alignment strategies rather than greedy one-shot actions.

------

### 3.4 Robust Evaluation and Traceability

- Every evaluation episode is saved as a **CSV trace**, including:
  - All actions (decoded to human-readable strings).
  - Full aberration values after each action.
  - Gate status labels (`c1a1_ok`, `high_order_ok`).
- Allows:
  - Post-hoc convergence analysis.
  - Comparison with human CEOS alignment logs.
  - Easy plotting and statistical evaluation.

------

## 4. Training Workflow

1. **Train C1A1 policy**

   ```
   Initial state → C1/A1 actions → reach C1A1 gate → terminate episode
   ```

2. **Train high-order policy with C1A1 guard**

   ```
   (C1A1 conditioning) → high-order actions → gate check → alternate
   ```