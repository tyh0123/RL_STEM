import os
import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt
import textwrap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# Config
# ============================================================
CSV_PATH = "A1_actions.csv"
FIG_DIR = "sindy_validation_plots_A1_actions_A1"
os.makedirs(FIG_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 10
THRESHOLD = 5.0
TOL = 1e-10

# ============================================================
# Utility: equation formatting
# ============================================================
def print_clean_sindy_equation(model, target_name="A1_afteraction", tol=1e-10):
    feature_names = model.get_feature_names()
    coefficients = model.coefficients().ravel()

    terms = []
    for coef, name in zip(coefficients, feature_names):
        if abs(coef) > tol:
            terms.append(f"{coef:+.6f} * {name}")

    eq = " ".join(terms).replace("+ -", "- ")
    print(f"\n{target_name} = {eq}")

def sindy_equation_string(
    model,
    target_name="A1_afteraction",
    tol=1e-10,
    rename_map=None,
):
    feature_names = model.get_feature_names()
    coefficients = model.coefficients().ravel()

    terms = []
    for coef, name in zip(coefficients, feature_names):
        if abs(coef) > tol:
            display_name = name
            if rename_map is not None:
                for k, v in rename_map.items():
                    display_name = display_name.replace(k, v)
            terms.append(f"{coef:+.3f}·{display_name}")

    eq = " ".join(terms).replace("+ -", "- ")
    return f"{target_name} = {eq}"

def wrap_equation(eq_str, max_chars=55):
    return "\n".join(textwrap.wrap(eq_str, width=max_chars))

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(CSV_PATH)
df["strength"] = df["strength"] / 100.0

# ------------------------------------------------------------
# Filter extreme y values
# ------------------------------------------------------------
y_all = df["A1_afteraction"].to_numpy(float)
p_low, p_high = 1, 99
y_min, y_max = np.percentile(y_all, [p_low, p_high])

mask = (y_all >= y_min) & (y_all <= y_max)
print(f"Filtering y outside [{y_min:.3f}, {y_max:.3f}]")
print(f"Kept {mask.sum()} / {len(mask)} samples")

df = df.loc[mask].reset_index(drop=True)

# ------------------------------------------------------------
# Build X and y
# ------------------------------------------------------------
X = df[["A1", "strength"]].to_numpy(float)
y = df["A1_afteraction"].to_numpy(float).reshape(-1, 1)
y_true = y.ravel()

# ============================================================
# Train-test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ============================================================
# Feature library
# ============================================================
library = ps.GeneralizedLibrary([
    ps.PolynomialLibrary(
        degree=2,
        include_interaction=True,
        include_bias=False
    ),
    ps.CustomLibrary(
        library_functions=[lambda x: np.abs(x)],
        function_names=[lambda x: f"|{x}|"],
    ),
    ps.CustomLibrary(
        library_functions=[lambda x: np.sqrt(np.abs(x))],
        function_names=[lambda x: f"sqrt(|{x}|)"],
    ),
])

optimizer = ps.STLSQ(
    threshold=THRESHOLD,
    normalize_columns=True
)

model = ps.SINDy(
    optimizer=optimizer,
    feature_library=library,
    feature_names=["A1", "strength"],
)

# ============================================================
# Fit
# ============================================================
model.fit(X_train, x_dot=y_train)

# ============================================================
# Evaluation
# ============================================================
y_test_pred = model.predict(X_test).ravel()
r2_test = r2_score(y_test.ravel(), y_test_pred)

print(f"\nTest R² = {r2_test:.6f}")
print("\nGeneric SINDy print:")
model.print()

print_clean_sindy_equation(model, "A1_afteraction", TOL)

# ============================================================
# Full-dataset validation
# ============================================================
y_pred = model.predict(X).ravel()
residuals = y_true - y_pred

r2_full = 1.0 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)
rmse = np.sqrt(np.mean(residuals**2))

print(f"\nFull R² = {r2_full:.6f}")
print(f"RMSE   = {rmse:.6f}")

# ============================================================
# Build wrapped equation (DISPLAY ONLY)
# ============================================================
rename_map = {"strength": "dC1"}

eq_raw = sindy_equation_string(
    model,
    target_name="A1_afteraction",
    tol=TOL,
    rename_map=rename_map
)
eq_str = wrap_equation(eq_raw, max_chars=55)

# ============================================================
# Plot 1: full range
# ============================================================
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.6)

mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx])

plt.xlabel("Actual (A1 After Action)")
plt.ylabel("Predicted (A1 After Action)")
plt.title("SINDy: Actual vs Predicted (Full Range)")

plt.text(
    0.03, 0.97,
    f"Total R² = {r2_full:.3f}\n{eq_str}",
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "actual_vs_predicted_full.png"), dpi=300)
plt.close()

# ============================================================
# Plot 2: 99% range
# ============================================================
x_lo, x_hi = np.percentile(y_true, [0, 99])
y_lo, y_hi = np.percentile(y_pred, [0, 99])

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.6)

mn = min(x_lo, y_lo)
mx = max(x_hi, y_hi)
plt.plot([mn, mx], [mn, mx])

plt.xlim(x_lo, x_hi)
plt.ylim(y_lo, y_hi)

plt.xlabel("Actual (A1 After Action)")
plt.ylabel("Predicted (A1 After Action)")
plt.title("SINDy: Actual vs Predicted (99%)")

plt.text(
    0.03, 0.97,
    f"Total R² = {r2_full:.3f}\n{eq_str}",
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "actual_vs_predicted_99pct.png"),
    dpi=300
)
plt.close()

print(f"\nPlots saved to {FIG_DIR}")