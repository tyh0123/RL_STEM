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
CSV_PATH = "result/SR/C1_actions.csv"
FIG_DIR = "sindy_validation_plots"
os.makedirs(FIG_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 47
THRESHOLD = 50.0
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

def sindy_equation_string(model, target_name="A1_afteraction", tol=1e-10):
    feature_names = model.get_feature_names()
    coefficients = model.coefficients().ravel()

    terms = []
    for coef, name in zip(coefficients, feature_names):
        if abs(coef) > tol:
            terms.append(f"{coef:+.3f}·{name}")

    eq = " ".join(terms).replace("+ -", "- ")
    return f"{target_name} = {eq}"

def wrap_equation(eq_str, max_chars=55):
    """
    Wrap equation string into multiple lines for plotting.
    """
    return "\n".join(textwrap.wrap(eq_str, width=max_chars))

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(CSV_PATH)

# ------------------------------------------------------------
# Filter small-strength samples
# ------------------------------------------------------------
strength_all = df["strength"].to_numpy(float)
strength_mask = np.abs(strength_all) >= 1.0
print(f"Filtering |strength| < 1")
print(f"Kept {strength_mask.sum()} / {len(strength_mask)} samples")

df = df.loc[strength_mask].reset_index(drop=True)

# ------------------------------------------------------------
# Filter extreme C1 values
# ------------------------------------------------------------
C1_all = df["C1"].to_numpy(float)
C1_max_abs = 100.0
C1_mask = np.abs(C1_all) <= C1_max_abs
print(f"Filtering |C1| > {C1_max_abs}")
print(f"Kept {C1_mask.sum()} / {len(C1_mask)} samples")

df = df.loc[C1_mask].reset_index(drop=True)

# ------------------------------------------------------------
# Build X and y
# ------------------------------------------------------------
X = df[["C1", "A1", "strength"]].to_numpy(float)
y = df["A1_afteraction"].to_numpy(float).reshape(-1, 1)
A1_true = y.ravel()

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
    feature_names=["C1", "A1", "strength"],
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

# ------------------------------------------------------------
# Full-dataset validation
# ------------------------------------------------------------
A1_pred = model.predict(X).ravel()
residuals = A1_true - A1_pred

r2_full = 1.0 - np.sum(residuals**2) / np.sum((A1_true - A1_true.mean())**2)
rmse = np.sqrt(np.mean(residuals**2))

print(f"\nFull R² = {r2_full:.6f}")
print(f"RMSE   = {rmse:.6f}")

# Build wrapped equation string (ONCE)
eq_raw = sindy_equation_string(model, "A1_afteraction", TOL)
eq_raw = eq_raw.replace("strength", "dC1")
eq_str = wrap_equation(eq_raw, max_chars=55)

# ============================================================
# Plot: full range
# ============================================================
plt.figure(figsize=(6, 6))
plt.scatter(A1_true, A1_pred, alpha=0.6)

mn, mx = min(A1_true.min(), A1_pred.min()), max(A1_true.max(), A1_pred.max())
plt.plot([mn, mx], [mn, mx])

plt.xlabel("Actual (A1 After Action)")
plt.ylabel("Predicted (A1 After Action)")
plt.title("SINDy SR: Actual vs Predicted (Full Range)")

plt.text(
    0.03, 0.97,
    f"Total R² = {r2_full:.3f}\n{eq_str}",
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "actual_vs_predicted_a1.png"), dpi=300)
plt.close()

# ============================================================
# Plot: central 80%
# ============================================================
x_lo, x_hi = np.percentile(A1_true, [0, 95])
y_lo, y_hi = np.percentile(A1_pred, [0, 95])

plt.figure(figsize=(6, 6))
plt.scatter(A1_true, A1_pred, alpha=0.6)

mn = min(x_lo, y_lo)
mx = max(x_hi, y_hi)
plt.plot([mn, mx], [mn, mx])

plt.xlim(x_lo, x_hi)
plt.ylim(y_lo, y_hi)

plt.xlabel("Actual A1 After Action")
plt.ylabel("Predicted A1 After Action")
plt.title("SINDy SR: Actual vs Predicted (zoomed)")

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
    os.path.join(FIG_DIR, "actual_vs_predicted_a1_95pct.png"),
    dpi=300
)
plt.close()

print(f"\nPlots saved to {FIG_DIR}")