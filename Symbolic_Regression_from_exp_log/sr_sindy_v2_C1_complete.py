import os
import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# Config
# ============================================================
CSV_PATH = "C1_actions.csv"
FIG_DIR = "sindy_validation_plots"
os.makedirs(FIG_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 23
THRESHOLD = 2.0
TOL = 1e-10

# ============================================================
# Utility: clean equation printer
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

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(CSV_PATH)

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
# Feature library (CORRECT + SAFE)
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

# ============================================================
# Full-dataset validation
# ============================================================
A1_pred = model.predict(X).ravel()
residuals = A1_true - A1_pred

r2_full = 1.0 - np.sum(residuals**2) / np.sum((A1_true - A1_true.mean())**2)
rmse = np.sqrt(np.mean(residuals**2))

print(f"\nFull R² = {r2_full:.6f}")
print(f"RMSE   = {rmse:.6f}")

# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(6, 6))
plt.scatter(A1_true, A1_pred, alpha=0.6)
mn, mx = min(A1_true.min(), A1_pred.min()), max(A1_true.max(), A1_pred.max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SINDy: Actual vs Predicted")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "actual_vs_predicted.png"), dpi=300)
plt.close()

print(f"\nPlots saved to {FIG_DIR}")