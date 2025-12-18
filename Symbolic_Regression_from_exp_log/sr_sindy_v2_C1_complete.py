import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# =========================
# Load data
# =========================
df = pd.read_csv("C1_actions.csv")

C1 = df["C1"].to_numpy(float)
A1 = df["A1"].to_numpy(float)
strength = df["strength"].to_numpy(float)
A1_true = df["A1_afteraction"].to_numpy(float)

# =========================
# SINDy symbolic model (RAW UNITS)
# =========================
def sindy_predict_A1(C1, A1, strength):
    return (
        0.082254 * C1
        + 0.518061 * A1
        + 0.148909 * strength
        - 0.002614 * (C1 * A1)
        - 0.002036 * (A1 ** 2)
        - 0.002376 * (A1 * strength)
        + 0.000626 * (strength ** 2)
        - 0.023967 * np.abs(C1)
        + 0.518061 * np.abs(A1)
        - 0.306736 * np.abs(strength)
        - 0.681305 * np.sqrt(np.abs(C1))
        + 2.081753 * np.sqrt(np.abs(A1))
    )

# =========================
# Prediction
# =========================
A1_pred = sindy_predict_A1(C1, A1, strength)

# =========================
# Metrics
# =========================
residuals = A1_true - A1_pred

ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((A1_true - A1_true.mean()) ** 2)

r2 = 1.0 - ss_res / ss_tot
rmse = np.sqrt(np.mean(residuals ** 2))

print(f"R²   = {r2:.6f}")
print(f"RMSE = {rmse:.6f}")

# =========================
# 1) Actual vs Predicted
# =========================
plt.figure(figsize=(6, 6))
plt.scatter(A1_true, A1_pred, alpha=0.6)
mn = min(A1_true.min(), A1_pred.min())
mx = max(A1_true.max(), A1_pred.max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual A1_afteraction")
plt.ylabel("Predicted A1_afteraction")
plt.title("SINDy: Actual vs Predicted")
plt.tight_layout()
plt.show()



# =========================
# 2) Time / index series
# =========================
# --- Actual vs Predicted (zoomed, percentile-based) ---
plt.figure(figsize=(6, 6))
plt.scatter(A1_true, A1_pred, alpha=0.6)

# Percentile-based limits
p_low, p_high = 1, 99
xmin, xmax = np.percentile(A1_true, [p_low, p_high])
ymin, ymax = np.percentile(A1_pred, [p_low, p_high])

plt.plot([xmin, xmax], [xmin, xmax])
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.xlabel("Actual A1_afteraction")
plt.ylabel("Predicted A1_afteraction")
plt.title("SINDy: Actual vs Predicted (1–99% zoom)")
plt.tight_layout()
plt.show()
plt.savefig("actual_vs_predicted_zoomed.png", dpi=300)
plt.close()
# =========================
# 3) Residuals
# =========================
plt.figure(figsize=(12, 4))
plt.plot(residuals)
plt.axhline(0)
plt.xlabel("Sample index")
plt.ylabel("Residual")
plt.title("Residuals")
plt.tight_layout()
plt.show()

# =========================
# 4) Residual histogram
# =========================
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=40)
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residual Histogram")
plt.tight_layout()
plt.show()