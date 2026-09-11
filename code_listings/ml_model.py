"""
ml_model.py
===========
Machine Learning Model for Transmission Line Parameter Prediction
=================================================================
Uses per-target Random Forest / Gradient Boosting Regressors with
comprehensive physics-informed feature engineering to achieve R^2 >= 0.97.

Strategy:
  - |Zo| and beta are nearly analytically deterministic -> RF excels
  - alpha depends on R,G,omega -> GB excels with engineered features
  - |Gamma_L| and VSWR depend on ZL vs Zo -> add Zo-related features

Training Pipeline:
  1. Generate 1000 synthetic data points in realistic coaxial/stripline ranges
  2. First analytically estimate Zo, then add it as a feature for Gamma/VSWR
  3. Log-transform wide-range features and targets
  4. Train best-performing model per target (RF or GB)
  5. Evaluate: R^2 score and MAPE on a held-out test set (20%)

Output targets (5):
    |Zo|, alpha, beta, |Gamma_L|, VSWR
"""

import numpy as np
import os
import pickle
import warnings

from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from analytics import (compute_all, compute_gamma, compute_Zo,
                       compute_reflection_coefficient, compute_VSWR)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL = os.path.join(_DIR, "ml_model.pkl")

# ---------------------------------------------------------------------------
# Realistic coaxial-cable parameter ranges
# ---------------------------------------------------------------------------
PARAM_RANGES = {
    "R":       (0.01,  150.0),   # Ohm/m
    "L":       (50e-9, 600e-9),  # H/m
    "G":       (1e-6,  1e-2),    # S/m
    "C":       (10e-12,300e-12), # F/m
    "f":       (1e6,   3e9),     # Hz
    "d":       (0.05,  20.0),    # m
    "|ZL|":    (5.0,   500.0),   # Ohm
    "aZL":     (-75.0, 75.0),    # degrees
}

TARGET_NAMES  = ["|Zo|", "alpha", "beta", "|Gamma_L|", "VSWR"]
FEATURE_NAMES = [
    "R", "log_L", "log_G", "log_C", "log_f", "d",
    "|ZL|", "aZL",
    "log_omega", "log_wL", "log_wC",
    "log_|Zs|", "log_|Yp|", "log_Q",
    "log_Zo_approx",          # approx Zo = sqrt(L/C)
    "ZL_over_Zo",             # |ZL|/Zo_approx
    "ZL_real_over_Zo",        # Re(ZL)/Zo_approx
    "ZL_imag_over_Zo",        # Im(ZL)/Zo_approx
    "log_alpha_approx",       # approx alpha = R/(2*Zo) + G*Zo/2
]


# ---------------------------------------------------------------------------
# Feature engineering (physics-informed)
# ---------------------------------------------------------------------------
def _engineer_features(R, L, G, C, f, d, ZL_mag, ZL_angle_deg):
    """Build enriched feature matrix with physics-derived features."""
    omega    = 2 * np.pi * f
    wL       = omega * L
    wC       = omega * C
    Z_series = np.sqrt(R**2 + wL**2)
    Y_shunt  = np.sqrt(G**2 + wC**2)
    Q        = wL / (R + 1e-15)

    # Approximate Zo from lossless formula (good first-order estimate)
    Zo_approx = np.sqrt(L / (C + 1e-30))

    # ZL components
    ZL_real = ZL_mag * np.cos(np.radians(ZL_angle_deg))
    ZL_imag = ZL_mag * np.sin(np.radians(ZL_angle_deg))

    # ZL vs Zo features
    ZL_over_Zo      = ZL_mag   / (Zo_approx + 1e-15)
    ZLr_over_Zo     = ZL_real  / (Zo_approx + 1e-15)
    ZLi_over_Zo     = ZL_imag  / (Zo_approx + 1e-15)

    # Approximate alpha (low-loss formula)
    alpha_approx = R / (2 * Zo_approx + 1e-15) + G * Zo_approx / 2

    X = np.column_stack([
        R,
        np.log10(np.maximum(L,        1e-15)),
        np.log10(np.maximum(G,        1e-15)),
        np.log10(np.maximum(C,        1e-15)),
        np.log10(np.maximum(f,        1.0)),
        d,
        ZL_mag,
        ZL_angle_deg,
        np.log10(np.maximum(omega,    1.0)),
        np.log10(np.maximum(wL,       1e-15)),
        np.log10(np.maximum(wC,       1e-15)),
        np.log10(np.maximum(Z_series, 1e-15)),
        np.log10(np.maximum(Y_shunt,  1e-15)),
        np.log10(np.maximum(Q,        1e-15)),
        np.log10(np.maximum(Zo_approx,1e-15)),
        ZL_over_Zo,
        ZLr_over_Zo,
        ZLi_over_Zo,
        np.log10(np.maximum(alpha_approx, 1e-15)),
    ])
    return X


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_dataset(n_samples=1000, random_seed=42):
    """
    Generate synthetic dataset by randomly sampling Tx line parameters
    and computing exact analytical labels.

    Returns
    -------
    X      : ndarray (n_samples, 19)  -- engineered features
    y_raw  : ndarray (n_samples, 5)   -- raw targets
    y_log  : ndarray (n_samples, 5)   -- log/linear transformed targets
    """
    rng = np.random.default_rng(random_seed)

    def U(key, n):
        lo, hi = PARAM_RANGES[key]
        return rng.uniform(lo, hi, n)

    R        = U("R",   n_samples)
    L        = U("L",   n_samples)
    G        = U("G",   n_samples)
    C        = U("C",   n_samples)
    f        = U("f",   n_samples)
    d        = U("d",   n_samples)
    ZL_mag   = U("|ZL|",n_samples)
    ZL_angle = U("aZL", n_samples)

    ZL_real = ZL_mag * np.cos(np.radians(ZL_angle))
    ZL_imag = ZL_mag * np.sin(np.radians(ZL_angle))

    X = _engineer_features(R, L, G, C, f, d, ZL_mag, ZL_angle)

    y_raw = np.zeros((n_samples, 5))
    for i in range(n_samples):
        res = compute_all(R[i], L[i], G[i], C[i], f[i], d[i],
                          ZL_real[i], ZL_imag[i])
        y_raw[i, 0] = res["|Zo|"]
        y_raw[i, 1] = res["alpha"]
        y_raw[i, 2] = res["beta"]
        y_raw[i, 3] = res["|Gamma_L|"]
        y_raw[i, 4] = min(res["VSWR"], 50.0)

    # Log-transform for wide-range targets, keep |Gamma_L| linear
    y_log = np.column_stack([
        np.log10(np.maximum(y_raw[:, 0], 1e-15)),   # log|Zo|
        np.log10(np.maximum(y_raw[:, 1], 1e-15)),   # log alpha
        np.log10(np.maximum(y_raw[:, 2], 1e-15)),   # log beta
        y_raw[:, 3],                                  # |Gamma_L| linear [0,1]
        np.log10(np.maximum(y_raw[:, 4], 1.0)),      # log VSWR
    ])

    return X, y_raw, y_log


# ---------------------------------------------------------------------------
# Select best algorithm per target
# ---------------------------------------------------------------------------
def _best_model_for_target(j):
    """
    Return the best sklearn model for each target index.
    GradientBoosting generalises better on harder, nonlinear targets.
    """
    configs = [
        # |Zo| -- nearly analytical from L,C
        RandomForestRegressor(
            n_estimators=600, max_features=0.6,
            min_samples_leaf=1, n_jobs=-1, random_state=j),
        # alpha -- sensitive to R, G, omega
        GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            min_samples_leaf=2, random_state=j),
        # beta -- nearly analytical from L, C, omega
        RandomForestRegressor(
            n_estimators=600, max_features=0.6,
            min_samples_leaf=1, n_jobs=-1, random_state=j),
        # |Gamma_L| -- depends on Zo vs ZL ratio
        GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            min_samples_leaf=2, random_state=j),
        # VSWR -- monotonic function of |Gamma_L|
        GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            min_samples_leaf=2, random_state=j),
    ]
    return configs[j]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_model(n_samples=1000, test_size=0.2, random_seed=42):
    """
    Train one model per target on synthetic Tx line data.

    Returns
    -------
    model_bundle : dict
    """
    print(f"[ML] Generating {n_samples} synthetic data points...")
    X, y_raw, y_log = generate_dataset(n_samples=n_samples,
                                       random_seed=random_seed)

    X_tr, X_te, \
    y_log_tr, y_log_te, \
    y_raw_tr, y_raw_te = train_test_split(
        X, y_log, y_raw,
        test_size=test_size, random_state=random_seed
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models      = []
    r2_scores   = []
    mape_scores = []

    print("[ML] Training one model per target...")
    for j, name in enumerate(TARGET_NAMES):
        model = _best_model_for_target(j)
        model.fit(X_tr_s, y_log_tr[:, j])
        models.append(model)

        pred_log = model.predict(X_te_s)
        r2 = r2_score(y_log_te[:, j], pred_log)

        # MAPE in original space
        if name in ("|Zo|", "alpha", "beta", "VSWR"):
            y_orig  = 10 ** y_log_te[:, j]
            p_orig  = 10 ** pred_log
        else:   # |Gamma_L|
            y_orig  = y_log_te[:, j]
            p_orig  = pred_log

        mask = np.abs(y_orig) > 1e-10
        mape = mean_absolute_percentage_error(
            y_orig[mask], p_orig[mask]) * 100 if mask.sum() > 0 else 0.0

        r2_scores.append(r2)
        mape_scores.append(mape)

        algo = "RF" if isinstance(model, RandomForestRegressor) else "GB"
        print(f"  [{j+1}/5] {name:<14} [{algo}]  R^2 = {r2:.4f}"
              f"   MAPE = {mape:.2f}%")

    y_pred_all = np.column_stack([m.predict(X_te_s) for m in models])
    r2_overall = r2_score(y_log_te, y_pred_all)

    print("\n[ML] Training Complete -- Performance on Test Set")
    print("=" * 55)
    print(f"  {'Target':<14}  {'R^2':>8}  {'MAPE (%)':>10}")
    print("-" * 55)
    for j, name in enumerate(TARGET_NAMES):
        print(f"  {name:<14}  {r2_scores[j]:>8.4f}  {mape_scores[j]:>10.2f}")
    print("-" * 55)
    print(f"  {'Overall R^2':<14}  {r2_overall:>8.4f}")
    print("=" * 55)

    if r2_overall >= 0.97:
        print("[ML] PASS: Target accuracy (R^2 >= 0.97) ACHIEVED.")
    else:
        print(f"[ML] INFO: Overall R^2 = {r2_overall:.4f}"
              f"  (target >= 0.97)")

    # Average feature importance
    fi_list = []
    for m in models:
        if hasattr(m, "feature_importances_"):
            fi_list.append(m.feature_importances_)
    fi = np.mean(fi_list, axis=0) if fi_list else np.zeros(X.shape[1])

    bundle = {
        "scaler":              scaler,
        "models":              models,
        "X_test":              X_te,
        "y_log_test":          y_log_te,
        "y_pred_all":          y_pred_all,
        "r2_scores":           r2_scores,
        "mape_scores":         mape_scores,
        "r2_overall":          r2_overall,
        "feature_importances": fi,
        "feature_names":       FEATURE_NAMES,
        "target_names":        TARGET_NAMES,
    }
    return bundle


# ---------------------------------------------------------------------------
# Save / Load model
# ---------------------------------------------------------------------------
def save_model(bundle, path=MODEL_PKL):
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[ML] Model saved to: {path}")


def load_model(path=MODEL_PKL):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"[ML] Model loaded from: {path}")
    return bundle


def get_or_train_model(n_samples=1000, force_retrain=False):
    """Return trained model bundle. Load from disk or train fresh."""
    if not force_retrain and os.path.exists(MODEL_PKL):
        try:
            return load_model()
        except Exception:
            pass
    bundle = train_model(n_samples=n_samples)
    save_model(bundle)
    return bundle


# ---------------------------------------------------------------------------
# Prediction interface
# ---------------------------------------------------------------------------
def predict(bundle, R, L, G, C, f, d, ZL_mag, ZL_angle_deg):
    """
    Predict Tx line outputs using the trained per-target models.

    Returns dict with predicted |Zo|, alpha, beta, |Gamma_L|, VSWR.
    """
    scaler = bundle["scaler"]
    models = bundle["models"]

    x = _engineer_features(
        np.array([R]), np.array([L]), np.array([G]), np.array([C]),
        np.array([f]), np.array([d]),
        np.array([ZL_mag]), np.array([ZL_angle_deg])
    )   # shape (1, 19)

    x_s = scaler.transform(x)
    preds = [m.predict(x_s)[0] for m in models]

    return {
        "|Zo|_pred":      10 ** preds[0],
        "alpha_pred":     10 ** preds[1],
        "beta_pred":      10 ** preds[2],
        "|Gamma_L|_pred": np.clip(preds[3], 0.0, 1.0),
        "VSWR_pred":      max(1.0, 10 ** preds[4]),
    }


# ---------------------------------------------------------------------------
# ML Performance Report Plot
# ---------------------------------------------------------------------------
def plot_ml_performance(bundle, save=True):
    """
    2x3 figure: actual vs predicted for each target + feature importance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_log_te   = bundle["y_log_test"]
    y_pred_all = bundle["y_pred_all"]
    r2_scores  = bundle["r2_scores"]
    fi         = bundle["feature_importances"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for j, name in enumerate(TARGET_NAMES):
        ax = axes[j]
        ax.scatter(y_log_te[:, j], y_pred_all[:, j],
                   alpha=0.5, s=18, color=colors[j], edgecolors="none")
        mn = min(y_log_te[:, j].min(), y_pred_all[:, j].min())
        mx = max(y_log_te[:, j].max(), y_pred_all[:, j].max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="Ideal")
        lbl = f"log10({name})" if name not in ("|Gamma_L|",) else name
        ax.set_title(f"{lbl}  (R^2 = {r2_scores[j]:.4f})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Feature importance
    ax6 = axes[5]
    fi_arr = np.array(fi)
    n_top  = min(12, len(fi_arr))
    idx    = np.argsort(fi_arr)[-n_top:]
    fn     = bundle["feature_names"]
    ax6.barh([fn[i] for i in idx], fi_arr[idx],
             color="#7f7f7f", alpha=0.8)
    ax6.set_title("Avg. Feature Importances",
                  fontsize=10, fontweight="bold")
    ax6.set_xlabel("Importance", fontsize=9)
    ax6.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    fig.suptitle(
        f"ML Model Performance  --  RF/GB (per-target)"
        f"  |  Overall R^2 = {bundle['r2_overall']:.4f}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()

    if save:
        from plots import PLOT_DIR
        path = os.path.join(PLOT_DIR, "08_ml_performance.png")
        os.makedirs(PLOT_DIR, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[ML] Performance plot saved: {path}")

    return fig


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bundle = get_or_train_model(n_samples=1000, force_retrain=True)
    plot_ml_performance(bundle, save=True)
    print("\n[ML] Sample prediction:")
    pred = predict(bundle,
                   R=10, L=250e-9, G=1e-4, C=100e-12,
                   f=100e6, d=2.0,
                   ZL_mag=75, ZL_angle_deg=30)
    for k, v in pred.items():
        print(f"  {k:<22} = {v:.6g}")
