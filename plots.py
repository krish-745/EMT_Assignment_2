"""
plots.py
========
Transmission Line Waveform & Analysis Plots
============================================
Generates all 7 required plots and saves them as high-resolution PNG files.

Plots produced:
  1. |V(d)| — Voltage magnitude along the line
  2. |I(d)| — Current magnitude along the line
  3. |Gamma(d)| — Reflection coefficient magnitude along the line
  4. Zin (Real & Imaginary) vs. distance
  5. VSWR vs. frequency
  6. alpha & beta vs. frequency
  7. Standing wave envelope (V_max and V_min)

All plots are also returned as matplotlib Figure objects for embedding in GUI.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for Tkinter embedding)
import matplotlib.pyplot as plt
import os

from analytics import (
    compute_gamma,
    compute_Zo,
    compute_reflection_coefficient,
    compute_VSWR,
    compute_voltage_current,
    compute_Gamma_along_line,
    compute_Zin_along_line,
    sweep_frequency,
    sweep_frequency_vswr,
)

# Output directory for saved PNGs
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_output")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style helper
# ---------------------------------------------------------------------------
COLORS = {
    "voltage":   "#1f77b4",
    "current":   "#ff7f0e",
    "gamma":     "#2ca02c",
    "zin_real":  "#d62728",
    "zin_imag":  "#9467bd",
    "vswr":      "#8c564b",
    "alpha":     "#e377c2",
    "beta":      "#7f7f7f",
    "envelope":  "#17becf",
}

def _style_axes(ax, title, xlabel, ylabel, grid=True):
    """Apply consistent styling to an axes object."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_fig(fig, filename):
    """Save figure to plots_output directory as 150-dpi PNG."""
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Plot 1: |V(d)| vs distance
# ---------------------------------------------------------------------------
def plot_voltage(R, L, G, C, f, d_total, ZL_real, ZL_imag, n=300, save=True):
    """
    Plot voltage magnitude |V(z)| along the transmission line.

    z=0 at load, z=d_total at source.
    """
    ZL    = complex(ZL_real, ZL_imag)
    gamma, alpha, beta = compute_gamma(R, L, G, C, f)
    Zo    = compute_Zo(R, L, G, C, f)
    Gamma_L = compute_reflection_coefficient(ZL, Zo)

    z_arr = np.linspace(0, d_total, n)
    V, _  = compute_voltage_current(z_arr, Gamma_L, gamma, Zo, V_plus=1.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_arr, np.abs(V), color=COLORS["voltage"], linewidth=2)
    _style_axes(ax,
                title="|V(z)| — Voltage Magnitude Along Transmission Line",
                xlabel="Distance from Load  z  (m)",
                ylabel="|V(z)|  (V, normalized)")
    ax.fill_between(z_arr, np.abs(V), alpha=0.15, color=COLORS["voltage"])
    ax.annotate(f"f = {f/1e6:.2f} MHz | d = {d_total} m",
                xy=(0.98, 0.96), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="gray")
    fig.tight_layout()
    if save:
        _save_fig(fig, "01_voltage_magnitude.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 2: |I(d)| vs distance
# ---------------------------------------------------------------------------
def plot_current(R, L, G, C, f, d_total, ZL_real, ZL_imag, n=300, save=True):
    """Plot current magnitude |I(z)| along the transmission line."""
    ZL    = complex(ZL_real, ZL_imag)
    gamma, alpha, beta = compute_gamma(R, L, G, C, f)
    Zo    = compute_Zo(R, L, G, C, f)
    Gamma_L = compute_reflection_coefficient(ZL, Zo)

    z_arr = np.linspace(0, d_total, n)
    _, I  = compute_voltage_current(z_arr, Gamma_L, gamma, Zo, V_plus=1.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_arr, np.abs(I), color=COLORS["current"], linewidth=2)
    _style_axes(ax,
                title="|I(z)| — Current Magnitude Along Transmission Line",
                xlabel="Distance from Load  z  (m)",
                ylabel="|I(z)|  (A, normalized)")
    ax.fill_between(z_arr, np.abs(I), alpha=0.15, color=COLORS["current"])
    ax.annotate(f"f = {f/1e6:.2f} MHz | d = {d_total} m",
                xy=(0.98, 0.96), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="gray")
    fig.tight_layout()
    if save:
        _save_fig(fig, "02_current_magnitude.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 3: |Gamma(z)| vs distance
# ---------------------------------------------------------------------------
def plot_reflection_coefficient(R, L, G, C, f, d_total, ZL_real, ZL_imag,
                                n=300, save=True):
    """Plot |Gamma(z)| — reflection coefficient magnitude along the line."""
    ZL    = complex(ZL_real, ZL_imag)
    gamma, _, _ = compute_gamma(R, L, G, C, f)
    Zo    = compute_Zo(R, L, G, C, f)
    Gamma_L = compute_reflection_coefficient(ZL, Zo)

    z_arr   = np.linspace(0, d_total, n)
    Gamma_z = compute_Gamma_along_line(z_arr, Gamma_L, gamma)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_arr, np.abs(Gamma_z), color=COLORS["gamma"], linewidth=2)
    _style_axes(ax,
                title="|Γ(z)| — Reflection Coefficient Magnitude Along Line",
                xlabel="Distance from Load  z  (m)",
                ylabel="|Γ(z)|  (dimensionless)")
    ax.axhline(y=np.abs(Gamma_L), linestyle="--", color="gray",
               linewidth=1, label=f"|ΓL| = {np.abs(Gamma_L):.3f}")
    ax.legend(fontsize=8)
    ax.fill_between(z_arr, np.abs(Gamma_z), alpha=0.15, color=COLORS["gamma"])
    fig.tight_layout()
    if save:
        _save_fig(fig, "03_reflection_coefficient.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Zin (Real & Imaginary) vs distance
# ---------------------------------------------------------------------------
def plot_Zin(R, L, G, C, f, d_total, ZL_real, ZL_imag, n=300, save=True):
    """Plot real and imaginary parts of Zin vs. distance from the load."""
    ZL    = complex(ZL_real, ZL_imag)
    gamma, _, _ = compute_gamma(R, L, G, C, f)
    Zo    = compute_Zo(R, L, G, C, f)

    z_arr  = np.linspace(1e-6, d_total, n)   # avoid z=0 (ZL itself)
    Zin_z  = compute_Zin_along_line(z_arr, Zo, ZL, gamma)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_arr, np.real(Zin_z), color=COLORS["zin_real"],
            linewidth=2, label="Re(Zin)")
    ax.plot(z_arr, np.imag(Zin_z), color=COLORS["zin_imag"],
            linewidth=2, label="Im(Zin)", linestyle="--")
    _style_axes(ax,
                title="Zin(z) — Input Impedance Along Transmission Line",
                xlabel="Distance from Load  z  (m)",
                ylabel="Impedance  (Ω)")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    if save:
        _save_fig(fig, "04_Zin_vs_distance.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 5: VSWR vs frequency
# ---------------------------------------------------------------------------
def plot_VSWR_vs_frequency(R, L, G, C, ZL_real, ZL_imag,
                           f_start=1e6, f_end=1e9, n=300, save=True):
    """Plot VSWR vs. frequency sweep."""
    freqs, vswrs = sweep_frequency_vswr(R, L, G, C, ZL_real, ZL_imag,
                                        f_start, f_end, n)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs / 1e6, vswrs, color=COLORS["vswr"], linewidth=2)
    _style_axes(ax,
                title="VSWR vs. Frequency",
                xlabel="Frequency  (MHz)",
                ylabel="VSWR  (dimensionless)")
    ax.axhline(y=1.0, linestyle="--", color="black", linewidth=1,
               label="VSWR = 1 (perfect match)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        _save_fig(fig, "05_VSWR_vs_frequency.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 6: alpha & beta vs frequency
# ---------------------------------------------------------------------------
def plot_alpha_beta_vs_frequency(R, L, G, C,
                                 f_start=1e6, f_end=1e9, n=300, save=True):
    """Plot attenuation (alpha) and phase constant (beta) vs. frequency."""
    freqs, alphas, betas, _ = sweep_frequency(R, L, G, C, f_start, f_end, n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.plot(freqs / 1e6, alphas, color=COLORS["alpha"], linewidth=2)
    _style_axes(ax1,
                title="Attenuation Constant α vs. Frequency",
                xlabel="",
                ylabel="α  (Np/m)")

    ax2.plot(freqs / 1e6, betas, color=COLORS["beta"], linewidth=2)
    _style_axes(ax2,
                title="Phase Constant β vs. Frequency",
                xlabel="Frequency  (MHz)",
                ylabel="β  (rad/m)")

    fig.suptitle("Propagation Parameters vs. Frequency",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save:
        _save_fig(fig, "06_alpha_beta_vs_frequency.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 7: Standing Wave Envelope
# ---------------------------------------------------------------------------
def plot_standing_wave(R, L, G, C, f, d_total, ZL_real, ZL_imag,
                       n=500, save=True):
    """
    Plot the standing wave pattern — voltage envelope (V_max and V_min).

    Envelope:
        V_max(z) = |V+| * (1 + |Gamma(z)|)   ← constructive
        V_min(z) = |V+| * (1 - |Gamma(z)|)   ← destructive
    """
    ZL    = complex(ZL_real, ZL_imag)
    gamma, _, _ = compute_gamma(R, L, G, C, f)
    Zo    = compute_Zo(R, L, G, C, f)
    Gamma_L = compute_reflection_coefficient(ZL, Zo)

    z_arr   = np.linspace(0, d_total, n)
    Gamma_z = compute_Gamma_along_line(z_arr, Gamma_L, gamma)
    V_max   = 1 + np.abs(Gamma_z)
    V_min   = np.abs(1 - np.abs(Gamma_z))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_arr, V_max, color=COLORS["voltage"], linewidth=2,
            label=r"$V_{max}$ (constructive)")
    ax.plot(z_arr, V_min, color=COLORS["current"], linewidth=2,
            linestyle="--", label=r"$V_{min}$ (destructive)")
    ax.fill_between(z_arr, V_min, V_max, alpha=0.15,
                    color=COLORS["envelope"], label="Standing wave region")
    _style_axes(ax,
                title="Standing Wave Envelope Along Transmission Line",
                xlabel="Distance from Load  z  (m)",
                ylabel="Voltage Envelope  (normalized)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        _save_fig(fig, "07_standing_wave_envelope.png")
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all 7 plots at once
# ---------------------------------------------------------------------------
def generate_all_plots(R, L, G, C, f, d_total, ZL_real, ZL_imag,
                       f_start=1e6, f_end=1e9, save=True):
    """
    Generate all 7 plots and return a list of (title, Figure) tuples.
    """
    figs = []

    figs.append(("Voltage |V(z)|",
                 plot_voltage(R, L, G, C, f, d_total, ZL_real, ZL_imag,
                              save=save)))
    figs.append(("Current |I(z)|",
                 plot_current(R, L, G, C, f, d_total, ZL_real, ZL_imag,
                              save=save)))
    figs.append(("Reflection |Γ(z)|",
                 plot_reflection_coefficient(R, L, G, C, f, d_total,
                                             ZL_real, ZL_imag, save=save)))
    figs.append(("Zin vs z",
                 plot_Zin(R, L, G, C, f, d_total, ZL_real, ZL_imag,
                          save=save)))
    figs.append(("VSWR vs f",
                 plot_VSWR_vs_frequency(R, L, G, C, ZL_real, ZL_imag,
                                        f_start, f_end, save=save)))
    figs.append(("α & β vs f",
                 plot_alpha_beta_vs_frequency(R, L, G, C, f_start, f_end,
                                              save=save)))
    figs.append(("Standing Wave",
                 plot_standing_wave(R, L, G, C, f, d_total,
                                    ZL_real, ZL_imag, save=save)))
    return figs
