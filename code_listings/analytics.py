"""
analytics.py
============
Transmission Line Analytics Engine
====================================
Computes all fundamental transmission line parameters for a general lossy line.

Parameters handled:
  - Distributed: R (Ohm/m), L (H/m), G (S/m), C (F/m)
  - Frequency: f (Hz)
  - Line length: d (m)
  - Load impedance: ZL (complex, Ohm)

Outputs computed:
  - Zo    : Characteristic impedance (complex, Ohm)
  - gamma : Propagation constant = alpha + j*beta  (complex, 1/m)
  - alpha : Attenuation constant (Np/m)
  - beta  : Phase constant (rad/m)
  - Gamma : Reflection coefficient at the load (complex)
  - VSWR  : Voltage Standing Wave Ratio
  - Zin   : Input impedance at distance d from load (complex, Ohm)
  - V(z)  : Voltage distribution along line
  - I(z)  : Current distribution along line
"""

import numpy as np


# ---------------------------------------------------------------------------
# Primary parameter functions
# ---------------------------------------------------------------------------

def compute_gamma(R, L, G, C, f):
    """
    Compute the complex propagation constant gamma = alpha + j*beta.

    Formula:  gamma = sqrt((R + j*omega*L)(G + j*omega*C))

    Parameters
    ----------
    R, L, G, C : float  -- distributed line parameters per unit length
    f          : float  -- frequency (Hz)

    Returns
    -------
    gamma : complex  (1/m)
    alpha : float    attenuation constant (Np/m)
    beta  : float    phase constant (rad/m)
    """
    omega = 2 * np.pi * f
    Z_series = R + 1j * omega * L   # series impedance per unit length
    Y_shunt  = G + 1j * omega * C   # shunt admittance per unit length
    gamma = np.sqrt(Z_series * Y_shunt)

    # Ensure the principal branch has a non-negative real part (physical)
    if np.real(gamma) < 0:
        gamma = -gamma

    alpha = np.real(gamma)
    beta  = np.imag(gamma)
    return gamma, alpha, beta


def compute_Zo(R, L, G, C, f):
    """
    Compute characteristic impedance Zo.

    Formula:  Zo = sqrt((R + j*omega*L) / (G + j*omega*C))

    Returns
    -------
    Zo : complex (Ohm)
    """
    omega = 2 * np.pi * f
    Z_series = R + 1j * omega * L
    Y_shunt  = G + 1j * omega * C
    Zo = np.sqrt(Z_series / Y_shunt)
    # Ensure positive real part
    if np.real(Zo) < 0:
        Zo = -Zo
    return Zo


def compute_reflection_coefficient(ZL, Zo):
    """
    Compute reflection coefficient at the load.

    Formula:  Gamma_L = (ZL - Zo) / (ZL + Zo)

    Returns
    -------
    Gamma_L : complex
    """
    return (ZL - Zo) / (ZL + Zo)


def compute_VSWR(Gamma_L):
    """
    Compute Voltage Standing Wave Ratio.

    Formula:  VSWR = (1 + |Gamma_L|) / (1 - |Gamma_L|)

    For a matched load (Gamma_L = 0), VSWR = 1.
    For a total reflection (|Gamma_L| = 1), VSWR -> infinity.

    Returns
    -------
    vswr : float  (1 <= VSWR < inf)
    """
    mag = np.abs(Gamma_L)
    mag = np.clip(mag, 0.0, 0.9999)   # avoid division by zero
    vswr = (1 + mag) / (1 - mag)
    return vswr


def compute_Zin(Zo, ZL, gamma, d):
    """
    Compute input impedance at distance d from the load.

    Formula:
        Zin = Zo * (ZL + Zo*tanh(gamma*d)) / (Zo + ZL*tanh(gamma*d))

    This is valid for both lossy and lossless lines.
    For lossless lines (alpha=0), tanh(j*beta*d) = j*tan(beta*d).

    Parameters
    ----------
    Zo    : complex  characteristic impedance
    ZL    : complex  load impedance
    gamma : complex  propagation constant
    d     : float    distance from load (m)

    Returns
    -------
    Zin : complex (Ohm)
    """
    tanh_gd = np.tanh(gamma * d)
    Zin = Zo * (ZL + Zo * tanh_gd) / (Zo + ZL * tanh_gd)
    return Zin


def compute_voltage_current(z_array, Gamma_L, gamma, Zo, V_plus=1.0):
    """
    Compute voltage and current wave distributions along the transmission line.

    The line is oriented with z=0 at the load. Positive z points toward source.

    V(z) = V+ * [e^(gamma*z) + Gamma_L * e^(-gamma*z)]
    I(z) = (V+/Zo) * [e^(gamma*z) - Gamma_L * e^(-gamma*z)]

    Parameters
    ----------
    z_array  : 1-D ndarray  positions along line (m), 0 = load end
    Gamma_L  : complex       reflection coefficient at load
    gamma    : complex       propagation constant
    Zo       : complex       characteristic impedance
    V_plus   : complex       amplitude of forward-traveling wave (default 1 V)

    Returns
    -------
    V : complex ndarray  voltage at each position
    I : complex ndarray  current at each position
    """
    z = np.asarray(z_array, dtype=complex)
    V = V_plus * (np.exp(gamma * z) + Gamma_L * np.exp(-gamma * z))
    I = (V_plus / Zo) * (np.exp(gamma * z) - Gamma_L * np.exp(-gamma * z))
    return V, I


def compute_Gamma_along_line(z_array, Gamma_L, gamma):
    """
    Compute the reflection coefficient at every point along the line.

    Gamma(z) = Gamma_L * e^(-2*gamma*z)

    (z=0 at load, z>0 toward source)

    Returns
    -------
    Gamma_z : complex ndarray
    """
    z = np.asarray(z_array, dtype=complex)
    return Gamma_L * np.exp(-2 * gamma * z)


def compute_Zin_along_line(z_array, Zo, ZL, gamma):
    """
    Compute input impedance at every point along the line.

    Returns
    -------
    Zin_z : complex ndarray
    """
    z = np.asarray(z_array, dtype=complex)
    tanh_gd = np.tanh(gamma * z)
    Zin_z = Zo * (ZL + Zo * tanh_gd) / (Zo + ZL * tanh_gd)
    return Zin_z


# ---------------------------------------------------------------------------
# Convenience "compute_all" function used by GUI and ML data generator
# ---------------------------------------------------------------------------

def compute_all(R, L, G, C, f, d, ZL_real, ZL_imag):
    """
    Compute all transmission line parameters for a given set of inputs.

    Parameters
    ----------
    R, L, G, C : float  distributed parameters (Ohm/m, H/m, S/m, F/m)
    f          : float  frequency (Hz)
    d          : float  line length / distance from load (m)
    ZL_real    : float  real part of load impedance (Ohm)
    ZL_imag    : float  imaginary part of load impedance (Ohm)

    Returns
    -------
    results : dict with keys:
        Zo, gamma, alpha, beta, Gamma_L, VSWR, Zin,
        |Zo|, angle_Zo_deg,
        |Gamma_L|, angle_Gamma_deg,
        |Zin|, angle_Zin_deg
    """
    ZL = complex(ZL_real, ZL_imag)

    gamma, alpha, beta = compute_gamma(R, L, G, C, f)
    Zo    = compute_Zo(R, L, G, C, f)
    Gamma_L = compute_reflection_coefficient(ZL, Zo)
    vswr    = compute_VSWR(Gamma_L)
    Zin     = compute_Zin(Zo, ZL, gamma, d)

    results = {
        # Complex values
        "Zo":       Zo,
        "gamma":    gamma,
        "Gamma_L":  Gamma_L,
        "Zin":      Zin,

        # Scalar real values
        "alpha":    alpha,
        "beta":     beta,
        "VSWR":     vswr,

        # Magnitude / angle (for display)
        "|Zo|":            np.abs(Zo),
        "angle_Zo_deg":    np.degrees(np.angle(Zo)),
        "|Gamma_L|":       np.abs(Gamma_L),
        "angle_Gamma_deg": np.degrees(np.angle(Gamma_L)),
        "|Zin|":           np.abs(Zin),
        "angle_Zin_deg":   np.degrees(np.angle(Zin)),
    }
    return results


# ---------------------------------------------------------------------------
# Frequency-sweep helpers (used by plots.py)
# ---------------------------------------------------------------------------

def sweep_frequency(R, L, G, C, f_start, f_end, n_points=200):
    """
    Sweep frequency from f_start to f_end and return alpha, beta, Zo arrays.

    Returns
    -------
    freqs  : ndarray of frequencies
    alphas : ndarray of attenuation constants
    betas  : ndarray of phase constants
    Zos    : ndarray of |Zo|
    """
    freqs  = np.linspace(f_start, f_end, n_points)
    alphas = np.zeros(n_points)
    betas  = np.zeros(n_points)
    Zos    = np.zeros(n_points)

    for i, fi in enumerate(freqs):
        g, a, b = compute_gamma(R, L, G, C, fi)
        alphas[i] = a
        betas[i]  = b
        Zos[i]    = np.abs(compute_Zo(R, L, G, C, fi))

    return freqs, alphas, betas, Zos


def sweep_frequency_vswr(R, L, G, C, ZL_real, ZL_imag,
                         f_start, f_end, n_points=200):
    """
    Sweep frequency and compute VSWR at each frequency.

    Returns
    -------
    freqs : ndarray
    vswrs : ndarray
    """
    ZL = complex(ZL_real, ZL_imag)
    freqs = np.linspace(f_start, f_end, n_points)
    vswrs = np.zeros(n_points)

    for i, fi in enumerate(freqs):
        Zo      = compute_Zo(R, L, G, C, fi)
        Gamma_L = compute_reflection_coefficient(ZL, Zo)
        vswrs[i] = compute_VSWR(Gamma_L)

    return freqs, vswrs
