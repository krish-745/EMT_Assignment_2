"""
main.py
=======
Transmission Line Analyzer -- Entry Point
=========================================
Engineering Electromagnetics | Module-1 Assignment

Usage:
    python main.py [--cli] [--train] [--plots]

Flags:
    (no flags)    Launch the full GUI application
    --cli         Run a command-line demonstration (no GUI)
    --train       Force retrain the ML model and exit
    --plots       Generate all plots for default parameters and exit

Author  : Student | Engineering Electromagnetics
Course  : Transmission Lines -- Module 1
"""

import sys
import os
import argparse
import numpy as np

# Ensure the project directory is in PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# CLI demonstration (no GUI, no Tkinter dependency)
# ---------------------------------------------------------------------------
def run_cli_demo():
    """
    Run a full command-line demonstration:
      1. Compute all analytical parameters for a sample Tx line
      2. Train / load the ML model
      3. Compare analytical vs. ML-predicted values
      4. Print a summary table
    """
    from analytics import compute_all
    from ml_model import get_or_train_model, predict

    print("\n" + "=" * 60)
    print("  TRANSMISSION LINE ANALYZER -- CLI Demo")
    print("  Engineering Electromagnetics | Module-1")
    print("=" * 60)

    # -- Example parameters -----------------------------------------
    params = dict(
        R        = 5.0,         # Ohm/m
        L        = 250e-9,      # H/m
        G        = 1e-4,        # S/m
        C        = 100e-12,     # F/m
        f        = 100e6,       # Hz  (100 MHz)
        d        = 2.0,         # m
        ZL_real  = 75.0,        # Ohm  -- must match compute_all() signature
        ZL_imag  = 30.0,        # Ohm
    )

    print("\n[1] INPUT PARAMETERS")
    print("-" * 40)
    for k, v in params.items():
        print(f"  {k:<8} = {v:.4g}")

    # -- Analytical computation ------------------------------------
    print("\n[2] ANALYTICAL RESULTS")
    print("-" * 40)
    res = compute_all(**params)

    a, b = res["alpha"], res["beta"]
    sign = "+" if b >= 0 else "-"
    print(f"  {'Zo':<20} = {res['|Zo|']:.4f} Ohm  /{res['angle_Zo_deg']:.2f} deg")
    print(f"  {'gamma':<20} = {a:.4e} {sign} j{abs(b):.4e}  1/m")
    print(f"  {'alpha':<20} = {res['alpha']:.6e}  Np/m")
    print(f"  {'beta':<20} = {res['beta']:.6e}  rad/m")
    print(f"  {'|Gamma_L|':<20} = {res['|Gamma_L|']:.6f}  "
          f"/{res['angle_Gamma_deg']:.2f} deg")
    print(f"  {'VSWR':<20} = {res['VSWR']:.4f}")
    print(f"  {'Zin':<20} = {res['|Zin|']:.4f} Ohm  /{res['angle_Zin_deg']:.2f} deg")

    # -- ML model -------------------------------------------------------
    print("\n[3] ML MODEL -- Random Forest Regressor")
    print("-" * 40)
    bundle = get_or_train_model(n_samples=500, force_retrain=False)
    print(f"  Overall R^2 (test set) = {bundle['r2_overall']:.4f}")
    print(f"  Status: {'PASS (>=0.97)' if bundle['r2_overall'] >= 0.97 else 'FAIL (<0.97)'}")

    print("\n  Per-target R^2 scores:")
    for name, r2, mape in zip(bundle["target_names"],
                               bundle["r2_scores"],
                               bundle["mape_scores"]):
        print(f"    {name:<14}  R^2 = {r2:.4f}   MAPE = {mape:.2f}%")

    # -- Compare analytical vs ML ----------------------------------
    print("\n[4] ANALYTICAL  vs.  ML PREDICTION")
    print("-" * 60)
    ZL = complex(params["ZL_real"], params["ZL_imag"])
    pred = predict(bundle,
                   R=params["R"], L=params["L"],
                   G=params["G"], C=params["C"],
                   f=params["f"], d=params["d"],
                   ZL_mag=abs(ZL),
                   ZL_angle_deg=np.degrees(np.angle(ZL)))

    comparisons = [
        ("|Zo| (Ohm)",    res["|Zo|"],      pred["|Zo|_pred"]),
        ("alpha (Np/m)",  res["alpha"],     pred["alpha_pred"]),
        ("beta (rad/m)",  res["beta"],      pred["beta_pred"]),
        ("|Gamma_L|",     res["|Gamma_L|"], pred["|Gamma_L|_pred"]),
        ("VSWR",          res["VSWR"],      pred["VSWR_pred"]),
    ]

    print(f"  {'Parameter':<18} {'Analytical':>14} {'ML Predicted':>14} {'Error%':>9}")
    print("  " + "-" * 58)
    for name, exact, ml_val in comparisons:
        err = abs(exact - ml_val) / (abs(exact) + 1e-12) * 100
        print(f"  {name:<18} {exact:>14.6g} {ml_val:>14.6g} {err:>8.2f}%")

    # -- Generate plots --------------------------------------------
    print("\n[5] GENERATING ALL 7 PLOTS (saved to plots_output/)")
    print("-" * 40)
    try:
        from plots import generate_all_plots
        import matplotlib.pyplot as plt
        figs = generate_all_plots(
            params["R"], params["L"], params["G"], params["C"],
            params["f"], params["d"], params["ZL_real"], params["ZL_imag"],
            f_start=1e6, f_end=1e9, save=True
        )
        for title, fig in figs:
            plt.close(fig)
        from plots import PLOT_DIR
        print(f"  [OK] All plots saved in: {PLOT_DIR}")

        # ML performance plot
        from ml_model import plot_ml_performance
        fig_ml = plot_ml_performance(bundle, save=True)
        plt.close(fig_ml)
    except Exception as e:
        print(f"  Plot generation error: {e}")

    print("\n" + "=" * 60)
    print("  CLI Demo Complete.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# GUI launcher
# ---------------------------------------------------------------------------
def run_gui():
    """Launch the Tkinter GUI application."""
    try:
        from gui import TxLineApp
        app = TxLineApp()
        app.mainloop()
    except Exception as e:
        print(f"[ERROR] GUI failed to start: {e}")
        print("Try running with --cli flag for command-line mode.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Transmission Line Analyzer -- Engineering Electromagnetics"
    )
    parser.add_argument("--cli",    action="store_true",
                        help="Run CLI demonstration (no GUI)")
    parser.add_argument("--train",  action="store_true",
                        help="Force retrain ML model and exit")
    parser.add_argument("--plots",  action="store_true",
                        help="Generate plots for default parameters and exit")

    args = parser.parse_args()

    if args.train:
        from ml_model import get_or_train_model, plot_ml_performance
        import matplotlib.pyplot as plt
        bundle = get_or_train_model(n_samples=500, force_retrain=True)
        fig = plot_ml_performance(bundle, save=True)
        plt.close(fig)
        print("Model training complete. Exiting.")
        return

    if args.plots:
        from plots import generate_all_plots
        import matplotlib.pyplot as plt
        figs = generate_all_plots(
            5.0, 250e-9, 1e-4, 100e-12,
            100e6, 2.0, 75.0, 30.0,
            f_start=1e6, f_end=1e9, save=True
        )
        for _, fig in figs:
            plt.close(fig)
        from plots import PLOT_DIR
        print(f"Plots saved in: {PLOT_DIR}")
        return

    if args.cli:
        run_cli_demo()
        return

    # Default: launch GUI
    run_gui()


if __name__ == "__main__":
    main()
