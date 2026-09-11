"""
gui.py
======
Tkinter GUI -- Transmission Line Analysis & Design Window
=========================================================
A complete interactive interface providing:
  - Input fields for all Tx line parameters
  - "Compute Analytics" button -> shows all derived parameters
  - "Plot Waveforms" button -> renders all 7 plots in a tabbed viewer
  - "ML Predict" button -> shows ML-predicted parameters vs. exact values
  - "Train/Reload Model" button -> (re-)trains the Random Forest model
  - Status bar with progress messages

Layout:
  +-----------------------------------------------------------------+
  |  TITLE BAR                                                      |
  +----------------------+------------------------------------------+
  |  INPUT PANEL         |  RESULTS PANEL                           |
  |  (R, L, G, C, f,     |  (Zo, gamma, alpha, beta, Gamma, VSWR, Zin)           |
  |   d, ZL_r, ZL_i)     |  + ML Predictions                       |
  +----------------------+------------------------------------------+
  |  BUTTON BAR                                                      |
  +-----------------------------------------------------------------+
  |  PLOT CANVAS (tabbed -- 7 tabs, one per plot)                    |
  +-----------------------------------------------------------------+
  |  STATUS BAR                                                      |
  +-----------------------------------------------------------------+
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import os

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from analytics import compute_all
from plots import generate_all_plots
import ml_model as mlm

# ---------------------------------------------------------------------------
# Colour / font constants
# ---------------------------------------------------------------------------
BG_MAIN    = "#1e1e2e"   # dark background
BG_PANEL   = "#2a2a3e"   # slightly lighter panel
BG_INPUT   = "#313147"
FG_TEXT    = "#cdd6f4"
FG_LABEL   = "#89b4fa"
FG_VALUE   = "#a6e3a1"
FG_ML      = "#f9e2af"
FG_TITLE   = "#cba6f7"
BTN_BG     = "#45475a"
BTN_FG     = "#cdd6f4"
BTN_ACTIVE = "#585b70"
ACCENT     = "#89dceb"

FONT_TITLE  = ("Segoe UI", 14, "bold")
FONT_HEADER = ("Segoe UI", 10, "bold")
FONT_LABEL  = ("Segoe UI", 9)
FONT_VALUE  = ("Consolas", 9, "bold")
FONT_BTN    = ("Segoe UI", 9, "bold")
FONT_STATUS = ("Segoe UI", 8)

# ---------------------------------------------------------------------------
# Default parameter values (coaxial-cable-like example)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "R":      "5.0",          # Ohm/m
    "L":      "250e-9",       # H/m
    "G":      "1e-4",         # S/m
    "C":      "100e-12",      # F/m
    "f":      "100e6",        # Hz   (100 MHz)
    "d":      "2.0",          # m
    "ZL_r":   "75.0",         # Ohm  (real part)
    "ZL_i":   "30.0",         # Ohm  (imaginary part)
    "f_start":"1e6",          # Hz  (freq sweep start)
    "f_end":  "1e9",          # Hz  (freq sweep end)
}

# ---------------------------------------------------------------------------
# Helper: formatted row in results panel
# ---------------------------------------------------------------------------
def _fmt(val, unit="", is_complex=False):
    if is_complex:
        r, i = np.real(val), np.imag(val)
        sign = "+" if i >= 0 else "-"
        return f"{r:.4f} {sign} j{abs(i):.4f}  {unit}"
    if isinstance(val, (float, np.floating)):
        return f"{val:.6g}  {unit}"
    return f"{val}  {unit}"


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------
class TxLineApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transmission Line Analyzer -- Engineering Electromagnetics")
        self.configure(bg=BG_MAIN)
        self.resizable(True, True)
        self.geometry("1280x820")
        self.minsize(1100, 700)

        # Model bundle (loaded lazily)
        self._bundle = None

        self._build_ui()
        self.after(200, self._load_model_async)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        """Build all UI sections."""
        self._build_title()
        self._build_main_frame()
        self._build_input_panel()
        self._build_results_panel()
        self._build_button_bar()
        self._build_plot_area()
        self._build_status_bar()

    def _build_title(self):
        frm = tk.Frame(self, bg=BG_MAIN)
        frm.pack(fill="x", padx=16, pady=(12, 0))
        tk.Label(frm,
                 text="*  Transmission Line Analysis & Design Window",
                 font=FONT_TITLE, bg=BG_MAIN, fg=FG_TITLE).pack(side="left")
        tk.Label(frm,
                 text="Engineering Electromagnetics | Module-1",
                 font=FONT_LABEL, bg=BG_MAIN, fg=FG_TEXT).pack(side="right")

    def _build_main_frame(self):
        self._main_frm = tk.Frame(self, bg=BG_MAIN)
        self._main_frm.pack(fill="x", padx=12, pady=6)

    def _build_input_panel(self):
        """Left panel: parameter input fields."""
        panel = tk.LabelFrame(self._main_frm, text="  [load]  Input Parameters  ",
                              bg=BG_PANEL, fg=FG_LABEL,
                              font=FONT_HEADER, bd=1, relief="groove",
                              padx=10, pady=8)
        panel.pack(side="left", fill="y", padx=(0, 6))

        self._entries = {}

        fields = [
            ("R  (Ohm/m)",      "R",      "Resistance per unit length"),
            ("L  (H/m)",      "L",      "Inductance per unit length"),
            ("G  (S/m)",      "G",      "Conductance per unit length"),
            ("C  (F/m)",      "C",      "Capacitance per unit length"),
            ("f  (Hz)",       "f",      "Frequency"),
            ("d  (m)",        "d",      "Line length / distance from load"),
            ("ZL_real  (Ohm)",  "ZL_r",   "Real part of load impedance"),
            ("ZL_imag  (Ohm)",  "ZL_i",   "Imaginary part of load impedance"),
            ("f_start  (Hz)", "f_start","Frequency sweep start"),
            ("f_end  (Hz)",   "f_end",  "Frequency sweep end"),
        ]

        for row_i, (label, key, tip) in enumerate(fields):
            tk.Label(panel, text=label, bg=BG_PANEL, fg=FG_LABEL,
                     font=FONT_LABEL, anchor="w", width=18).grid(
                         row=row_i, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=DEFAULTS.get(key, ""))
            ent = tk.Entry(panel, textvariable=var, width=14,
                           bg=BG_INPUT, fg=FG_TEXT,
                           insertbackground=FG_TEXT,
                           font=FONT_VALUE, bd=0, relief="flat")
            ent.grid(row=row_i, column=1, padx=(6, 0), pady=3)
            self._entries[key] = var

    def _build_results_panel(self):
        """Right panel: computed results display."""
        panel = tk.LabelFrame(self._main_frm, text="  [chart]  Computed Parameters  ",
                              bg=BG_PANEL, fg=FG_LABEL,
                              font=FONT_HEADER, bd=1, relief="groove",
                              padx=10, pady=8)
        panel.pack(side="left", fill="both", expand=True)

        # Analytical results
        tk.Label(panel, text="-- Analytical Results --",
                 bg=BG_PANEL, fg=ACCENT,
                 font=FONT_HEADER).grid(row=0, column=0, columnspan=2,
                                        sticky="w", pady=(0, 4))

        self._result_vars = {}
        result_fields = [
            ("Zo",               "|Zo|",         "Ohm"),
            ("/_Zo",             "angle_Zo_deg", " deg"),
            ("gamma  (alpha + jbeta)",  "gamma_str",    ""),
            ("alpha  (attenuation)", "alpha",        "Np/m"),
            ("beta  (phase)",       "beta",         "rad/m"),
            ("Gamma at load",        "|Gamma_L|",    ""),
            ("/_Gamma",              "angle_Gamma_deg"," deg"),
            ("VSWR",             "VSWR",         ""),
            ("Zin",             "|Zin|",         "Ohm"),
            ("/_Zin",           "angle_Zin_deg", " deg"),
        ]

        for row_i, (label, key, unit) in enumerate(result_fields):
            tk.Label(panel, text=label, bg=BG_PANEL, fg=FG_LABEL,
                     font=FONT_LABEL, anchor="w", width=22).grid(
                         row=row_i+1, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="--")
            tk.Label(panel, textvariable=var, bg=BG_PANEL, fg=FG_VALUE,
                     font=FONT_VALUE, anchor="w", width=30).grid(
                         row=row_i+1, column=1, sticky="w", padx=6)
            self._result_vars[key] = var

        # Divider
        ttk.Separator(panel, orient="horizontal").grid(
            row=len(result_fields)+2, column=0, columnspan=2,
            sticky="ew", pady=8)

        # ML results
        tk.Label(panel, text="-- ML Predictions (Random Forest) --",
                 bg=BG_PANEL, fg=FG_ML,
                 font=FONT_HEADER).grid(
                     row=len(result_fields)+3, column=0, columnspan=2,
                     sticky="w", pady=(0, 4))

        self._ml_vars = {}
        ml_fields = [
            ("|Zo| predicted",      "|Zo|_pred",      "Ohm"),
            ("alpha  predicted",        "alpha_pred",     "Np/m"),
            ("beta  predicted",        "beta_pred",      "rad/m"),
            ("|Gamma| predicted",       "|Gamma_L|_pred", ""),
            ("VSWR predicted",      "VSWR_pred",      ""),
            ("Model R^2 (overall)",  "r2_overall",     ""),
        ]

        base = len(result_fields) + 4
        for row_i, (label, key, unit) in enumerate(ml_fields):
            tk.Label(panel, text=label, bg=BG_PANEL, fg=FG_ML,
                     font=FONT_LABEL, anchor="w", width=22).grid(
                         row=base + row_i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="--")
            tk.Label(panel, textvariable=var, bg=BG_PANEL, fg=FG_ML,
                     font=FONT_VALUE, anchor="w", width=30).grid(
                         row=base + row_i, column=1, sticky="w", padx=6)
            self._ml_vars[key] = var

    def _build_button_bar(self):
        """Action buttons."""
        bar = tk.Frame(self, bg=BG_MAIN)
        bar.pack(fill="x", padx=12, pady=4)

        btn_cfg = dict(bg=BTN_BG, fg=BTN_FG, font=FONT_BTN,
                       activebackground=BTN_ACTIVE, activeforeground=FG_TEXT,
                       bd=0, padx=16, pady=6, cursor="hand2", relief="flat")

        tk.Button(bar, text="[settings]  Compute Analytics",
                  command=self._compute, **btn_cfg).pack(side="left", padx=4)
        tk.Button(bar, text="[plot]  Plot All Waveforms",
                  command=self._plot_all, **btn_cfg).pack(side="left", padx=4)
        tk.Button(bar, text="[AI]  ML Predict",
                  command=self._ml_predict, **btn_cfg).pack(side="left", padx=4)
        tk.Button(bar, text="[reload]  Train / Reload Model",
                  command=self._train_model, **btn_cfg).pack(side="left", padx=4)
        tk.Button(bar, text="[repeat]  Reset Defaults",
                  command=self._reset_defaults, **btn_cfg).pack(side="left", padx=4)

    def _build_plot_area(self):
        """Tabbed plot canvas area."""
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        # Style the notebook
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",
                        background=BG_MAIN, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=BTN_BG, foreground=FG_TEXT,
                        font=FONT_LABEL, padding=(8, 4))
        style.map("TNotebook.Tab",
                  background=[("selected", BG_PANEL)],
                  foreground=[("selected", ACCENT)])

        self._tab_frames = {}
        self._tab_canvases = {}

        tab_labels = [
            "Voltage |V|",
            "Current |I|",
            "Reflection |Gamma|",
            "Zin vs z",
            "VSWR vs f",
            "alpha & beta vs f",
            "Standing Wave",
            "ML Performance",
        ]
        for label in tab_labels:
            frm = tk.Frame(self._nb, bg=BG_PANEL)
            self._nb.add(frm, text=f"  {label}  ")
            self._tab_frames[label] = frm

        # Placeholder labels in each tab
        for label, frm in self._tab_frames.items():
            tk.Label(frm, text=f"[ {label} -- click 'Plot All Waveforms' ]",
                     bg=BG_PANEL, fg=FG_TEXT, font=FONT_LABEL).pack(
                         expand=True)

    def _build_status_bar(self):
        self._status_var = tk.StringVar(value="Ready. Enter parameters and click Compute.")
        bar = tk.Frame(self, bg="#11111b", bd=1, relief="sunken")
        bar.pack(fill="x", side="bottom")
        tk.Label(bar, textvariable=self._status_var,
                 bg="#11111b", fg="#6c7086",
                 font=FONT_STATUS, anchor="w", padx=8).pack(fill="x")

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    def _status(self, msg):
        self._status_var.set(msg)
        self.update_idletasks()

    def _get_params(self):
        """Parse all entry fields and return a dict of floats."""
        try:
            p = {k: float(v.get()) for k, v in self._entries.items()}
            return p
        except ValueError as e:
            messagebox.showerror("Input Error",
                                 f"Invalid parameter value:\n{e}")
            return None

    def _reset_defaults(self):
        for k, var in self._entries.items():
            var.set(DEFAULTS.get(k, ""))
        self._status("Parameters reset to defaults.")

    # -----------------------------------------------------------------------
    # Compute Analytics
    # -----------------------------------------------------------------------
    def _compute(self):
        p = self._get_params()
        if p is None:
            return
        self._status("Computing analytical parameters...")
        try:
            res = compute_all(p["R"], p["L"], p["G"], p["C"],
                              p["f"], p["d"], p["ZL_r"], p["ZL_i"])

            # Format gamma as string
            a, b = res["alpha"], res["beta"]
            sign = "+" if b >= 0 else "-"
            gamma_str = f"{a:.4e} {sign} j{abs(b):.4e}"

            self._result_vars["|Zo|"].set(_fmt(res["|Zo|"], "Ohm"))
            self._result_vars["angle_Zo_deg"].set(_fmt(res["angle_Zo_deg"], " deg"))
            self._result_vars["gamma_str"].set(gamma_str)
            self._result_vars["alpha"].set(_fmt(res["alpha"], "Np/m"))
            self._result_vars["beta"].set(_fmt(res["beta"], "rad/m"))
            self._result_vars["|Gamma_L|"].set(_fmt(res["|Gamma_L|"], ""))
            self._result_vars["angle_Gamma_deg"].set(
                _fmt(res["angle_Gamma_deg"], " deg"))
            self._result_vars["VSWR"].set(_fmt(res["VSWR"], ""))
            self._result_vars["|Zin|"].set(_fmt(res["|Zin|"], "Ohm"))
            self._result_vars["angle_Zin_deg"].set(
                _fmt(res["angle_Zin_deg"], " deg"))

            self._status("[OK] Analytics computed successfully.")
        except Exception as e:
            messagebox.showerror("Computation Error", str(e))
            self._status("Error during computation.")

    # -----------------------------------------------------------------------
    # Plot All Waveforms
    # -----------------------------------------------------------------------
    def _plot_all(self):
        p = self._get_params()
        if p is None:
            return
        self._status("Generating all 7 waveform plots -- please wait...")
        self._compute()   # refresh analytics first

        def _worker():
            try:
                figs = generate_all_plots(
                    p["R"], p["L"], p["G"], p["C"],
                    p["f"], p["d"], p["ZL_r"], p["ZL_i"],
                    f_start=p["f_start"], f_end=p["f_end"],
                    save=True
                )
                self.after(0, lambda: self._render_plots(figs))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "Plot Error", str(e)))
                self.after(0, lambda: self._status("Plot generation failed."))

        threading.Thread(target=_worker, daemon=True).start()

    def _render_plots(self, figs):
        """Embed matplotlib figures into Notebook tabs."""
        tab_labels = [
            "Voltage |V|",
            "Current |I|",
            "Reflection |Gamma|",
            "Zin vs z",
            "VSWR vs f",
            "alpha & beta vs f",
            "Standing Wave",
        ]
        for (title, fig), tab_label in zip(figs, tab_labels):
            frm = self._tab_frames[tab_label]
            # Clear old widgets
            for w in frm.winfo_children():
                w.destroy()

            canvas = FigureCanvasTkAgg(fig, master=frm)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._tab_canvases[tab_label] = canvas

        self._status("[OK] All 7 plots generated and saved to plots_output/")

        # Also try ML performance plot if model is ready
        if self._bundle is not None:
            try:
                fig_ml = mlm.plot_ml_performance(self._bundle, save=True)
                frm = self._tab_frames["ML Performance"]
                for w in frm.winfo_children():
                    w.destroy()
                canvas = FigureCanvasTkAgg(fig_ml, master=frm)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                plt.close(fig_ml)
            except Exception:
                pass

        for fig in [f for _, f in figs]:
            plt.close(fig)

    # -----------------------------------------------------------------------
    # ML Predict
    # -----------------------------------------------------------------------
    def _ml_predict(self):
        if self._bundle is None:
            messagebox.showinfo("Model Not Ready",
                                "ML model is still loading. Please wait a moment.")
            return
        p = self._get_params()
        if p is None:
            return
        self._status("Running ML prediction...")
        try:
            ZL = complex(p["ZL_r"], p["ZL_i"])
            ZL_mag   = abs(ZL)
            ZL_angle = np.degrees(np.angle(ZL))

            pred = mlm.predict(
                self._bundle,
                R=p["R"], L=p["L"], G=p["G"], C=p["C"],
                f=p["f"], d=p["d"],
                ZL_mag=ZL_mag, ZL_angle_deg=ZL_angle
            )

            self._ml_vars["|Zo|_pred"].set(
                f"{pred['|Zo|_pred']:.4f}  Ohm")
            self._ml_vars["alpha_pred"].set(
                f"{pred['alpha_pred']:.4e}  Np/m")
            self._ml_vars["beta_pred"].set(
                f"{pred['beta_pred']:.4e}  rad/m")
            self._ml_vars["|Gamma_L|_pred"].set(
                f"{pred['|Gamma_L|_pred']:.4f}")
            self._ml_vars["VSWR_pred"].set(
                f"{pred['VSWR_pred']:.4f}")
            self._ml_vars["r2_overall"].set(
                f"{self._bundle['r2_overall']:.4f}")

            self._status("[OK] ML prediction complete.")
        except Exception as e:
            messagebox.showerror("ML Error", str(e))
            self._status("ML prediction failed.")

    # -----------------------------------------------------------------------
    # Train / Reload Model
    # -----------------------------------------------------------------------
    def _train_model(self):
        self._status("Training ML model (500 samples) -- please wait...")

        def _worker():
            try:
                bundle = mlm.get_or_train_model(n_samples=1000,
                                                force_retrain=True)
                self._bundle = bundle
                r2 = bundle['r2_overall']
                status = "PASS" if r2 >= 0.97 else "INFO"
                self.after(0, lambda: self._status(
                    f"[{status}] Model trained. R^2 = {r2:.4f}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "Training Error", str(e)))
                self.after(0, lambda: self._status("Model training failed."))

        threading.Thread(target=_worker, daemon=True).start()

    def _load_model_async(self):
        """Silently load (or train) model in background on startup."""
        def _worker():
            try:
                self._bundle = mlm.get_or_train_model(n_samples=1000,
                                                       force_retrain=False)
                r2 = self._bundle['r2_overall']
                self.after(0, lambda: self._status(
                    f"ML model ready. R^2 = {r2:.4f}  -- Enter parameters and click Compute."))
            except Exception as e:
                self.after(0, lambda: self._status(
                    f"ML model load failed: {e}. Use 'Train/Reload Model' button."))

        threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    app = TxLineApp()
    app.mainloop()
