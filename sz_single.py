"""
Entry point for solving scalarised neutron–star configurations.

This script defines the physical parameters of interest, then
constructs a parameter dictionary and dispatches the heavy lifting
to :mod:`Utils.core`.  Each star (light/heavy) is processed in a
separate process via :class:`concurrent.futures.ProcessPoolExecutor`.
The results are plotted afterwards via
``Utils.graphics_single.plotResults_multi`` using the data returned
from the solver.

Compared to the original ``sz_single.py``, the solver functions
``solve_model`` and ``run_solve_model_captured`` have been moved
into :mod:`Utils.core` for modularity.
"""

import time
import numpy as np
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import curve_fit

from Utils.params import (
    M,
    SM,
    c,
    G_N,
    lmbda_to_SI,
    m_ev_to_SI,
    nu_ev_to_SI,
    rho0_lightS,
    rho0_heavyS,
)
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.graphics_single import plotResults_multi, custom_print
from Utils.core import run_solve_model_captured

# ---------- USER DEFINED PARAMETERS ----------
# Equation of state and integration control
p_eqState = p_SLy4
rho_eqState = rho_SLy4
frac_pc = 1e-10  # Fraction of p_c to stop integration

plot_mode_count = (
    None  # Number of distinct mode numbers to plot; None plots all found modes
)

# Physical couplings
m = 1e-12  # scalar mass (eV)
lmbda = 0.0  # self–coupling (dimensionless)
nu = 0.0  # vacuum expectation value (eV)
xi = 100  # non-minimal coupling
# xi = 49000 * np.sqrt(lmbda)

# Choose integrator method based on parameter regime
method = "BDF"

tail = (250_000, 300_000)  # r region to fit into, meters
n_int_points = 1000  # number of points to use for integration
n_coarse = 31  # number of points to use for coarse integration
r_max = 3e5  # maximum radius for integration (km)

# Shooting parameters
a, b = 1e-10 * M, M  # bracket for σ₀


target_shooting = [0.0]
if (lmbda != 0.0) and (nu_ev_to_SI(nu)**2 > m_ev_to_SI(m)**2 / lmbda_to_SI(lmbda)):  
    sigma_min = np.sqrt(nu_ev_to_SI(nu)**2 - m_ev_to_SI(m)**2 / lmbda_to_SI(lmbda))
    target_shooting = [abs(sigma_min), -abs(sigma_min)]
abs_cut = a * 1e-2
rel_cut = 1e-2
merge_tol = a * 1e-2


def sigma_over_M_coulomb_tail(r_km, *, entry, **_):
    """Asymptotic sigma/M = Q/r for the mode represented by entry."""
    sol = entry.get("sol")

    def sigma_model(r, Q):
        return Q / r

    r = sol.t
    # Auto-estimate initial guess from a simple 1/r Coulomb-like fit
    mask = (r >= tail[0]) & (r <= tail[1])
    r_fit = r[mask]
    sigma_fit = sol.y[2][mask]
    sigmap_fit = sol.y[3][mask]

    Q0 = np.median(-sigmap_fit * r_fit**2)  # rough charge estimate

    # Wrap model to fix lambda
    def model_fixed_lam(r_fit, Q):
        return sigma_model(r_fit, Q)

    popt, pcov = curve_fit(
        model_fixed_lam,
        r_fit,
        sigma_fit,
        p0=Q0,
        bounds=(0.0, np.inf),
        maxfev=50_000,
    )

    scalar_charge = popt[0]
    r_m = np.asarray(r_km, dtype=float) * 1e3
    return scalar_charge / (M * r_m)


def sigma_over_M_coulomb_tail2(r_km, *, entry, **_):
    """Asymptotic sigma/M = Q/r for the mode represented by entry."""
    sol = entry.get("sol")
    r = sol.t

    # Isolate the same safe window for the simple median estimator
    mask = (r >= tail[0]) & (r <= tail[1])
    r_fit = r[mask]
    sigmap_fit = sol.y[3][mask]

    # Unbiased geometric charge estimation
    scalar_charge = np.median(-sigmap_fit * r_fit**2)
    r_m = np.asarray(r_km, dtype=float) * 1e3
    return scalar_charge / (M * r_m)


def sigma_over_M_quartic_tail(r_km, *, entry, lmbda, **_):
    """Log-improved quartic tail in the same units as sigma.png."""
    geom_scalar_charge = entry.get("scalar_charge")
    scalar_charge = geom_scalar_charge * M * G_N / (c * c)
    r_bar = entry.get("r_bar")
    r_m = np.asarray(r_km, dtype=float) * 1e3

    lmbda_si = lmbda_to_SI(lmbda)
    radicand = 1.0 + 2.0 * lmbda_si * scalar_charge**2 * np.log(r_m / r_bar)

    with np.errstate(divide="ignore", invalid="ignore"):
        y = scalar_charge / (r_m * M * np.sqrt(radicand))
    return np.where(radicand > 0.0, y, np.nan)


def sigma_over_M_mass_tail(r_km, *, entry, m, **_):
    """Log-improved quartic tail in the same units as sigma.png."""
    geom_scalar_charge = entry.get("scalar_charge")
    scalar_charge = geom_scalar_charge * M * G_N / (c * c)
    r_m = np.asarray(r_km, dtype=float) * 1e3
    m_SI = m_ev_to_SI(m)

    return np.exp(-m_SI * r_m) * (scalar_charge / (M * r_m))


# Optional functions to overlay on the generated plots.
#
# Each function receives r in km and must return y-values in the units of the
# selected plot:
#   "pressure" -> Pa, "mu2" -> km^-2, "sigma" -> sigma/M_Pl.
#
# Use per_entry=True when the function needs per-mode quantities such as
# scalar_charge, r_bar, ADM_mass, or the numerical solution itself.
extra_plot_functions = [
    # {
    #    "plot": "sigma",
    #    "function": sigma_over_M_coulomb_tail,
    #    "per_entry": True,
    #    "label": r"Q/r fit",
    #    "color": "tab:red",
    # },
    {
        "plot": "sigma",
        "function": sigma_over_M_coulomb_tail2,
        "per_entry": True,
        "label": r"Q/r",
        "color": "tab:blue",
    },
    # {
    #   "plot": "sigma",
    #   "function": sigma_over_M_quartic_tail,
    #   "per_entry": True,
    #   "label": r"\frac{Q}{r \, \sqrt{1+2\lambda Q^2\,\ln(r/\bar{r})}}",
    #   "color": "tab:red",
    # },
    {
        "plot": "sigma",
        "function": sigma_over_M_mass_tail,
        "per_entry": True,
        "label": r"\frac{Q}{r} e^{-\mu r}",
        "color": "tab:red",
    },
]
# extra_plot_functions = None
 

def main() -> None:
    """Run the solver for light and heavy stars and plot results."""
    # Print run header
    custom_print(
        f"\nξ = {xi:.0g} | µ = {m:.2e} eV | λ = {lmbda:.2e} | ν = {nu:.2e} eV",
        style="bold",
    )
    # Construct a parameter dictionary to be passed to the core solver
    params = {
        "p_eqState": p_eqState,
        "rho_eqState": rho_eqState,
        "frac_pc": frac_pc,
        "xi": xi,
        "m": m,
        "lmbda": lmbda,
        "nu": nu,
        "method": method,
        "a": a,
        "b": b,
        "target_shooting": target_shooting,
        "abs_cut": abs_cut,
        "rel_cut": rel_cut,
        "merge_tol": merge_tol,
        "max_plot_modes": plot_mode_count,
        "tail": tail,
        "n_int_points": n_int_points,
        "n_coarse": n_coarse,
        "r_max": r_max,
    }
    t0 = time.perf_counter()
    jobs = [rho0_lightS]  # List of central densities for light and heavy stars
    # Launch separate processes for light and heavy stars
    results = []

    with ProcessPoolExecutor(max_workers=2) as ex:
        for label, out, ok, result in ex.map(
            run_solve_model_captured, repeat(params), jobs
        ):
            # Print logs in order
            custom_print(f"\n===== {label} star log (ok={ok}) =====", style="bold")
            print(out, end="")
            results.append((label, ok, result))
    t_tot = time.perf_counter() - t0
    custom_print(f"\nExecution time: {t_tot:.0f} s", style="dim")
    # After solving, invoke plotting for each star
    for label, ok, result in results:
        if not ok or result is None:
            continue

        plot_entries = result["plot_entries"]
        path = result["path"]

        # Skip plotting if no scalarized solutions were found
        if not plot_entries or not path:
            custom_print(
                f"No scalarized solutions for {label} star.",
                color="yellow",
            )
            continue

        vacuum_sols = result["vacuum_sols"]
        nu_val = result["nu_val"]

        plotResults_multi(
            plot_entries,
            xi,
            lmbda,
            m,
            nu_val,
            vacuum_sols,
            path,
            zoom_regions = [
        {
            "x_min": 200,
            "x_max": 201,
            "y_min": 4.33e-5,
            "y_max": 4.4e-5,
            "n_points": 2,
        },
        {
            "x_min": 250,
            "x_max": 251,
            "y_min": -1.028e-4,
            "y_max": -1.018e-4,
            "n_points": 2,
        }
    ],
            star_label=label,
            extra_functions=extra_plot_functions,
        )


if __name__ == "__main__":
    main()
