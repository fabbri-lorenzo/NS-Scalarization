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

from Utils.params import (
    M,
    v_higgs,
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

# Physical couplings
m = 1e-5  # scalar mass (eV)
lmbda = 0.0  # self–coupling (dimensionless)
nu = 0.0  # vacuum expectation value (eV)
xi = 10  # non-minimal coupling
# xi = 49000 * np.sqrt(lmbda)  # non-minimal coupling

# Choose integrator method based on parameter regime
method = "BDF"

# Shooting parameters
a, b = 1e-10 * M, M  # bracket for σ₀
target_shooting = [0.0] if nu == 0.0 else [abs(nu), -abs(nu)]
abs_cut = a * 1e-2
rel_cut = 1e-2
merge_tol = a * 1e-2

def main() -> None:
    # Print run header
    custom_print(
        f"\nξ = {xi:.0g} | µ = {m:.2e} | λ = {lmbda:.2e} | ν = {nu:.2e} | ν/M = {nu/M:.2e}",
        style="bold",
    )
    """Run the solver for light and heavy stars and plot results."""
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
    }
    t0 = time.perf_counter()
    jobs = [rho0_lightS, rho0_heavyS]
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

        plotResults_multi(plot_entries, xi, m, nu_val, vacuum_sols, path)


if __name__ == "__main__":
    main()
