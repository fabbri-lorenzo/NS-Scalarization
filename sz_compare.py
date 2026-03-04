# sz_compare_lambda.py
import time
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from Utils.params import M, rho0_lightS, rho0_heavyS
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.core import run_solve_model_captured
from Utils.graphics_single import custom_print
from Utils.graphics_compare import plotResults_compare_lambda


# ---------- USER DEFINED PARAMETERS ----------
p_eqState = p_SLy4
rho_eqState = rho_SLy4
frac_pc = 1e-10

xi = 500
m = 0.0
nu = 0.0
method = "BDF"

# compare these two lambdas:
lmbda_1 = 0.0
lmbda_2 = 1e-60  

a, b = 1e-10 * M, M
target_shooting = [0.0] if nu == 0.0 else [abs(nu), -abs(nu)]
abs_cut = a * 1e-2
rel_cut = 1e-2
merge_tol = a * 1e-2


def make_params(lmbda):
    return {
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

def format_lambda_latex(lmbda):
    if lmbda == 0:
        return r"$\lambda=0$"

    exp = int(np.floor(np.log10(abs(lmbda))))
    base = lmbda / (10**exp)

    # If base is ~1, show only 10^exp
    if np.isclose(base, 1.0):
        return rf"$\lambda=10^{{{exp}}}$"
    else:
        return rf"$\lambda={base:.1f}\times 10^{{{exp}}}$"
    
def main():
    lambdas = [lmbda_1, lmbda_2]
    jobs = [("light", rho0_lightS)] #, ("heavy", rho0_heavyS)]

    custom_print(
        f"\nCOMPARE RUN | ξ={xi} | µ={m:.2e} | ν={nu:.2e} | λ in {lambdas}",
        style="bold",
    )

    t0 = time.perf_counter()

    # Run 2 lambdas × 2 stars = 4 jobs
    run_specs = []
    for lmbda in lambdas:
        params = make_params(lmbda)
        for star_label, rho0 in jobs:
            run_specs.append((lmbda, star_label, params, rho0))

    results = {}  # (lmbda, star_label) -> (ok, result, logs)

    with ProcessPoolExecutor(max_workers=4) as ex:
        # map expects (params, rho0). We'll keep lmbda/star_label outside.
        futures = []
        for lmbda, star_label, params, rho0 in run_specs:
            futures.append((lmbda, star_label, ex.submit(run_solve_model_captured, params, rho0)))

        for lmbda, star_label, fut in futures:
            label, out, ok, result = fut.result()
            custom_print(f"\n===== {star_label} | λ={lmbda:g} log (ok={ok}) =====", style="bold")
            print(out, end="")
            results[(lmbda, star_label)] = (ok, result)

    custom_print(f"\nExecution time: {time.perf_counter() - t0:.0f} s", style="dim")

    # Plot per star type, overlaying the two lambdas
    for star_label, _rho0 in jobs:
        entries_by_lambda = {}
        savepath = None
        vacuum_sols = None
        nu_val = None

        for lmbda in lambdas:
            ok, result = results.get((lmbda, star_label), (False, None))
            if not ok or result is None:
                continue

            plot_entries = result.get("plot_entries", [])
            if plot_entries:
                entries_by_lambda[lmbda] = plot_entries

            # pick a base savepath; you can also build a dedicated compare folder
            if savepath is None:
                savepath = result.get("path", None)

            if vacuum_sols is None:
                vacuum_sols = result.get("vacuum_sols", None)

            if nu_val is None:
                nu_val = result.get("nu_val", nu)

        if not entries_by_lambda or savepath is None:
            custom_print(f"\nNo comparable solutions found for {star_label} star.", color="yellow")
            continue

        # Put compare plots into a subfolder to avoid overwriting single-lambda plots
        compare_path = savepath.rstrip("/\\") + f"/compare_lambda_{lambdas[0]:g}_vs_{lambdas[1]:g}/"

        plotResults_compare_lambda(
            entries_by_lambda=entries_by_lambda,
            xi=xi,
            m=m,
            nu=nu_val,
            vacuum_sols=vacuum_sols if vacuum_sols is not None else {"+": 0, "-": 0},
            savepath=compare_path,
            lambda_labels = {l: format_lambda_latex(l) for l in lambdas},
            lambda_linestyles={lmbda_1: "-", lmbda_2: "--"},
        )


if __name__ == "__main__":
    main()