"""
Parameter sweep driver comparing scalarised solutions across λ or μ.

Run this script to solve the stellar structure for a set of coupling
values and generate overlaid plots that highlight how the pressure,
effective mass and scalar field profiles change between them.
"""
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from Utils.params import M, rho0_lightS, rho0_heavyS
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.core import run_solve_model_captured
from Utils.graphics_single import custom_print
from Utils.graphics_compare import plotResults_compare


# ---------- USER DEFINED PARAMETERS ----------
p_eqState = p_SLy4
rho_eqState = rho_SLy4
frac_pc = 1e-10

xi = 500
m_fixed = 0.0  # used when comparing lambdas
lmbda_fixed = 0.0  # used when comparing mus
nu = 0.0
method = "BDF"

# choose what to sweep: "lambda" or "mu"
compare_param = "mu"

# values to compare for each mode
lmbda_1 = 0.0
lmbda_2 = 1e-60
lambda_values = [lmbda_1, lmbda_2]

m_1 = 0.0
m_2 = 1e-20
m_3 = 1e-12
mu_values = [m_1, m_2, m_3]

a, b = 1e-10 * M, M
target_shooting = 0.0
abs_cut = a * 1e-2
rel_cut = 1e-2
merge_tol = a * 1e-2


def make_params(lmbda, m_val):
    """Assemble the solver parameter dictionary for one (λ, μ) pair."""
    return {
        "p_eqState": p_eqState,
        "rho_eqState": rho_eqState,
        "frac_pc": frac_pc,
        "xi": xi,
        "m": m_val,
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
    """Nicely format λ values for plot legends (handles λ=0 separately)."""
    if lmbda == 0:
        return r"$\lambda=0$"

    exp = int(np.floor(np.log10(abs(lmbda))))
    base = lmbda / (10**exp)

    if np.isclose(base, 1.0):
        return rf"$\lambda=10^{{{exp}}}$"
    else:
        return rf"$\lambda={base:.1f}\times 10^{{{exp}}}$"


def format_mu_latex(mu):
    """Nicely format μ values for plot legends (handles μ=0 separately)."""
    if mu == 0:
        return r"$\mu=0$"

    exp = int(np.floor(np.log10(abs(mu))))
    base = mu / (10**exp)

    if np.isclose(base, 1.0):
        return rf"$\mu=10^{{{exp}}}$"
    else:
        return rf"$\mu={base:.1f}\times 10^{{{exp}}}$"


def main():
    """Run the comparison sweep, collect logs, and create overlay plots."""
    if compare_param == "lambda":
        sweep_values = lambda_values
        fixed_mu = m_fixed
        fixed_lambda = None
    elif compare_param == "mu":
        sweep_values = mu_values
        fixed_mu = None
        fixed_lambda = lmbda_fixed
    else:
        raise ValueError("compare_param must be 'lambda' or 'mu'")

    jobs = [("Light", rho0_lightS)]  # add ("Heavy", rho0_heavyS) if needed

    if compare_param == "lambda":
        custom_print(
            f"\nCOMPARE RUN | ξ={xi} | µ={fixed_mu:.2e} | ν={nu:.2e} | λ in {sweep_values}",
            style="bold",
        )
    else:
        custom_print(
            f"\nCOMPARE RUN | ξ={xi} | λ={lmbda_fixed:.2e} | ν={nu:.2e} | µ in {sweep_values}",
            style="bold",
        )

    t0 = time.perf_counter()

    # Run N parameter values × stars
    run_specs = []
    for param_val in sweep_values:
        lmbda = param_val if compare_param == "lambda" else lmbda_fixed
        m_val = fixed_mu if compare_param == "lambda" else param_val
        params = make_params(lmbda, m_val)
        for star_label, rho0 in jobs:
            run_specs.append((param_val, lmbda, m_val, star_label, params, rho0))

    results = {}  # (param_val, star_label) -> (ok, result)

    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = []
        for param_val, lmbda, m_val, star_label, params, rho0 in run_specs:
            futures.append(
                (
                    param_val,
                    lmbda,
                    m_val,
                    star_label,
                    ex.submit(run_solve_model_captured, params, rho0),
                )
            )

        for param_val, lmbda, m_val, star_label, fut in futures:
            label, out, ok, result = fut.result()
            if compare_param == "lambda":
                custom_print(
                    f"\n===== {star_label} | λ={lmbda:g} log (ok={ok}) =====",
                    style="bold",
                )
            else:
                custom_print(
                    f"\n===== {star_label} | μ={m_val:g} log (ok={ok}) =====",
                    style="bold",
                )
            print(out, end="")
            results[(param_val, star_label)] = (ok, result)

    custom_print(f"\nExecution time: {time.perf_counter() - t0:.0f} s", style="dim")

    # Plot per star type, overlaying the parameter sweep
    for star_label, _rho0 in jobs:
        entries_by_param = {}
        vacuum_sols = None
        nu_val = None

        for param_val in sweep_values:
            ok, result = results.get((param_val, star_label), (False, None))
            if not ok or result is None:
                continue

            plot_entries = result.get("plot_entries", [])
            if plot_entries:
                entries_by_param[param_val] = plot_entries

            if vacuum_sols is None:
                vacuum_sols = result.get("vacuum_sols", None)

            if nu_val is None:
                nu_val = result.get("nu_val", nu)

        if not entries_by_param:
            custom_print(f"\nNo comparable solutions found for {star_label} star.", color="yellow")
            continue

        if compare_param == "lambda":
            compare_path = f"Results/compare/lambda_{sweep_values[0]:g}_vs_{sweep_values[1]:g}_mu_{m_fixed:g}/"
            title = rf"$\xi={float(xi):g}$, $\mu={float(m_fixed):g}\,$eV, $v={float(nu_val):g}\,$eV"
        else:
            compare_path = f"Results/compare/mu_{sweep_values[0]:g}_vs_{sweep_values[1]:g}_lambda_{lmbda_fixed:g}/"
            title = rf"$\xi={float(xi):g}$, $\lambda={float(lmbda_fixed):g}$, $v={float(nu_val):g}\,$eV"

        plotResults_compare(
            entries_by_param=entries_by_param,
            xi=xi,
            m=m_fixed if compare_param == "lambda" else sweep_values[0],
            nu=nu_val,
            vacuum_sols=vacuum_sols if vacuum_sols is not None else {"+": 0, "-": 0},
            savepath=compare_path,
            star_label=star_label,
            param=compare_param,
            title=title,
            # sigma_zoom={
            #    "xlim": (0.0, 0.1),
            #    "ylim": (1e-4, 5e-2),
            #    "loc": "center",
            #    "width": "20%",
            #    "height": "20%",
            #    "mark": True,
            #    "mark_loc1": 2,
            #    "mark_loc2": 4,
            # },
        )


if __name__ == "__main__":
    main()
