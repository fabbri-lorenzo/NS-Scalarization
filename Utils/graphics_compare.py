# Utils/graphics_compare.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, SymmetricalLogLocator, FuncFormatter

from Utils.params import M
from Utils.graphics_single import (
    nu_line,
    _extract_mode_and_sign,
    _choose_color,
    _signed_label,
    _resample_sol_component,
)

dpi_val = 600
    
def clean_x(val, pos=None):
    # integers -> no decimals
    if np.isclose(val, round(val), atol=1e-10):
        return str(int(round(val)))
    # others like 0.5 -> keep as is
    return f"{val:g}"    

def even_decade_only(val, pos=None):
    if val == 0:
        return "0"
    sign = "-" if val < 0 else ""
    v = abs(val)

    exp = np.log10(v)
    if not np.isfinite(exp):
        return ""

    e = int(np.round(exp))

    # label only exact decades (1 * 10^e)
    if not np.isclose(v, 10**e, rtol=0, atol=v * 1e-12):
        return ""

    # label only even exponents
    if (e % 2) != 0:
        return ""

    return rf"${sign}10^{{{e}}}$"

def plotResults_compare_lambda(
    entries_by_lambda,
    xi,
    m,
    nu,
    vacuum_sols,
    savepath,
    *,
    lambda_labels=None,
    lambda_linestyles=None,
    n_resample=4000,
    lw=1.6,
):
    """
    Overlay plots for multiple lambda values.
    entries_by_lambda: dict {lmbda_value: entries_list}
      where each entries_list is the same structure you already pass to plotResults_multi.
    Distinguishes lambdas via line style.
    """
    os.makedirs(savepath, exist_ok=True)

    lambdas = list(entries_by_lambda.keys())

    if lambda_labels is None:
        lambda_labels = {l: rf"$\lambda={l:g}$" for l in lambdas}

    if lambda_linestyles is None:
        base = ["-", "--", ":", "-."]
        lambda_linestyles = {l: base[i % len(base)] for i, l in enumerate(lambdas)}

    # ========== PRESSURE ==========
    plt.figure()
    for lmbda in lambdas:
        entries = entries_by_lambda[lmbda]
        ls = lambda_linestyles[lmbda]

        for e in entries:
            sol = e["sol"]
            r_plot_m = np.linspace(sol.t[0], sol.t[-1], n_resample)  # meters
            p_plot = _resample_sol_component(sol, idx=0, r_plot=r_plot_m)

            n_val, vac_sign = _extract_mode_and_sign(e, nu)
            color = _choose_color(n_val, vac_sign, nu)

            label = _signed_label(r"P", n_val, vac_sign, nu)

            R_s_m = e["r_star"] * 1e3  # km -> m
            x_norm = r_plot_m / R_s_m

            plt.plot(x_norm, p_plot, color=color, linestyle=ls, linewidth=lw, label=label)

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xlabel(r"$r/R_s$")
    plt.xlim(left=0, right=2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.tick_params(axis="x", which="major", length=4, width=1.2)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)
    plt.ylabel("P [Pa]")
    plt.grid(False)
    # Main legend (modes/colors)
    main_legend = plt.legend(loc="upper left", frameon=False)
    
    # Lambda legend (linestyle mapping)
    lambda_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=lambda_linestyles[l],
            linewidth=1.6,
            label=lambda_labels[l]
        )
        for l in lambdas
    ]
    
    lambda_legend = plt.legend(
        handles=lambda_handles,
        loc="upper right",
        frameon=True,
    )
    
    plt.gca().add_artist(main_legend)
    plt.gca().add_artist(lambda_legend)
    
    plt.title("Pressure")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "pressure_compare_lambda.png"), dpi=dpi_val)

    # ========== MU_EFF^2 ==========
    plt.figure()
    for lmbda in lambdas:
        entries = entries_by_lambda[lmbda]
        ls = lambda_linestyles[lmbda]

        for e in entries:
            r_mu = e.get("r_mu", np.array([]))
            mu2 = e.get("mu2", np.array([]))
            if r_mu is None or mu2 is None or r_mu.size < 2:
                continue

            # r_mu is meters -> convert to km for interpolation
            r_mu_km = r_mu / 1e3
            order = np.argsort(r_mu_km, kind="mergesort")
            r_mu_km = r_mu_km[order]
            mu2_m = mu2[order]

            # unique x
            r_mu_km, unique_idx = np.unique(r_mu_km, return_index=True)
            mu2_m = mu2_m[unique_idx]
            if r_mu_km.size < 2:
                continue

            r_plot_km = np.linspace(r_mu_km[0], r_mu_km[-1], n_resample)  # km
            mu2_plot_km = np.interp(r_plot_km, r_mu_km, mu2_m) * 1e6  # m^-2 -> km^-2

            n_val, vac_sign = _extract_mode_and_sign(e, nu)
            color = _choose_color(n_val, vac_sign, nu)

            label = _signed_label(r"\mu_{\rm eff}^2", n_val, vac_sign, nu)

            # IMPORTANT: here everything is in km
            R_s_km = e["r_star"]  # already km
            x_norm = r_plot_km / R_s_km

            plt.plot(x_norm, mu2_plot_km, color=color, linestyle=ls, linewidth=lw, label=label)

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xlabel(r"$r/R_s$")
    plt.ylabel(r"$\mu_{\rm eff}^2$ [Km$^{-2}$]")
    ax = plt.gca()
    linthresh = 1e-12
    
    plt.yscale("symlog", linthresh=linthresh)
    
    # ticks that work on BOTH positive and negative sides
    ax.yaxis.set_major_locator(
        SymmetricalLogLocator(base=10, linthresh=linthresh, subs=(1.0,))
    )
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(base=10, linthresh=linthresh, subs=np.arange(2, 10) * 0.1)
    )
    
    # label only even decades
    ax.yaxis.set_major_formatter(FuncFormatter(even_decade_only))
    
    ax.tick_params(axis="y", which="major", length=6, width=1)
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)
    
    plt.xlim(left=0, right=2)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(clean_x))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.tick_params(axis="x", which="major", length=4, width=1.2)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)    
    plt.title("Effective mass squared")
    plt.grid(False)
    # Main legend (modes/colors)
    main_legend = plt.legend(loc="upper left", frameon=False)
    
    # Lambda legend (linestyle mapping)
    lambda_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=lambda_linestyles[l],
            linewidth=1.6,
            label=lambda_labels[l]
        )
        for l in lambdas
    ]
    
    lambda_legend = plt.legend(
        handles=lambda_handles,
        loc="upper right",
        frameon=True,
    )
    
    ax = plt.gca()
    ax.margins(y=0.05)  # 5% headroom

    
    plt.gca().add_artist(main_legend)
    plt.gca().add_artist(lambda_legend)
    
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "mu2_compare_lambda.png"), dpi=dpi_val, bbox_inches="tight")

    # ========== SIGMA ==========
    plt.figure()
    for lmbda in lambdas:
        entries = entries_by_lambda[lmbda]
        ls = lambda_linestyles[lmbda]

        for e in entries:
            sol = e["sol"]
            r_plot_m = np.linspace(sol.t[0], sol.t[-1], n_resample)  # meters
            sigma_plot = _resample_sol_component(sol, idx=2, r_plot=r_plot_m)

            n_val, vac_sign = _extract_mode_and_sign(e, nu)
            color = _choose_color(n_val, vac_sign, nu)

            label = _signed_label(r"\sigma/M_{Pl}", n_val, vac_sign, nu)

            R_s_m = e["r_star"] * 1e3  # km -> m
            x_norm = r_plot_m / R_s_m

            plt.plot(x_norm, sigma_plot / M, color=color, linestyle=ls, linewidth=lw, label=label)

    nu_line(nu, vacuum_sols)
    plt.xlabel(r"$r/R_s$")
    plt.xlim(left=0, right=2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(clean_x))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.tick_params(axis="x", which="major", length=4, width=1.2)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)
    ax = plt.gca()
    linthresh = 1e-8
    
    plt.yscale("symlog", linthresh=linthresh)
    
    # ticks that work on BOTH positive and negative sides
    ax.yaxis.set_major_locator(
    SymmetricalLogLocator(base=10, linthresh=linthresh, subs=(1.0,))
)
    ax.yaxis.set_minor_locator(
    SymmetricalLogLocator(base=10, linthresh=linthresh, subs=np.arange(2, 10) * 0.1)
)
    # label only even decades (your function already supports negatives)
    ax.yaxis.set_major_formatter(FuncFormatter(even_decade_only))
    
    ax.tick_params(axis="y", which="major", length=6, width=1)
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)
    
    ax = plt.gca()
    ax.margins(y=0.05)  # 5% headroom

    
    
    # Main legend (modes/colors)
    main_legend = plt.legend(loc="center right", bbox_to_anchor=(1, 0.65),  frameon=True)
    
    # Lambda legend (linestyle mapping)
    lambda_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=lambda_linestyles[l],
            linewidth=1.6,
            label=lambda_labels[l]
        )
        for l in lambdas
    ]
    
    lambda_legend = plt.legend(
        handles=lambda_handles,
        loc="upper right",
        bbox_to_anchor=(1, 0.9), 
        frameon=True,
    )
    
    plt.gca().add_artist(main_legend)
    plt.gca().add_artist(lambda_legend)
     
    plt.grid(False)
    plt.title(rf"$\xi={float(xi):g}$, $\mu={float(m):g}$, $v={float(nu):g}$")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "sigma_compare_lambda.png"), dpi=dpi_val)