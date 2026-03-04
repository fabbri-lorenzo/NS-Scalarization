import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from Utils.params import M


R_star_color = 'c'
dpi_val = 600

pos_root_colors = [
    "#1f77b4",
    "#9305FF",
    "#f4828f",
    "#1e988a",
    "#e1de2c",
    "#0c8a32",
    "#704809",
    "#b41f44",
]

neg_root_colors = [
    "#4e91c0",  # n=0, -ν
    "#a185d3",  # n=1, –ν
    "#f0a3ab",  # n=2, –ν
    "#5fb4a9",  # n=3, –ν
]


def nu_line(nu, vacuum_sols):
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    if nu != 0.0:
        if vacuum_sols["+"] > 0:
            plt.axhline(
                y=nu / M,
                color="pink",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=r"$v/M_{Pl}$",
            )
        if vacuum_sols["-"] > 0:
            plt.axhline(
                y=-nu / M,
                color="magenta",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=r"$-v/M_{Pl}$",
            )


def _extract_mode_and_sign(entry, nu):
    # n from "n=..." label; default 0
    try:
        n = int(entry["label"].split("=")[1])
    except Exception:
        n = 0
    vac_sign = entry.get("vacuum_sign", 0)
    return n, vac_sign


def _choose_color(n, vac_sign, nu):
    if n >= len(pos_root_colors):
        return "#0a0002"
    palette = neg_root_colors if (nu != 0.0 and vac_sign < 0) else pos_root_colors
    return palette[n % len(palette)]


def _signed_label(base_math, n, vac_sign, nu):
    # base_math is a raw math string, e.g. r"\mu_{\rm eff}^2" or r"\sigma/M_{Pl}"
    if nu == 0.0 or vac_sign == 0:
        return rf"${base_math} \, [n={n}]$"
    s = "+" if vac_sign > 0 else "-"
    return rf"${base_math} \, [n={n}^{{{s}}}]$"


def _resample_sol_component(sol, idx, r_plot):
    if hasattr(sol, "sol") and callable(sol.sol):
        return sol.sol(r_plot)[idx]
    return np.interp(r_plot, sol.t, sol.y[idx])


def plotResults_multi(entries, xi, m, nu, vacuum_sols, savepath):
    """
    entries: list of dicts with keys
      ['label','sol','r_star','r_mu','mu2']
    Makes one figure per quantity, overlaying curves for each entry.
    """
    os.makedirs(savepath, exist_ok=True)

    # style cycle (extend if you add more modes)
    styles = [
        dict(linestyle='-',   linewidth=1.6),
        dict(linestyle='-',  linewidth=1.6),
        dict(linestyle='-',  linewidth=1.6),
        dict(linestyle='-',   linewidth=1.6),
    ]

    # --- Pressure profile ---
    plt.figure()

    avg_R_star = float(
        np.nanmean([e["r_star"] for e in entries if e.get("r_star") is not None])
    )

    for i, e in enumerate(entries):
        sol = e["sol"]
        r_plot = np.linspace(sol.t[0], sol.t[-1], 4000)
        p_plot = _resample_sol_component(sol, idx=0, r_plot=r_plot)

        n_val, vac_sign = _extract_mode_and_sign(e, nu)
        color = _choose_color(n_val, vac_sign, nu)
        label = _signed_label(r"P", n_val, vac_sign, nu)

        sty = styles[i % len(styles)]
        R_s = e["r_star"] * 1e3  # convert km → m to match sol.t units
        x_norm = r_plot / R_s
        plt.plot(x_norm, p_plot, color=color, label=label, **sty)

    # plt.axvline(
    #    x=avg_R_star,
    #    color=R_star_color,
    #    linestyle="--",
    #    alpha=0.75,
    #    label="Star radius",
    # )
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
    plt.legend()
    plt.title("Pressure")
    plt.tight_layout()
    plt.savefig(savepath + "pressure.png", dpi=dpi_val)
    # plt.show()

    # --- Effective mass squared profile (uniform resample of logged points) ---
    plt.figure()
    for i, e in enumerate(entries):
        r_mu, mu2 = e["r_mu"], e["mu2"]
        if r_mu.size == 0:
            continue

        # ensure strictly increasing x for interpolation
        r_mu_km = r_mu / 1e3
        order = np.argsort(r_mu_km, kind="mergesort")
        r_mu_km = r_mu_km[order]
        mu2_m = mu2[order]

        unique_x, unique_idx = np.unique(r_mu_km, return_index=True)
        r_mu_km = unique_x
        mu2_m   = mu2_m[unique_idx]

        if r_mu_km.size < 2:
            continue

        r_plot = np.linspace(r_mu_km[0], r_mu_km[-1], 4000)
        mu2_plot_km = np.interp(r_plot, r_mu_km, mu2_m) * 1e6  # m^-2 -> Km^-2

        n_val, vac_sign = _extract_mode_and_sign(e, nu)
        color = _choose_color(n_val, vac_sign, nu)
        label = _signed_label(r"\mu_{\rm eff}^2", n_val, vac_sign, nu)

        sty = styles[i % len(styles)]
        R_s = e["r_star"] * 1e3  # convert km → m to match sol.t units
        x_norm = r_plot / R_s
        plt.plot(x_norm, mu2_plot_km, color=color, label=label, **sty)
    # plt.axvline(
    #    x=avg_R_star,
    #    color=R_star_color,
    #    linestyle="--",
    #    alpha=0.75,
    #    label="Star radius",
    # )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xlabel(r"$r/R_s$")
    plt.ylabel(r"$\mu_{\rm eff}^2$ [Km$^{-2}$]")
    plt.xlim(left=0, right=2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.tick_params(axis="x", which="major", length=4, width=1.2)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)
    plt.title("Effective mass squared")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "mu2.png", dpi=dpi_val, bbox_inches="tight")
    # plt.show()

    # --- Scalar field σ(r) ---
    plt.figure()
    for i, e in enumerate(entries):
        sol = e["sol"]
        r_plot = np.linspace(sol.t[0], sol.t[-1], 4000)
        sigma_plot = _resample_sol_component(sol, idx=2, r_plot=r_plot)  # σ(r)

        n_val, vac_sign = _extract_mode_and_sign(e, nu)
        color = _choose_color(n_val, vac_sign, nu)
        label = _signed_label(r"\sigma/M_{Pl}", n_val, vac_sign, nu)

        sty = styles[i % len(styles)]
        R_s = e["r_star"] * 1e3  # convert km → m to match sol.t units
        x_norm = r_plot / R_s
        plt.plot(x_norm, sigma_plot / M, color=color, label=label, **sty)

    nu_line(nu, vacuum_sols)
    # plt.axvline(
    #    x=avg_R_star,
    #    color=R_star_color,
    #    linestyle="--",
    #    alpha=0.75,
    #    label="Star radius",
    # )
    plt.xlabel(r"$r/R_s$")
    plt.xlim(left=0, right=2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.tick_params(axis="x", which="major", length=4, width=1.2)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)
    plt.legend()
    plt.grid(False)
    plt.title(rf"$\xi={float(xi):g}$, $\mu={float(m):g}$, $v={float(nu):g}$")
    plt.tight_layout()
    plt.savefig(savepath + "sigma.png", dpi=dpi_val)
    # plt.show()

    ## --- Normalized scalar field ---
    # plt.figure()
    # for i, e in enumerate(entries):
    #    sol = e["sol"]; sty = styles[i % len(styles)]
    #    sigma0 = sol.y[2,0]
    #    r_plot = np.linspace(sol.t[0], sol.t[-1], 4000)
    #    if hasattr(sol, "sol") and callable(sol.sol):
    #        sigma_plot = sol.sol(r_plot)[3]
    #    else:
    #        sigma_plot = np.interp(r_plot, sol.t, sol.y[2])
    #
    #    plt.plot(r_plot/1e3, sigma_plot/sigma0, label=f"σ(r)/σ₀ [{e['label']}]", **sty)
    #
    ## vertical lines at R_star
    #
    # plt.axvline(x=avg_R_star, color=R_star_color, linestyle='--', alpha=0.25)
    #
    ## annotate sigma0 values in the corner of the plot
    # text_lines = []
    # for i, e in enumerate(entries):
    #    sigma0 = e["sol"].y[3,0]/M # in Plank masses
    #    text_lines.append(f"{e['label']}: σ₀ = {sigma0:.2e}")
    #
    # plt.text(
    #    0.98, 0.02, "\n".join(text_lines),
    #    transform=plt.gca().transAxes,
    #    va="bottom", ha="right",
    #    fontsize=9,
    #    bbox=dict(boxstyle="round", facecolor="white", alpha=1.0)
    # )
    #
    #
    # plt.xlabel('r [Km]')
    # plt.legend(); plt.grid(False)
    # plt.title('Scalar field σ(r)')
    # plt.tight_layout()
    # plt.savefig(savepath + "sigma_norm.png", dpi=dpi_val)
    # plt.show()


# --- Colored printing utility ---
import sys
from colorama import Fore, Style, init

# Initialize colorama (important for Windows, harmless on Mac/Linux)
init(autoreset=True)

def custom_print(text, color="white", style="normal", stream=None):
    colors = {
        "black": Fore.BLACK,
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
        "gray": Fore.LIGHTBLACK_EX,
    }

    styles = {
        "normal": "",
        "bold": Style.BRIGHT,
        "dim": Style.DIM,
    }

    color_code = colors.get(color.lower(), Fore.WHITE)
    style_code = styles.get(style.lower(), "")

    out = stream if stream is not None else sys.stdout
    print(f"{style_code}{color_code}{text}{Style.RESET_ALL}", file=out)
