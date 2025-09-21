import os
import matplotlib.pyplot as plt
import numpy as np
from Utils.params import c, G_N, M, SM


R_star_color = 'c'
dpi_val = 600


def printResults_multi(entries, xi, star_weight):
    """
    entries: list of dicts with keys
      ['label','sol','r_star','r_mu','mu2']
    Makes one figure per quantity, overlaying curves for each entry.
    """
    savepath = f"Results/Results_{star_weight}_xi={xi:.0f}/"
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
    star_radii = []
    for i, e in enumerate(entries):
        sol = e["sol"]
        sty = styles[i % len(styles)]
        # smooth resampling from dense solution
        r_plot = np.linspace(sol.t[0], sol.t[-1], 4000)
        if hasattr(sol, "sol") and callable(sol.sol):
            y_plot = sol.sol(r_plot)
            p_plot = y_plot[0]
        else:
            p_plot = np.interp(r_plot, sol.t, sol.y[0])
        plt.plot(r_plot / 1e3, p_plot, **sty)
        star_radii.append(e["r_star"])
    avg_R_star = np.mean([r for r in star_radii if r is not None])
    plt.axvline(x=avg_R_star, color=R_star_color, linestyle="--", alpha=0.25)
    plt.xlabel('r (Km)')
    plt.ylabel("P (Pa)")
    plt.grid(True); plt.title('Pressure profile EMG')
    plt.tight_layout()
    plt.savefig(savepath + "pressure.png", dpi=dpi_val)
    plt.show()

    #    # --- Mass and metric ---
    # plt.figure()
    # for i, e in enumerate(entries):
    #    sol = e["sol"]; sty = styles[i % len(styles)]
    #    r_plot = np.linspace(sol.t[0], sol.t[-1], 4000)
    #    if hasattr(sol, "sol") and callable(sol.sol):
    #        y_plot = sol.sol(r_plot)
    #        m_plot = y_plot[1]
    #        phi_plot = y_plot[2]
    #    else:
    #        m_plot   = np.interp(r_plot, sol.t, sol.y[1])
    #        phi_plot = np.interp(r_plot, sol.t, sol.y[2])
    #
    #    Phi_norm_plot = phi_plot - phi_plot[-1]
    #    plt.plot(r_plot/1e3, m_plot*(c*c/G_N)/SM, label=f"m(r) [{e['label']}]", **sty)
    #
    #
    # plt.axvline(x=avg_R_star, color=R_star_color, linestyle='--', alpha=0.25)
    #
    # for i, e in enumerate(entries):
    #    sol = e["sol"]; sty = styles[i % len(styles)].copy(); sty.update(alpha=0.6)
    #    r_plot = np.linspace(sol.t[0], sol.t[-1], 4000)
    #    if hasattr(sol, "sol") and callable(sol.sol):
    #        phi_plot = sol.sol(r_plot)[2]
    #    else:
    #        phi_plot = np.interp(r_plot, sol.t, sol.y[2])
    #    Phi_norm_plot = phi_plot - phi_plot[-1]
    #    plt.plot(r_plot/1e3, np.exp(Phi_norm_plot), label=f"e^(2Φ) [{e['label']}]", **sty)
    #
    # plt.xlabel('r (Km)')
    # plt.legend(); plt.grid(True); plt.title('Mass & Metric profiles EMG')
    # plt.tight_layout()
    # plt.savefig(savepath + "mass_metric.png", dpi=dpi_val)
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
        mu2_m = mu2[order]  # mu2 in m^-2

        # drop duplicate radii if any (interp needs strictly increasing x)
        unique_x, unique_idx = np.unique(r_mu_km, return_index=True)
        r_mu_km = unique_x
        mu2_m   = mu2_m[unique_idx]

        # uniform grid over available range (smooth look)
        if r_mu_km.size < 2:
            continue
        r_plot = np.linspace(r_mu_km[0], r_mu_km[-1], 4000)

        # interpolate and convert to Km^-2: (1 m^-2) = 1e6 Km^-2
        mu2_plot_km = np.interp(r_plot, r_mu_km, mu2_m) * 1e6

        sty = styles[i % len(styles)]
        plt.plot(r_plot, mu2_plot_km, label=rf"$\mu_{{\rm eff}}^2$ [{e['label']}]", **sty)

    plt.axvline(x=avg_R_star, color=R_star_color, linestyle="--", alpha=0.25)
    plt.axhline(0.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
    plt.xlabel('r (Km)'); plt.ylabel(r'$\mu_{\rm eff}^2$ (Km$^{-2}$)')
    plt.title('Effective Mass Squared Profile')
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "mu2.png", dpi=dpi_val, bbox_inches="tight")
    plt.show()

    # --- Scalar field σ(r) ---
    plt.figure()
    for i, e in enumerate(entries):
        sol = e["sol"]; sty = styles[i % len(styles)]
        r_plot = np.linspace(sol.t[0], sol.t[-1], 4000)
        if hasattr(sol, "sol") and callable(sol.sol):
            sigma_plot = sol.sol(r_plot)[2]
        else:
            sigma_plot = np.interp(r_plot, sol.t, sol.y[2])
        plt.plot(r_plot/1e3, sigma_plot/M, label=f"σ/M_Pl [{e['label']}]", **sty)

    plt.axvline(x=avg_R_star, color=R_star_color, linestyle="--", alpha=0.25)
    plt.xlabel('r (Km)')
    plt.legend(); plt.grid(True)
    plt.title('Scalar field σ(r) for EMG')
    plt.tight_layout()
    plt.savefig(savepath + "sigma_profile.png", dpi=dpi_val)
    plt.show()

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
# plt.xlabel('r (Km)')
# plt.legend(); plt.grid(True)
# plt.title('Scalar field σ(r) for EMG')
# plt.tight_layout()
# plt.savefig(savepath + "sigma_norm.png", dpi=dpi_val)
# plt.show()

# --- Colored printing utility ---
from colorama import Fore, Style, init

# Initialize colorama (important for Windows, harmless on Mac/Linux)
init(autoreset=True)

def custom_print(text, color="white", style="normal"):
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

    print(f"{style_code}{color_code}{text}{Style.RESET_ALL}")
