"""Baseline GR TOV integration using the SLy4 equation of state (no scalar)."""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from Utils.params import c, G_N, M2, SM, rho0_lightS, rho0_heavyS
from Utils.EOS import p_SLy4, rho_SLy4

p_eqState = p_SLy4       # Pressure as function of density
rho_eqState = rho_SLy4   # Density as function of pressure


def _fit_polytrope_to_SLy4(rho_ref, span=2.5, n=100):
    lo = max(rho_ref / span, rho_ref * 1e-3)
    hi = rho_ref * span
    rhos = np.geomspace(lo, hi, n)
    ps = p_SLy4(rhos)
    b, a = np.polyfit(np.log(rhos), np.log(ps), 1)  # y = b x + a
    Gamma = b
    K = np.exp(a)
    return K, Gamma


# === Initial Conditions ===
def initial_conditions(r0, rho0=rho0_lightS):
    """Central expansion for pure-GR star used as a sanity check."""
    p0 = p_eqState(rho0)
    m0 = (4/3) * np.pi * rho0 * r0**3 *G_N/(c*c) #reduced mass
    Phi2 = (p0+2*rho0/3)/(4*M2) * r0**2  # from TOV, O(r^2) expansion
    Phi_c = 0.0
    p2 = -(p0+rho0*c*c)*Phi2 
    p_c = p0 + p2* r0**2
    m_c = m0 
    return [p_c, m_c, Phi_c]


# === System of First-Order ODEs ===
def tov_system(r, y):
    p, m, Phi= y

    if p <= 1e-3:
        return [0.0, 0.0, 0.0]   # stop evolving once surface reached

    rho = rho_eqState(p)
    eps = rho*c*c 

    den = r-2*m

    # Check for near Schwarzschild condition
    if den <= 0:
        raise RuntimeError(f"r ≤ 2m (would hit a horizon) at r={r:.3g}")

    # dPhi/dr
    dPhi = (2*m + r**3*(p/M2)) / (2*r*den)

    # dm/dr
    dm = r*r/2 * (eps/M2)

    # dp/dr
    dp = -(eps + p) * dPhi

    # print(" P=",p, "\n")
    return [dp, dm, dPhi]


# === Integrator Wrapper ===
def integrate_star(r0, r_max, rho0=rho0_lightS, verbose=True):
    """Integrate GR TOV from r0 to r_max, stopping at the surface."""
    r_span = (r0, r_max)
    y0 = initial_conditions(r0, rho0=rho0)
    p_c = y0[0]

    def surface(r, y):
        return y[0] - 1e-10 * p_c  # pressure
    surface.terminal = True
    surface.direction = -1

    sol = solve_ivp(
        tov_system,
        r_span,
        y0,
        method="BDF",
        rtol=1e-12,
        atol=1e-10,
        events=surface,
    )

    if verbose:
        print("» solver msg:", sol.message)
    return sol


def sci_cdot(y, pos):
    if y == 0:
        return r"$0$"
    sgn = "-" if y < 0 else ""
    y = abs(y)
    exp = int(np.floor(np.log10(y)))
    mant = y / (10**exp)
    if np.isclose(mant, 1.0):
        return rf"${sgn}10^{{{exp}}}$"
    return rf"${sgn}{mant:.3g}\cdot10^{{{exp}}}$"


KNOWN_GR_MASSES = {
    "light": 1.12,
    "heavy": 2.08,
}


def mass_in_solar_masses(sol):
    """Return the final gravitational mass from a GR TOV solution."""
    return sol.y[1, -1] * c**2 / (G_N * SM)


def plot_mass_vs_central_density(
    rho0_min=5e17,
    rho0_max=4e18,
    n_points=40,
    r0=1e-2,
    r_max=3e5,
    savepath="Tests/Results_rhoe17",
):
    """Scan central density and plot stellar mass in solar masses."""
    os.makedirs(savepath, exist_ok=True)

    rho0_values = np.linspace(rho0_min, rho0_max, n_points)
    masses = np.full_like(rho0_values, np.nan, dtype=float)

    for i, rho0 in enumerate(rho0_values):
        sol = integrate_star(r0, r_max, rho0=rho0, verbose=False)
        if sol.success and sol.t_events and len(sol.t_events[0]) > 0:
            masses[i] = mass_in_solar_masses(sol)
        else:
            print(f"Skipped rho0={rho0:.3e}: {sol.message}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rho0_values, masses, color="black", linewidth=1.8)
    star_colors = {"L": "#1f77b4", "H": "#d62728"}
    ax.axvline(
        rho0_lightS,
        color=star_colors["L"],
        linestyle="--",
        alpha=0.7,
        label="Light star",
    )
    ax.axvline(
        rho0_heavyS,
        color=star_colors["H"],
        linestyle="--",
        alpha=0.7,
        label="Heavy star",
    )

    ax.set_xlim(rho0_min, rho0_max)
    ax.set_xlabel(r"$\rho_0$ [Kg/m$^3$]")
    ax.set_ylabel(r"$\mathcal{M}$ [$M_{\odot}$]")
    ax.xaxis.set_major_formatter(FuncFormatter(sci_cdot))
    ax.xaxis.get_offset_text().set_visible(False)
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(savepath, "mass_vs_rho0.png")
    plt.savefig(out_path, dpi=600)
    plt.close(fig)
    return rho0_values, masses, out_path


# === Testing GR reduction ===
if __name__ == "__main__":

    r0 = 1e-2  # Start radius in m
    r_max = 3e5  # Max radius in m
    r_points = int(1e8)
    r_span = (r0, r_max)
    r_eval = np.linspace(*r_span, r_points)
    sol = integrate_star(r0, r_max)

    r = sol.t  # m
    P = sol.y[0]  # Pa
    rho = rho_SLy4(P)  # kg/m^3

    Phi_surf = sol.y[2, -1]
    Phi_norm = sol.y[2] - Phi_surf

    rho = rho_eqState(sol.y[0])  # kg/m^3

    R = r[-1]

    print("Integration successful:", sol.success)
    print("Final radius:", R / 1e3, "Km")
    print("Final pressure:", sol.y[0][-1], "Pa")
    print(
        "Final mass:", sol.y[1][-1] * (c * c / G_N) / SM, "Solar Masses"
    )  # in Solar Masses
    _, _, mass_scan_path = plot_mass_vs_central_density()
    print("Mass scan plot:", mass_scan_path)

    ## --- Polytrope fitted to SLy4 near the central density ---
    # K_poly, Gamma_poly = _fit_polytrope_to_SLy4(rho0_lightS, span=2.5, n=120)
    #
    # def p_poly(rho):
    #    return K_poly * rho**Gamma_poly
    #
    # def rho_poly(p):
    #    return (p / K_poly) ** (1.0 / Gamma_poly)
    #
    # _p_eq_old, _rho_eq_old = p_eqState, rho_eqState
    # p_eqState, rho_eqState = p_poly, rho_poly
    # sol_poly = integrate_star(r0, r_max)
    # r_poly = sol_poly.t
    # P_poly = sol_poly.y[0]
    # rho_poly_prof = rho_eqState(P_poly)
    # p_eqState, rho_eqState = _p_eq_old, _rho_eq_old
    #
    # print("Integration successful:", sol_poly.success)
    # print("Final radius:", r_poly[-1] / 1e3, "Km")
    # print("Final pressure:", sol_poly.y[0][-1], "Pa")
    # print(
    #    "Final mass:", sol_poly.y[1][-1] * (c * c / G_N) / SM, "Solar Masses"
    # )  # in Solar Masses

    # ===== Plotting section: pressure + mass profiles =====
    # main EoS solution
    # r = sol.t
    # P = sol.y[0]
    # R = r[-1]

    # polytrope solution
    # r_poly = sol_poly.t
    # P_poly = sol_poly.y[0]

    # fig, (axP, axM) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 7))
#
## -------------------------
## Pressure subplot
## -------------------------
# axP.plot(r / 1e3, P, color="blue", label="Pressure SLy4")
# axP.plot(r_poly / 1e3, P_poly, color="red", label="Pressure Polytrope")
#
# axP.axvline(
#    x=R / 1e3, color="blue", linestyle="--", alpha=0.6, label="Star radius SLy4"
# )
# axP.axvline(
#    x=r_poly[-1] / 1e3,
#    color="red",
#    linestyle="--",
#    alpha=0.6,
#    label="Star radius Polytrope",
# )
#
# axP.set_ylabel("P [Pa]")
# axP.set_xlim(left=0)
#
# axP.yaxis.set_major_formatter(FuncFormatter(sci_cdot))
# axP.yaxis.get_offset_text().set_visible(False)
#
# axP.grid(True, which="both", alpha=0.2)
# axP.legend()
#
## -------------------------
## Mass subplot
## -------------------------
# M_sun = 1.98847e30  # kg
#
# M_prof = sol.y[1] * (c**2 / G_N) / M_sun
# M_poly_prof = sol_poly.y[1] * (c**2 / G_N) / M_sun
#
# axM.plot(r / 1e3, M_prof, color="blue", label="Mass SLy4")
# axM.plot(r_poly / 1e3, M_poly_prof, color="red", label="Mass Polytrope")
#
# axM.axvline(
#    x=R / 1e3, color="blue", linestyle="--", alpha=0.6, label="Star radius SLy4"
# )
# axM.axvline(
#    x=r_poly[-1] / 1e3,
#    color="red",
#    linestyle="--",
#    alpha=0.6,
#    label="Star radius Polytrope",
# )
#
# axM.set_xlabel("r [km]")
# axM.set_ylabel(r"$M [M_\odot]$")
# axM.grid(True, which="both", alpha=0.2)
# axM.legend()
#
# plt.tight_layout()
# plt.savefig("Tests/Results_rhoe17/comparison.png", dpi=600)
# plt.show()
