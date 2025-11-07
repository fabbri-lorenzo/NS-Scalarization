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
def initial_conditions(r0):
    rho0 = rho0_lightS # Density in Kg/m^3 - will give 1.12 solar masses in GR
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
     
    #print(" P=",p, "\n")
    return [dp, dm, dPhi]

# === Integrator Wrapper ===
def integrate_star(r0, r_max):
    r_span = (r0, r_max)
    y0 = initial_conditions(r0)
    p_c = y0[0]

    def surface(r, y):
        return y[0] - 1e-15*p_c  # pressure
    surface.terminal = True
    surface.direction = -1

    sol = solve_ivp(
        tov_system,
        r_span,
        y0,
        method="RK45",
        rtol=1e-6,
        atol=np.array([1e-8, 1e-10, 1e-10]),
        events=surface,
    )

    print("» solver msg:", sol.message)
    return sol


def shade_crust_regions(ax, R_m, r_cc_m, r_drip_m):
    """
    Shade core / inner crust / outer crust on the current axes (x in km)
    and return handles for a secondary legend.
    """
    km = 1e3
    ax.set_axisbelow(True)

    spans = [
        (0, r_cc_m / km, "#74c476", "Core"),
        (r_cc_m / km, r_drip_m / km, "#fd8d3c", "Inner crust"),
        (r_drip_m / km, R_m / km, "#9ecae1", "Outer crust"),
    ]

    region_handles, region_labels = [], []
    for x0, x1, color, label in spans:
        ax.axvspan(x0, x1, facecolor=color, alpha=0.25, ec="none", zorder=0)
        region_handles.append(Patch(facecolor=color, alpha=0.25, ec="none"))
        region_labels.append(label)

    # Return handles/labels for secondary legend
    return region_handles, region_labels


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

    # --- Polytrope fitted to SLy4 near the central density ---
    K_poly, Gamma_poly = _fit_polytrope_to_SLy4(rho0_lightS, span=2.5, n=120)

    def p_poly(rho):
        return K_poly * rho**Gamma_poly

    def rho_poly(p):
        return (p / K_poly) ** (1.0 / Gamma_poly)

    _p_eq_old, _rho_eq_old = p_eqState, rho_eqState
    p_eqState, rho_eqState = p_poly, rho_poly
    sol_poly = integrate_star(r0, r_max)
    r_poly = sol_poly.t
    P_poly = sol_poly.y[0]
    rho_poly_prof = rho_eqState(P_poly)
    p_eqState, rho_eqState = _p_eq_old, _rho_eq_old

    print("Integration successful:", sol_poly.success)
    print("Final radius:", r_poly[-1] / 1e3, "Km")
    print("Final pressure:", sol_poly.y[0][-1], "Pa")
    print(
        "Final mass:", sol_poly.y[1][-1] * (c * c / G_N) / SM, "Solar Masses"
    )  # in Solar Masses

    # ===== Plotting section =====
    fig, ax = plt.subplots()

    ax.plot(r / 1e3, P, color="blue", label="Pressure [Pa] (SLy4)")
    ax.plot(r / 1e3, rho * c * c, color="teal", label="Energy density [J/m³] (SLy4)")

    # Polytrope overlays
    ax.plot(
        r_poly / 1e3,
        P_poly,
        linestyle="--",
        color="blue",
        alpha=0.8,
        label="Pressure [Pa] (poly)",
    )
    ax.plot(
        r_poly / 1e3,
        rho_poly_prof * c * c,
        linestyle="--",
        color="teal",
        alpha=0.8,
        label="Energy density [J/m³] (poly)",
    )

    # Stellar radius line
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(
        x=R / 1e3, color="c", linestyle="--", alpha=0.75, label="Star radius (SLy4)"
    )
    # ax.axvline(
    #    x=r_poly[-1] / 1e3,
    #    color="slateblue",
    #    linestyle="--",
    #    alpha=0.75,
    #    label="Star radius (poly)",
    # )

    ax.set_xlabel("r [Km]")
    ax.set_xlim(left=0)

    # --- 10^n tick labels, no ×10^n offset ---
    ax.yaxis.set_major_formatter(FuncFormatter(sci_cdot))
    ax.yaxis.get_offset_text().set_visible(False)

    # === Define radii (m) ===
    R_m = R
    r_cc_m = 10.4e3
    r_drip_m = 11.4e3

    # === Shade regions and collect handles ===
    region_handles, region_labels = shade_crust_regions(ax, R_m, r_cc_m, r_drip_m)

    # === First legend: physical quantities ===
    legend1 = ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.35), framealpha=0.9)
    # legend1 = ax.legend(loc="best", framealpha=0.9)

    ## === Second legend: structure ===
    legend2 = ax.legend(
        region_handles,
        region_labels,
        title="Stellar Regions (SLy4)",
        loc="center left",
        bbox_to_anchor=(0.01, 0.6),
        framealpha=0.9,
    )
    ax.add_artist(legend1)  # ensure both legends appear

    ax.grid(False)
    fig.tight_layout()
    fig.savefig("Tests/Results_rhoe17/pressureGR_shaded.png", dpi=300)
    plt.show()

    # --- Mass and metric ---
    plt.plot(
        r / 1e3,
        sol.y[1] * (c * c / G_N) / (SM),
        color="indigo",
        label="Mass m(r) in Solar Masses",
    )
    plt.plot(
        sol.t / 1e3, np.exp(Phi_norm), color="darkcyan", label="Metric Function e^2Φ(r)"
    )
    plt.xlabel("r [Km]")
    plt.legend()
    plt.grid(True)
    plt.title("Mass Metric GR")
    plt.savefig("Tests/Results_rhoe17/mass_metricGR.png", dpi=300)
    plt.show()
