import numpy as np
from scipy.integrate import solve_ivp

from Utils.params import c, G_N, M, SM, lmbda_EMG, rho0_lightS, rho0_heavyS
from Utils.TOV_EMG import initial_conditions, make_tov_EMG, StoppingConditions
from Utils.shooting import shoot_sigma0 
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.graphics import printResults_multi, custom_print

from tqdm import tqdm

# ---------- config ----------
p_eqState   = p_SLy4
rho_eqState = rho_SLy4

rho0    = rho0_lightS
xi_val  = 10
lmbda_val = lmbda_EMG
frac_pc = 1e-10

custom_print(f"\nxi = {xi_val} | λ = {lmbda_val:.2e} | ρ0 = {rho0:.1e}", style="bold")

# ---------- helpers ----------
def count_nodes_sigma(r, sigma):
    s = np.asarray(sigma, dtype=float)
    if s.size < 2:
        return 0
    nz = s != 0
    if nz.sum() < 2:
        return 0
    s = s[nz]
    sign = np.sign(s)
    return int(np.sum((sign[1:] * sign[:-1]) < 0.0))

def node_count_to_2R(sol, R_star_m, idx_sigma=2):
    if R_star_m is None or not np.isfinite(R_star_m):
        r = sol.t
        sigma = sol.y[idx_sigma]
    else:
        r = sol.t
        sigma = sol.y[idx_sigma]
        m = r <= 2.0 * R_star_m
        r = r[m]
        sigma = sigma[m]
    return count_nodes_sigma(r, sigma)

# Integrator wrapper (returns OdeResult, μ² log, and R_* in meters)
def integrate_star(sigma0, p_eqState, rho_eqState, r0, r_max, xi=xi_val, stop_at_2r=True, record_mu2=False):
    r_span = (r0, r_max)
    y0 = initial_conditions(r0, sigma0, p_eqState, xi, rho0)
    p_c = y0[0]

    mu2_log = []
    if record_mu2:
        tov = make_tov_EMG(p_c, frac_pc, rho_eqState, xi, mu2_recorder=mu2_log.append)
    else:
        tov = make_tov_EMG(p_c, frac_pc, rho_eqState, xi)

    events = StoppingConditions(p_c, frac_pc)
    ev_surface = events.pressure_limit()

    if stop_at_2r:
        ev_2R = events.double_radius()
        sol = solve_ivp(
            tov, r_span, y0, dense_output=True, method='RK45',
            rtol=1e-6, atol=1e-9, events=[ev_surface, ev_2R]
        )
    else:
        sol = solve_ivp(
            tov, r_span, y0, dense_output=True, method='RK45',
            rtol=1e-6, atol=1e-9
        )

    R_star_m = events.R_star  # set after solve
    return sol, mu2_log, R_star_m

# ---------- main ----------
if __name__ == "__main__":
    r0     = 1e-2         # m
    r_max  = 3e5          # m
    r_eval = np.linspace(r0, r_max, int(1e6))  # (not strictly needed here)

    # --- 1) Find ALL σ0 roots via fsolve (absolute-first, then relative) ---
    s0_list = shoot_sigma0(
        integrate_fn=integrate_star,
        p_eqState=p_eqState,
        rho_eqState=rho_eqState,
        r0=r0,
        r_max=r_max,
        xi=xi_val,
        bracket=(-1e-7*M, M),    # make sure units match σ0
        rho0=rho0,
        lmbda=lmbda_val,
        abs_threshold=1e-10*M,
        tol_relative=1e-2,
        n_seeds=41,
        xtol=1e-12,
        maxfev=400,
        idx_sigma=2,
        merge_tol=1e-8*M
    )

    if not s0_list:
        raise RuntimeError("No σ0 roots found in the given bracket.")

    # --- 2) For each σ0: integrate to 2R and count nodes ---
    per_mode = {}     # n -> list of dicts with entries
    all_entries = []  # for plotting

    for s0 in tqdm(s0_list, desc="\nIntegrating σ0 candidates to 2R", unit="sol"):
        sol, mu2_log, R_star_m = integrate_star(
            s0, p_eqState, rho_eqState, r0, r_max, stop_at_2r=True, record_mu2=True
        )

        if R_star_m is None:
            custom_print(f"[WARN] R_* not found for σ₀={s0/M:.4e} M_Pl; skipping 2R node check.", color="yellow")
            continue

        n_nodes = node_count_to_2R(sol, R_star_m, idx_sigma=2)
        per_mode.setdefault(n_nodes, []).append(dict(
            s0=s0, sol=sol, mu2_log=mu2_log, R_star_m=R_star_m
        ))

    # --- 3) Select desired modes (e.g., n=0,1): choose smallest |σ0| if multiple ---
    modes = (0, 1)
    sigma0_by_mode = {}
    for n in modes:
        cands = per_mode.get(n, [])
        if not cands:
            custom_print(f"[MISS] No candidate with n={n} nodes.", color="red")
            continue
        pick = min(cands, key=lambda d: abs(d["s0"]))
        sigma0_by_mode[n] = pick["s0"]
        custom_print(f"[OK] n={n}: σ₀ = {pick['s0']/M:.4e} M_Pl", color="cyan")

    # --- 4) Final integrate (again) for selected modes for plotting + diagnostics ---
    plot_entries = []
    for n, s0 in tqdm(list(sigma0_by_mode.items()), desc="\nFinal integrations", unit="mode"):
        sol, mu2_log, R_star_m = integrate_star(
            s0, p_eqState, rho_eqState, r0, r_max, stop_at_2r=True, record_mu2=True
        )
        r_star_km = (R_star_m/1e3) if (R_star_m is not None) else np.nan

        # Background leg for ADM mass & scalar charge
        sol_bnd, _, _ = integrate_star(s0, p_eqState, rho_eqState, r0, r_max, stop_at_2r=False, record_mu2=False)

        def adm_mass_from_solution(sol, k_tail=15):
            r   = sol.t
            Psi = sol.y[1]
            r_tail   = r[-k_tail:]
            Psi_tail = Psi[-k_tail:]
            dPsi_tail = np.gradient(Psi_tail, r_tail)
            M_vals = - (r_tail**2) * dPsi_tail * np.exp(-2.0*Psi_tail)
            M_len  = np.median(M_vals)
            M_kg   = M_len * (c*c) / G_N
            return M_kg

        ADM_mass = adm_mass_from_solution(sol_bnd, k_tail=15)
        R_log = R_star_m if R_star_m is not None else r_max  # simple fallback
        if lmbda_val != 0.0:
            scalar_charge = -r_max**2 * np.sqrt(np.log(r_max/max(R_log, 1.0))) * sol_bnd.y[3][-1]/M * (c*c) / G_N
        else:
            scalar_charge = -r_max**2 * sol_bnd.y[3][-1]/M * (c*c) / G_N

        print("ADM mass / M_sun = ", f"{ADM_mass/SM:.2e}")
        print("Scalar charge / M_sun = ", f"{scalar_charge/SM:.2e}")
        print("Q/M ratio = ", f"{scalar_charge/ADM_mass:.2e}")

        r_mu, mu2 = (np.array(mu2_log, dtype=float).T if len(mu2_log) else (np.array([]), np.array([])))
        plot_entries.append({
            "label": f"n={n}",
            "sol": sol,
            "r_star": r_star_km,
            "r_mu": r_mu,
            "mu2": mu2
        })

    # --- 5) Plots
    printResults_multi(plot_entries, xi_val)