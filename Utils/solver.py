from scipy.integrate import solve_ivp
import numpy as np
from Utils.TOV_EMG import initial_conditions, make_tov_EMG, StoppingConditions  
from Utils.params import c, G_N

# ---------- helpers ----------
def count_nodes_sigma(sigma):
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
    return count_nodes_sigma(sigma)

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


# Integrator wrapper (returns OdeResult, μ² log, and R_* in meters)
def integrate_star(sigma0, p_eqState, rho_eqState, r0, r_max, xi, rho0, frac_pc, stop_at_2r=True, record_mu2=False):
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

