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


def adm_mass_from_sol(sol, k_tail=15):
    r = sol.t
    Psi = sol.y[1]
    r_tail = r[-k_tail:]
    Psi_tail = Psi[-k_tail:]
    dPsi_tail = np.gradient(Psi_tail, r_tail)
    M_vals = -(r_tail**2) * dPsi_tail * np.exp(-2.0 * Psi_tail)
    M_len = np.median(M_vals)
    M_kg = M_len * (c * c) / G_N
    return M_kg


def tail_charge_with_Rlog(sol, R_log, *, idx_sigmap=3, r_window=("frac", 0.5, 0.95)):
    """
    Given R_log, estimate Q from the tail using
        Q(r) = - sigma'(r) * r^2 * sqrt(ln(r / R_log))
    and return the median over the chosen tail window.

    Arguments:
    ----------
    sol : OdeSolution (from solve_ivp)
        Full solution object containing t (r grid) and y (state vars).
    R_log : float
        The fitted R_log constant used inside the sqrt log term.
    idx_sigmap : int
        Index of sigma' inside sol.y (default 3).
    r_window : tuple
        ('frac', f1, f2) → use fraction of the integration domain.
        (r1, r2) → use explicit physical radii.
    """
    import numpy as np

    r = np.asarray(sol.t)
    sp = np.asarray(sol.y[idx_sigmap], dtype=float)

    # choose tail region
    if isinstance(r_window, tuple) and len(r_window) == 3 and r_window[0] == "frac":
        _, f1, f2 = r_window
        r1 = r.min() + f1 * (r.max() - r.min())
        r2 = r.min() + f2 * (r.max() - r.min())
    else:
        r1, r2 = r_window

    mask = (r >= r1) & (r <= r2) & np.isfinite(sp)
    r_tail = r[mask]
    sp_tail = sp[mask]

    if r_tail.size < 10:
        # fallback: last 20% of the domain
        mask = (r >= r.min() + 0.8 * (r.max() - r.min())) & np.isfinite(sp)
        r_tail = r[mask]
        sp_tail = sp[mask]

    # compute Q(r) samples
    L = np.log(np.maximum(r_tail, 1.0)) - np.log(max(R_log, 1.0))
    L = np.maximum(L, 1e-12)  # avoid sqrt(0)
    Q_samples = -sp_tail * (r_tail**2) * np.sqrt(L)

    return float(np.median(Q_samples))


# Robust tail fit for (Q, R_log) in the λ≠0 case
def fit_tail_Rlog_and_Q(
    sol,
    *,
    idx_sigma=2,
    idx_sigmap=3,
    r_window=(
        1.0e4,
        1.0e5,
    ),  # you can also pass ('frac', 0.2, 0.6) to use a fraction of r_max
    min_points=40
):
    """
    Fit the far-field tail to sigma'(r) ≈ -Q / (r^2 * sqrt(ln(r/R_log))).
    Returns (Q_fit, R_log_fit) in code units (Q in your native units for sigma).

    Notes:
    - Enforces R_log < min(r_window) to keep the log positive across the fit window.
    - Uses only points with finite, non-zero sigma'.
    - If the explicit numeric window has too few points, falls back to the last 50% of the domain.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    r = np.asarray(sol.t)
    # default indices: y[idx_sigma] = sigma, y[idx_sigmap] = sigma'
    sigma_p = np.asarray(sol.y[idx_sigmap], dtype=float)

    # Allow fractional window definition
    if isinstance(r_window, tuple) and len(r_window) == 3 and r_window[0] == "frac":
        _, f1, f2 = r_window
        r1 = r.min() + f1 * (r.max() - r.min())
        r2 = r.min() + f2 * (r.max() - r.min())
    else:
        r1, r2 = r_window

    mask = (r >= r1) & (r <= r2) & np.isfinite(sigma_p) & (sigma_p != 0.0)
    r_fit = r[mask]
    sp_fit = sigma_p[mask]

    # Fallback: use last 50% if we don't have enough samples
    if r_fit.size < min_points:
        r1_fb = r.min() + 0.5 * (r.max() - r.min())
        mask_fb = (r >= r1_fb) & np.isfinite(sigma_p) & (sigma_p != 0.0)
        r_fit = r[mask_fb]
        sp_fit = sigma_p[mask_fb]

    if r_fit.size < min_points:
        raise RuntimeError(
            "fit_tail_Rlog_and_Q: insufficient tail points for a stable fit"
        )

    # Model: sigma'(r) = -Q / (r^2 * sqrt(ln(r/R_log)))
    def model(rvals, Q, lnR):
        L = np.log(rvals) - lnR
        # numerical safety: clamp L away from 0 to avoid divide-by-zero during fitting
        L = np.maximum(L, 1e-12)
        return -Q / (rvals**2 * np.sqrt(L))

    # Initial guesses: Q ~ -sp * r^2 (ignoring the sqrt log); take median magnitude
    q0_samples = -sp_fit * (r_fit**2)
    q0 = np.median(q0_samples)
    lnR0 = np.log(max(r_fit.min() * 0.1, 1.0))

    # Constrain R_log < min(r_fit) so ln(r/R_log) stays positive in-window
    upper_lnR = np.log(r_fit.min() * 0.999)
    bounds = ([-np.inf, -np.inf], [np.inf, upper_lnR])

    popt, _ = curve_fit(
        model, r_fit, sp_fit, p0=(q0, lnR0), bounds=bounds, maxfev=20000
    )
    Q_fit, lnR_fit = popt
    R_log_fit = float(np.exp(lnR_fit))
    return float(Q_fit), R_log_fit


# Robust tail fit for (Q, R_log) in the λ≠0 case
def fit_tail_Rlog_and_Q(
    sol,
    *,
    idx_sigma=2,
    idx_sigmap=3,
    r_window=(
        1.0e4,
        1.0e5,
    ),  # you can also pass ('frac', 0.2, 0.6) to use a fraction of r_max
    min_points=40
):
    """
    Fit the far-field tail to sigma'(r) ≈ -Q / (r^2 * sqrt(ln(r/R_log))).
    Returns (Q_fit, R_log_fit) in code units (Q in your native units for sigma).

    Notes:
    - Enforces R_log < min(r_window) to keep the log positive across the fit window.
    - Uses only points with finite, non-zero sigma'.
    - If the explicit numeric window has too few points, falls back to the last 50% of the domain.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    r = np.asarray(sol.t)
    # default indices: y[idx_sigma] = sigma, y[idx_sigmap] = sigma'
    sigma_p = np.asarray(sol.y[idx_sigmap], dtype=float)

    # Allow fractional window definition
    if isinstance(r_window, tuple) and len(r_window) == 3 and r_window[0] == "frac":
        _, f1, f2 = r_window
        r1 = r.min() + f1 * (r.max() - r.min())
        r2 = r.min() + f2 * (r.max() - r.min())
    else:
        r1, r2 = r_window

    mask = (r >= r1) & (r <= r2) & np.isfinite(sigma_p) & (sigma_p != 0.0)
    r_fit = r[mask]
    sp_fit = sigma_p[mask]

    # Fallback: use last 50% if we don't have enough samples
    if r_fit.size < min_points:
        r1_fb = r.min() + 0.5 * (r.max() - r.min())
        mask_fb = (r >= r1_fb) & np.isfinite(sigma_p) & (sigma_p != 0.0)
        r_fit = r[mask_fb]
        sp_fit = sigma_p[mask_fb]

    if r_fit.size < min_points:
        raise RuntimeError(
            "fit_tail_Rlog_and_Q: insufficient tail points for a stable fit"
        )

    # Model: sigma'(r) = -Q / (r^2 * sqrt(ln(r/R_log)))
    def model(rvals, Q, lnR):
        L = np.log(rvals) - lnR
        # numerical safety: clamp L away from 0 to avoid divide-by-zero during fitting
        L = np.maximum(L, 1e-12)
        return -Q / (rvals**2 * np.sqrt(L))

    # Initial guesses: Q ~ -sp * r^2 (ignoring the sqrt log); take median magnitude
    q0_samples = -sp_fit * (r_fit**2)
    q0 = np.median(q0_samples)
    lnR0 = np.log(max(r_fit.min() * 0.1, 1.0))

    # Constrain R_log < min(r_fit) so ln(r/R_log) stays positive in-window
    upper_lnR = np.log(r_fit.min() * 0.999)
    bounds = ([-np.inf, -np.inf], [np.inf, upper_lnR])

    popt, _ = curve_fit(
        model, r_fit, sp_fit, p0=(q0, lnR0), bounds=bounds, maxfev=20000
    )
    Q_fit, lnR_fit = popt
    R_log_fit = float(np.exp(lnR_fit))
    return float(Q_fit), R_log_fit


# Scal_charge_from_sol now auto-fits R_log when λ ≠ 0 and tail-averages Q
def scal_charge_from_sol(
    sol,
    lmbda,
    r_max,
    *,
    idx_sigma=2,
    idx_sigmap=3,
    r_window_Q=("frac", 0.6, 0.95),  # window to aggregate Q after R_log is known
    R_log_override=None  # if you already know R_log, pass it to skip the fit
):
    """
    Compute scalar charge Q with correct λ-dependent asymptotics.

    - For λ == 0: uses your original 1/r^2 tail: Q = - r_max^2 * sigma'(r_max)
    - For λ != 0:
        * If R_log_override is None: fit (Q, R_log) on the tail from sigma'(r).
        * Then re-estimate Q by robust tail-averaging with the fitted R_log (reduces noise).
    Returns Q in code units, then converts to your physical units via (c^2/G_N)/M, matching your convention.
    """
    import numpy as np
    from Utils.params import M, c, G_N  # keep your existing import style

    if lmbda == 0.0:
        Q_num = -(r_max**2) * float(sol.y[idx_sigmap][-1])
        Q = Q_num / M * (c * c) / G_N
        return Q

    # λ ≠ 0: determine R_log (fit if not provided), then compute robust Q

    if R_log_override is None:
        r_window_fit = (r_max * 0.5, r_max)
        Q_fit, R_log = fit_tail_Rlog_and_Q(
            sol, idx_sigma=idx_sigma, idx_sigmap=idx_sigmap, r_window=r_window_fit
        )
    else:
        R_log = float(R_log_override)
        Q_fit = None  # not used further

    # Robust tail-aggregated Q using the chosen/fitted R_log
    Q_num = tail_charge_with_Rlog(
        sol, R_log, idx_sigmap=idx_sigmap, r_window=r_window_Q
    )

    # Convert to your physical normalization
    Q = Q_num / M * (c * c) / G_N
    return Q


# Integrator wrapper (returns OdeResult, μ² log, and R_* in meters)
def integrate_star(
    sigma0,
    p_eqState,
    rho_eqState,
    r0,
    r_max,
    xi,
    lmbda,
    rho0,
    frac_pc,
    stop_at_2r=True,
    record_mu2=False,
):
    r_span = (r0, r_max)
    y0 = initial_conditions(r0, sigma0, p_eqState, xi, rho0, lmbda)

    # If the center expansion is invalid (NaNs), return a minimal stub result.
    if not np.all(np.isfinite(y0)):

        class _Stub:
            pass

        sol = _Stub()
        sol.t = np.array([r0], dtype=float)
        sol.y = np.array(y0, dtype=float).reshape(4, 1)
        R_star_m = None
        mu2_log = []
        return sol, mu2_log, R_star_m

    p_c = y0[0]

    mu2_log = []
    if record_mu2:
        tov = make_tov_EMG(
            p_c, frac_pc, rho_eqState, xi, lmbda, mu2_recorder=mu2_log.append
        )
    else:
        tov = make_tov_EMG(p_c, frac_pc, rho_eqState, xi, lmbda)

    events = StoppingConditions(p_c, frac_pc)
    ev_surface = events.pressure_limit()
    ev_blow = events.blowup_guard(xi)

    method = "BDF" if xi < 0.0 else "RK45"

    if stop_at_2r:
        ev_2R = events.double_radius()
        sol = solve_ivp(
            tov,
            r_span,
            y0,
            dense_output=True,
            method=method,
            rtol=1e-6,
            atol=1e-9,
            events=[ev_surface, ev_2R, ev_blow],
        )
    else:
        sol = solve_ivp(
            tov,
            r_span,
            y0,
            dense_output=True,
            method=method,
            rtol=1e-6,
            atol=1e-9,
            events=[ev_blow],
        )

    R_star_m = events.R_star  # set after solve
    return sol, mu2_log, R_star_m
