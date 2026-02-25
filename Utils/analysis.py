from scipy.integrate import solve_ivp
import numpy as np
from Utils.TOV import initial_conditions, make_tov, StoppingConditions
from Utils.params import c, G_N


# ---------- helpers ----------
def count_nodes_sigma(sigma, target, tol=None, hysteresis=3.0):
    # subtract the vacuum baseline
    s = np.asarray(sigma, dtype=float) - float(target)
    # choose a scale‑aware tolerance if the caller didn’t specify one
    A = np.nanmax(np.abs(s)) if s.size else 0.0
    if tol is None:
        tol = 1e-12 if not np.isfinite(A) else max(1e-12, 1e-6 * A)
    # drop NaNs and points close to zero
    m = np.isfinite(s) & (np.abs(s) > tol)
    s = s[m]
    if s.size < 2:
        return 0
    # compress consecutive sign blocks
    sign = np.sign(s)
    keep = np.concatenate(([True], sign[1:] != sign[:-1]))
    s = s[keep]
    sign = sign[keep]
    # apply a hysteresis: only count a flip if the block spans more than hysteresis×tol
    flips, i0 = 0, 0
    for i1 in range(1, sign.size):
        if sign[i1] != sign[i0]:
            block = s[i0 : i1 + 1]
            if np.nanmax(block) - np.nanmin(block) > hysteresis * tol:
                flips += 1
            i0 = i1
    return flips


def node_count_to_2R(sol, R_star_m, target, idx_sigma=2):
    r = np.asarray(sol.t, dtype=float)
    sigma = np.asarray(sol.y[idx_sigma], dtype=float)
    if R_star_m is not None and np.isfinite(R_star_m):
        mask = r <= 2.0 * R_star_m
        if np.any(mask):
            sigma = sigma[mask]
    return count_nodes_sigma(sigma, target)


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
    L = np.log(r_tail) - np.log(R_log)
    Q_samples = -sp_tail * (r_tail**2) * np.sqrt(L)

    return float(np.median(Q_samples))


# utility to robustly sample the tail (uses dense_output if present)
def _sample_tail(sol, idx_sigmap=3, r_window=("frac", 0.5, 0.95), n_points=400):
    """
    Returns (r_tail, sigma_p_tail) sampled uniformly in r over the chosen window.
    If sol.sol is available (dense_output=True), we evaluate it; otherwise we downsample sol.t/sol.y.
    """
    import numpy as np

    r_all = np.asarray(sol.t, dtype=float)
    # choose window (fractional or absolute)
    if isinstance(r_window, tuple) and len(r_window) == 3 and r_window[0] == "frac":
        _, f1, f2 = r_window
        r1 = r_all.min() + f1 * (r_all.max() - r_all.min())
        r2 = r_all.min() + f2 * (r_all.max() - r_all.min())
    else:
        r1, r2 = r_window

    if r2 <= r1:
        r1, r2 = r1, max(r1 * (1.0 + 1e-6), r1 + 1e-6)

    # primary: dense_output
    r_tail = None
    sp_tail = None
    if hasattr(sol, "sol") and callable(sol.sol):
        r_tail = np.linspace(r1, r2, int(max(50, n_points)))
        y_tail = sol.sol(r_tail)  # shape (n_state, n_points)
        sp_tail = np.asarray(y_tail[idx_sigmap], dtype=float)
    else:
        # fallback: use raw points within window, then (optionally) interpolate
        mask = (r_all >= r1) & (r_all <= r2)
        r_raw = r_all[mask]
        sp_raw = np.asarray(sol.y[idx_sigmap], dtype=float)[mask]
        if r_raw.size >= 10:
            r_tail = r_raw
            sp_tail = sp_raw
        else:
            # last-ditch: take last chunk of domain
            mask = r_all >= (r_all.min() + 0.8 * (r_all.max() - r_all.min()))
            r_tail = r_all[mask]
            sp_tail = np.asarray(sol.y[idx_sigmap], dtype=float)[mask]

    # final sanitation
    finite = np.isfinite(sp_tail)
    r_tail = r_tail[finite]
    sp_tail = sp_tail[finite]
    return r_tail, sp_tail


# Robust tail fit for (Q, R_log) in the λ≠0 case
def fit_tail_Rlog_and_Q(
    sol,
    *,
    idx_sigma=2,
    idx_sigmap=3,
    r_window=("frac", 0.2, 0.6),  # use fractional by default to avoid empty windows
    min_points=40,
    n_points=600
):
    """
    Fit sigma'(r) ≈ -Q / (r^2 * sqrt(ln(r/R_log))) over the far-field tail.
    Uses dense_output to ensure enough points; includes multi-stage fallbacks.
    Returns (Q_fit, R_log_fit).
    """
    import numpy as np
    from scipy.optimize import curve_fit

    # 1) try the requested window
    r_fit, sp_fit = _sample_tail(
        sol, idx_sigmap=idx_sigmap, r_window=r_window, n_points=n_points
    )

    # 2) fallback if insufficient
    if r_fit.size < min_points:
        # use last 50% of domain
        r_fit, sp_fit = _sample_tail(
            sol, idx_sigmap=idx_sigmap, r_window=("frac", 0.5, 0.98), n_points=n_points
        )

    # 3) last fallback
    if r_fit.size < min_points:
        r_fit, sp_fit = _sample_tail(
            sol, idx_sigmap=idx_sigmap, r_window=("frac", 0.7, 0.995), n_points=n_points
        )

    if r_fit.size < min_points:
        raise RuntimeError(
            "fit_tail_Rlog_and_Q: insufficient tail points for a stable fit"
        )

    # model
    def model(rvals, Q, lnR):
        L = np.log(rvals) - lnR
        L = np.maximum(L, 1e-12)
        return -Q / (rvals**2 * np.sqrt(L))

    # initial guesses
    q0_samples = -sp_fit * (r_fit**2)
    q0 = float(np.median(q0_samples[np.isfinite(q0_samples)]))
    lnR0 = float(np.log(max(r_fit.min() * 0.2, 1.0)))

    # bounds to keep ln(r/R) > 0 in-window
    upper_lnR = float(np.log(r_fit.min() * 0.999))
    bounds = ([-np.inf, -np.inf], [np.inf, upper_lnR])

    popt, _ = curve_fit(
        model, r_fit, sp_fit, p0=(q0, lnR0), bounds=bounds, maxfev=50000
    )
    Q_fit, lnR_fit = popt
    return float(Q_fit), float(np.exp(lnR_fit))


# Scal_charge_from_sol now auto-fits R_log when λ ≠ 0 and tail-averages Q
def tail_charge_yukawa(
    sol, m, nu, *, idx_sigma=2, r_window=("frac", 0.6, 0.95), n_points=400
):
    """
    Compute Q for a Yukawa tail:
        σ(r) ≈ σ_∞ + (Q/r) * e^{-m r},  with  σ_∞ = ±|nu|.
    Robustly estimate Q over a tail window using:
        Q(r) = r * e^{m r} * (σ(r) - σ_∞),
    then return the median over the window.

    Notes:
    - Using σ (not σ′) avoids the asymptotic (1 + m r) blow-up in σ′.
    - σ_∞ sign is chosen by matching the tail to +|nu| or -|nu|.
    """
    import numpy as np

    # choose window (fractional or absolute) and sample, using dense_output when available
    r_all = np.asarray(sol.t, dtype=float)
    if isinstance(r_window, tuple) and len(r_window) == 3 and r_window[0] == "frac":
        _, f1, f2 = r_window
        r1 = r_all.min() + f1 * (r_all.max() - r_all.min())
        r2 = r_all.min() + f2 * (r_all.max() - r_all.min())
    else:
        r1, r2 = r_window
    if r2 <= r1:
        r2 = max(r1 * (1.0 + 1e-6), r1 + 1e-6)

    # sample tail
    if hasattr(sol, "sol") and callable(sol.sol):
        r_tail = np.linspace(r1, r2, int(max(50, n_points)))
        y_tail = sol.sol(r_tail)  # (n_state, n_points)
        sigma_tail = np.asarray(y_tail[idx_sigma], dtype=float)
    else:
        mask = (r_all >= r1) & (r_all <= r2)
        r_tail = r_all[mask]
        sigma_tail = np.asarray(sol.y[idx_sigma], dtype=float)[mask]
        if r_tail.size < 10:  # fallback: last 20% of domain
            mask = r_all >= (r_all.min() + 0.8 * (r_all.max() - r_all.min()))
            r_tail = r_all[mask]
            sigma_tail = np.asarray(sol.y[idx_sigma], dtype=float)[mask]

    finite = np.isfinite(sigma_tail)
    r_tail = r_tail[finite]
    sigma_tail = sigma_tail[finite]
    if r_tail.size < 10:
        raise RuntimeError("tail_charge_yukawa: insufficient tail points")

    # pick σ_∞ = ±|nu| by best match to the tail
    nu_abs = float(abs(nu))
    s_plus = nu_abs
    s_minus = -nu_abs
    # choose the sign minimizing the L2 mismatch in the window
    if np.mean((sigma_tail - s_plus) ** 2) <= np.mean((sigma_tail - s_minus) ** 2):
        sigma_inf = s_plus
    else:
        sigma_inf = s_minus

    # Q(r) samples: r * e^{m r} * (σ - σ_inf)
    Q_samples = r_tail * np.exp(m * r_tail) * (sigma_tail - sigma_inf)

    # be robust to small subleading 1/r corrections: median is stable
    return float(np.median(Q_samples))


def scal_charge_from_sol(
    sol,
    m2,
    lmbda,
    nu,
    r_max,
    *,
    idx_sigma=2,
    idx_sigmap=3,
    r_window_Q=("frac", 0.6, 0.95),
    R_log_override=None
):
    """
    Scalar charge with correct asymptotics by regime:
      - DEF (λ==0 and ν==0):       Q = - r_max^2 * σ'(r_max).
      - Pure quartic (λ>0, ν==0):  fit R_log (or use override) and tail-average Q_log(r) = - r^2 σ' sqrt(ln(r/R_log)).
      - Yukawa (ν!=0):             use σ-tail: Q(r) = r * e^{m r} * (σ - σ_∞), median over a tail window.

    Returns Q in physical normalization, matching previous convention.
    """
    import numpy as np
    from Utils.params import M, c, G_N

    # --- DEF: massless linear (Coulomb) ---
    if (lmbda == 0.0) and (nu == 0.0) and (m2 == 0.0):
        Q_num = -(r_max**2) * float(sol.y[idx_sigmap][-1])
        return Q_num / M * (c * c) / G_N

    # physical mass exponent (same units as 1/r)
    m_mass = float(np.sqrt(max(0.0, m2)))  # m2 is μ^2

    # --- MASSIVE (ν==0): Yukawa around σ∞=0 ---
    if (nu == 0.0) and (m_mass > 0.0):
        Q_num = float(sol.y[idx_sigma][-1]) * np.exp(m_mass * r_max) * r_max
        return Q_num / M * (c * c) / G_N

    # --- YUKAWA: m = sqrt(2 λ) |ν|, use σ-based estimator (stable) ---
    if nu != 0.0:
        m = float(np.sqrt(max(0.0, 2.0 * lmbda)) * abs(nu))
        if m == 0.0:
            # degenerate corner: falls back to DEF definition
            Q_num = -(r_max**2) * float(sol.y[idx_sigmap][-1])
        else:
            Q_num = tail_charge_yukawa(
                sol, m, nu, idx_sigma=idx_sigma, r_window=r_window_Q
            )
        return Q_num / M * (c * c) / G_N

    # --- PURE QUARTIC: ν==0, λ>0 → log-improved Coulomb ---
    # fit R_log if needed, then aggregate with your existing estimator
    if R_log_override is not None:
        R_log = float(R_log_override)
    else:
        # try several windows to avoid "insufficient tail points"
        tried = [
            ("frac", 0.2, 0.6),
            ("frac", 0.4, 0.9),
            ("frac", 0.6, 0.98),
        ]
        R_log = None
        last_err = None
        for win in tried:
            try:
                _, R_log = fit_tail_Rlog_and_Q(
                    sol,
                    idx_sigma=idx_sigma,
                    idx_sigmap=idx_sigmap,
                    r_window=win,
                    min_points=40,
                    n_points=800,
                )
                break
            except Exception as e:
                last_err = e
                R_log = None
        if R_log is None:
            # graceful fallback: keep ln(r/R)>0 on the tail
            r_all = np.asarray(sol.t, dtype=float)
            r_min_tail = r_all.min() + 0.6 * (r_all.max() - r_all.min())
            R_log = float(max(1.0, 0.2 * r_min_tail))

    # aggregate Q over tail for the quartic case
    Q_num = tail_charge_with_Rlog(
        sol, R_log, idx_sigmap=idx_sigmap, r_window=r_window_Q
    )
    return Q_num / M * (c * c) / G_N


# Integrator wrapper (returns OdeResult, μ² log, and R_* in meters)
def integrate_star(
    sigma0,
    p_eqState,
    rho_eqState,
    r0,
    r_max,
    xi,
    m2,
    lmbda,
    nu,
    rho0,
    frac_pc,
    method="RK45",
    stop_at_2r=True,
    record_mu2=False,
):
    r_span = (r0, r_max)

    y0 = initial_conditions(r0, sigma0, p_eqState, xi, m2, rho0, lmbda, nu)

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
        tov = make_tov(
            p_c, frac_pc, rho_eqState, xi, m2, lmbda, nu, mu2_recorder=mu2_log.append
        )
    else:
        tov = make_tov(p_c, frac_pc, rho_eqState, xi, m2, lmbda, nu)

    events = StoppingConditions(p_c, frac_pc)
    ev_surface = events.pressure_limit()
    ev_blow = events.blowup_guard(xi)

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
            # min_step=1e-5, #only avaible for LSODA
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
            # min_step=1e-5, #only avaible for LSODA
            events=[ev_blow],
        )

    R_star_m = events.R_star  # set after solve
    return sol, mu2_log, R_star_m
