from scipy.integrate import solve_ivp
import numpy as np
from Utils.TOV import initial_conditions, make_tov, StoppingConditions
from Utils.params import c, G_N
from Utils.graphics_single import custom_print

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


def adm_mass_from_sol(sol, tail):
    r = sol.t
    Psi = sol.y[1]
    mask = (r >= tail[0]) & (r <= tail[1])
    r_tail = r[mask]
    Psi_tail = Psi[mask]
    dPsi_tail = np.gradient(Psi_tail, r_tail)
    M_vals = -(r_tail**2) * dPsi_tail * np.exp(-2.0 * Psi_tail)
    M_len = np.median(M_vals)
    M_kg = M_len * (c * c) / G_N
    return M_kg


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


def tail_quartic(n, lam, r_tail, sigma_tail):
    from scipy.optimize import curve_fit

    def sigma_model(r, Q, lnr_bar, lam):
        """
        sigma(r) ~ Q / (r * sqrt(1 + 2*Q^2*lambda*ln(r/r_bar)))
        """
        return Q / (r * np.sqrt(1 + 2 * lam * Q**2 * (np.log(r) - lnr_bar)))

    # Auto-estimate initial guess from a simple 1/r Coulomb-like fit
    Q0 = np.median(sigma_tail * r_tail)  # rough charge estimate
    lnr_bar0 = np.log(r_tail[0])  # start of tail region
    g0 = [Q0, lnr_bar0]

    # Wrap model to fix lambda
    def model_fixed_lam(r, Q, lnr_bar):
        return sigma_model(r, Q, lnr_bar, lam)

    if n % 2 == 0:
        correct_bounds = ([0.0, 0.0], [np.inf, np.log(r_tail[0])])  # Q > 0 for even n
    else:
        correct_bounds = ([-np.inf, 0.0], [0.0, np.log(r_tail[0])])  # Q < 0 for odd n

    popt, pcov = curve_fit(
        model_fixed_lam,
        r_tail,
        sigma_tail,
        p0=g0,
        bounds=correct_bounds,
        maxfev=50_000,
    )

    Q_fit, lnr_bar_fit = popt
    return Q_fit, np.exp(lnr_bar_fit)  # return Q and r_bar


def scal_charge_from_sol(
    sol,
    n,
    m2,
    lmbda,
    nu,
    r_max,
    tail,
    *,
    idx_sigma=2,
    idx_sigmap=3,
    r_window_Q=("frac", 0.6, 0.95),
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

    # print(f"Total integration points: {len(sol.t)}")

    r = sol.t
    mask = (r >= tail[0]) & (r <= tail[1])
    r_tail = r[mask]
    sigma = sol.y[idx_sigma]
    sigma_tail = sigma[mask]
    sigmap = sol.y[idx_sigmap]
    sigmap_tail = sigmap[mask]

    # --- DEF: massless linear (Coulomb) ---
    if (lmbda == 0.0) and (nu == 0.0) and (m2 == 0.0):
        Q_vals = -(r_tail**2) * sigmap_tail
        Q_num = np.median(Q_vals)
        return Q_num / M * (c * c) / G_N, None  # convert to physical units

    # physical mass exponent (same units as 1/r)
    m_mass = float(np.sqrt(max(0.0, m2)))  # m2 is μ^2

    # --- MASSIVE (ν==0): Yukawa around σ∞=0 ---
    if (nu == 0.0) and (m_mass > 0.0):
        Q_num = np.average(
            -np.exp(m_mass * r_tail) * sigmap_tail * r_tail**2 / (1 + m_mass * r_tail)
        )
        #Q_num = np.average(-np.exp(m_mass * r_tail)) * sigma_tail*r_tail
        return Q_num / M * (c * c) / G_N, None  # convert to physical units

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
        return Q_num / M * (c * c) / G_N, None  # convert to physical units

    # --- PURE QUARTIC: ν==0, λ>0 → log-improved Coulomb ---
    r_bar = None
    if (nu == 0.0) and (lmbda > 0.0):
        Q_num, r_bar = tail_quartic(n, lmbda, r_tail, sigma_tail)

    return Q_num / M * (c * c) / G_N, r_bar  # convert to physical units


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
    n_int_points=1000,
):
    """Integrate one candidate σ₀ from r0 to r_max; optionally stop at 2R and log μ²."""
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
    ev_blow = events.blowup_guard()

    if stop_at_2r:
        ev_2R = events.double_radius()
        t_eval = np.linspace(r0, r_max, 1000)

        sol = solve_ivp(
            tov,
            r_span,
            y0,
            t_eval=t_eval,
            dense_output=True,
            method=method,
            rtol=1e-8,
            atol=1e-10,
            events=[ev_surface, ev_2R, ev_blow],
        )
    else:
        if n_int_points is not None:
            t_eval = np.logspace(np.log10(r0), np.log10(r_max), n_int_points)
            t_eval[0] = r0
            t_eval[-1] = r_max
        else:
            t_eval = None

        sol = solve_ivp(
            tov,
            r_span,
            y0,
            t_eval=t_eval,
            dense_output=True,
            method=method,
            rtol=1e-6,
            atol=1e-8,
            events=[ev_blow],
        )

    R_star_m = events.R_star  # set after solve
    return sol, mu2_log, R_star_m
