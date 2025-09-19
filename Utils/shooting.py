import numpy as np
from scipy.optimize import fsolve

from Utils.TOV_EMG import psi2_center
from Utils.params import c, M, lmbda_EMG as LAMBDA_DEFAULT


def _sigma_rmax(s0, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma=2):
    """
    Integrate the system with central scalar amplitude ``s0`` and return the value
    of the scalar field at the outer boundary ``r_max``.  The index ``idx_sigma``
    selects which component of the solution vector corresponds to the scalar
    field.  If the integration fails or returns non‑finite values, a large
    number is returned so that the root‐finding routines can ignore it.
    """
    sol, _, _ = integrate_fn(
        s0, p_eqState, rho_eqState, r0, r_max, xi,
        stop_at_2r=False, record_mu2=False
    )
    return float(sol.y[idx_sigma, -1])


def _residual(s0, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma=2):
    """Wrapper around ``_sigma_rmax`` that guards against exceptions and
    non‑finite results.  Returns a large value on failure.
    """
    try:
        val = _sigma_rmax(s0, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma)
        if not np.isfinite(val):
            return 1e300
        return val
    except Exception:
        return 1e300


def _unique_sorted(vals, tol=1e-8):
    """Given a list of floats, return a sorted list with near‑duplicates merged."""
    if not vals:
        return []
    vals = sorted(vals)
    out = [vals[0]]
    for v in vals[1:]:
        if abs(v - out[-1]) > tol * max(1.0, abs(v), abs(out[-1])):
            out.append(v)
    return out


def _x_to_s(x, a, b):
    """Map x∈ℝ to s∈(a,b) using tanh.  This avoids exponent overflow."""
    t = np.tanh(x)
    return 0.5 * (a + b) + 0.5 * (b - a) * t


def _s_to_x(s, a, b):
    """Inverse map s∈(a,b) to x∈ℝ via atanh; clip the argument to avoid ±1."""
    t = (2.0 * (s - 0.5 * (a + b))) / (b - a)
    t = np.clip(t, -1.0 + 1e-15, 1.0 - 1e-15)
    return np.arctanh(t)


def shoot_sigma0(
    integrate_fn,
    p_eqState,
    rho_eqState,
    r0,
    r_max,
    xi,
    bracket,  # (a, b) in SAME UNITS as σ0
    rho0,  # for Ψ2 check
    lmbda=LAMBDA_DEFAULT,
    abs_threshold=1e-10,
    tol_relative=1e-2,
    n_seeds=41,
    sigma0_min_abs=0.0,
    xtol=1e-12,
    maxfev=400,
    idx_sigma=2,
    merge_tol=1e-8,
):
    """
    Shoot for central scalar amplitudes ``σ0`` such that the scalar field at
    ``r_max`` vanishes.  A list of acceptable roots is returned.  Roots are
    accepted if their residual is below ``abs_threshold``.  If none meet that
    criterion, a relaxed relative threshold ``tol_relative`` is used instead.
    Only roots for which ``psi2_center`` is positive are returned.
    """
    a, b = map(float, bracket)
    if not (np.isfinite(a) and np.isfinite(b) and a < b):
        raise ValueError("Invalid bracket")

    p0 = float(p_eqState(rho0))
    eps0 = float(rho0 * c * c)

    # uniform seeds in s‐space for scaling and fallback
    S_uniform = np.linspace(a, b, int(n_seeds))
    F_uniform = []
    for s0 in S_uniform:
        F_uniform.append(_residual(float(s0), integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma))
    F_uniform = np.array(F_uniform, float)

    # choose a scale for the root finder based on median finite residual
    finite = np.isfinite(F_uniform)
    if np.any(finite):
        sigma_scale = max(1.0, np.median(np.abs(F_uniform[finite])))
    else:
        sigma_scale = max(1.0, abs(b - a))

    def f_arr_x(x_vec):
        s = _x_to_s(float(x_vec[0]), a, b)
        return np.array(
            [
                _residual(
                    s, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma
                )
                / sigma_scale
            ]
        )

    roots = []
    # Use fsolve on a set of seeds to locate roots.
    k_extra = min(8, len(S_uniform) // 4)
    idx_k = np.argsort(np.where(finite, np.abs(F_uniform), np.inf))[:k_extra]
    S_seeds = np.unique(np.concatenate([S_uniform, S_uniform[idx_k]]))
    roots = []
    for s0 in S_seeds:
        try:
            x0 = _s_to_x(float(s0), a, b)
            x_star, info, ier, _ = fsolve(
                f_arr_x, x0=[x0], full_output=True, xtol=xtol, maxfev=maxfev
            )
            s_star = _x_to_s(float(x_star[0]), a, b)
            fr = _residual(s_star, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma)
            if not np.isfinite(fr):
                continue
            abs_ok = (abs(fr) <= abs_threshold)
            rel_ok = s_star != 0.0 and abs(fr / s_star) <= tol_relative
            if ier == 1 or abs_ok or rel_ok:
                roots.append(s_star)
        except Exception:
            continue

    # filter out values outside the bracket or below the minimal |σ0|
    roots = [r for r in roots if (a <= r <= b) and (abs(r) >= sigma0_min_abs)]
    roots = _unique_sorted(roots, tol=merge_tol)

    # evaluate residuals for acceptance; apply absolute then relative threshold
    evals = []
    for r in roots:
        fr = _residual(r, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma)
        evals.append((r, fr))
    # absolute cut
    abs_ok_roots = [r for (r, fr) in evals if abs(fr) <= abs_threshold]
    chosen = abs_ok_roots
    if not chosen:
        # relaxed relative cut
        rel_ok_roots = [
            r for (r, fr) in evals if (r != 0.0 and abs(fr / r) <= tol_relative)
        ]
        chosen = rel_ok_roots

    # Ψ2>0 post‑check
    final = []
    for r in chosen:
        psi2 = psi2_center(r, p0, eps0, xi, lmbda)
        if np.isfinite(psi2) and (psi2 > 0.0):
            final.append(r)

    return _unique_sorted(final, tol=merge_tol)
