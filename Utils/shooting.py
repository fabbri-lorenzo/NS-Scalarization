# Utils/shooting_EMG.py

import numpy as np
from scipy.optimize import fsolve
from Utils.TOV_EMG import psi2_center
from Utils.params import c, M, lmbda_EMG as LAMBDA_DEFAULT

def _sigma_rmax(s0, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma=2):
    sol, _, _ = integrate_fn(
        s0, p_eqState, rho_eqState, r0, r_max, xi,
        stop_at_2r=False, record_mu2=False
    )
    return float(sol.y[idx_sigma, -1])

def _residual(s0, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma=2):
    try:
        val = _sigma_rmax(s0, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma)
        if not np.isfinite(val):
            return 1e300
        return val
    except Exception:
        return 1e300

def _unique_sorted(vals, tol=1e-8):
    if not vals:
        return []
    vals = sorted(vals)
    out = [vals[0]]
    for v in vals[1:]:
        if abs(v - out[-1]) > tol * max(1.0, abs(v), abs(out[-1])):
            out.append(v)
    return out

def _x_to_s(x, a, b):
    # maps x∈ℝ → s∈(a,b) using tanh (no exp overflow)
    t = np.tanh(x)
    return 0.5*(a + b) + 0.5*(b - a)*t

def _s_to_x(s, a, b):
    # inverse map s∈(a,b) → x∈ℝ via atanh; clip to avoid ±1
    t = (2.0*(s - 0.5*(a + b))) / (b - a)
    t = np.clip(t, -1.0 + 1e-15, 1.0 - 1e-15)
    return np.arctanh(t)

def shoot_sigma0(
    integrate_fn,
    p_eqState,
    rho_eqState,
    r0,
    r_max,
    xi,
    bracket,                 # (a, b) in SAME UNITS as σ0
    rho0,                    # for Ψ2 check
    lmbda=LAMBDA_DEFAULT,
    abs_threshold=1e-10,
    tol_relative=1e-2,
    n_seeds=41,
    xtol=1e-12,
    maxfev=400,
    idx_sigma=2,
    merge_tol=1e-8
):
    a, b = map(float, bracket)
    if not (np.isfinite(a) and np.isfinite(b) and a < b):
        raise ValueError("Invalid bracket")

    p0   = float(p_eqState(rho0))
    eps0 = float(rho0 * c * c)

    # uniform seeds in s-space
    S_uniform = np.linspace(a, b, int(n_seeds))
    
    # evaluate residuals once (unscaled) for diagnostics + scaling
    F_uniform = []
    for s0 in S_uniform:
        F_uniform.append(_residual(float(s0), integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma))
    F_uniform = np.array(F_uniform, float)
    
    # pick a sane scale: median finite |f| (fallback to bracket size)
    finite = np.isfinite(F_uniform)
    if np.any(finite):
        sigma_scale = max(1.0, np.median(np.abs(F_uniform[finite])))
    else:
        sigma_scale = max(1.0, abs(b - a))
    
    def f_arr_x(x_vec):
        s = _x_to_s(float(x_vec[0]), a, b)
        return np.array([ _residual(s, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma) / sigma_scale ])
         
    # augment seeds with k best coarse minima of |f|
    k_extra = min(8, len(S_uniform)//4)
    idx_k = np.argsort(np.where(finite, np.abs(F_uniform), np.inf))[:k_extra]
    S_seeds = np.unique(np.concatenate([S_uniform, S_uniform[idx_k]]))

    # run fsolve in x-space (bounded)
    roots = []
    for s0 in S_seeds:
        try:
            x0 = _s_to_x(float(s0), a, b)
            root, info, ier, _ = fsolve(f_arr_x, x0=[x0], full_output=True, xtol=xtol, maxfev=maxfev)
            x_star = float(root[0])
            s_star = _x_to_s(x_star, a, b)  # guaranteed in [a,b]
    
            fr = _residual(s_star, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma)
            if not np.isfinite(fr):
                continue
    
            abs_ok = (abs(fr) <= abs_threshold)
            rel_ok = (s_star != 0.0 and abs(fr / s_star) <= tol_relative)
    
            # accept if converged OR meets absolute OR relative criterion
            if ier == 1 or abs_ok or rel_ok:
                roots.append(s_star)
        except Exception:
            continue


    roots = _unique_sorted(roots, tol=merge_tol)
    #print(roots, '\n')# Debug

    # Evaluate residuals once for acceptance
    evals = []
    for r in roots:
        fr = _residual(r, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma)
        evals.append((r, fr))

    # Absolute-first set
    abs_ok = [r for (r, fr) in evals if abs(fr) <= abs_threshold]

    # If no absolute hits, relax to relative
    chosen = abs_ok
    if not chosen:
        rel_ok = [r for (r, fr) in evals if (r != 0.0 and abs(fr / r) <= tol_relative)]
        chosen = rel_ok

    # Ψ2>0 post-check
    final = []
    for r in chosen:
        psi2 = psi2_center(r, p0, eps0, xi, lmbda)
        if np.isfinite(psi2) and (psi2 > 0.0):
            final.append(r)

    return _unique_sorted(final, tol=merge_tol)