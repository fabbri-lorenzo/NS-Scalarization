import math
import numpy as np
from itertools import pairwise
from scipy.integrate import solve_ivp
from Utils.TOV_EMG import make_tov_system, initial_conditions, StoppingConditions

def integrate_sigma0(sigma0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc):
    r_span = (r0, r_max)
    y0 = initial_conditions(r0, sigma0, p_eqState)
    p_c = y0[0]

    tov = make_tov_system(p_c,frac_pc, rho_eqState, xi)
    sol = solve_ivp(
        tov, r_span, y0,
        method='RK45',
        rtol=1e-6, atol=1e-9,
    )
    
    return sol


# -------------------------
# Endpoint extraction / mismatch
# -------------------------
def _sigma_at_rmax(sol, idx_sigma=3):
    """
    Extract sigma(r_max) from a SciPy OdeResult.
    Adjust idx_sigma if your state ordering differs.
    """
    if not hasattr(sol, "y"):
        raise TypeError("integrate_sigma0 must return a SciPy OdeResult (with .y).")
    return float(sol.y[idx_sigma, -1])

def endpoint_mismatch(sol, target=0.0, extractor=_sigma_at_rmax, **extractor_kwargs):
    """
    Compute f = sigma(r_max) - target (or any boundary residual you prefer).
    Use 'extractor' to pull the endpoint value out of 'sol'.
    """
    val = extractor(sol, **extractor_kwargs)
    f = val - target
    # Guard against NaNs/inf from bad integrations
    if not math.isfinite(f):
        return float("inf")
    return f

# -------------------------
# Residual wrapper
# -------------------------
def residual(sigma0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc, target=0.0, extractor=_sigma_at_rmax,
             extractor_kwargs=None):
    """
    Map sigma0 -> mismatch f(sigma0). Forwards all args to your integrator.
    """
    extractor_kwargs = extractor_kwargs or {}
    sol = integrate_sigma0(sigma0, p_eqState,rho_eqState,xi, r0, r_max, frac_pc)
    return endpoint_mismatch(sol, target=target, extractor=extractor, **extractor_kwargs)

# -------------------------
# Helpers for non‑monotonic f(sigma0): scan, bracket, solve
# -------------------------
def _safe_residual(s0, *args, **kwargs):
    try:
        r = residual(s0, *args, **kwargs)
        return r if math.isfinite(r) else float("inf")
    except Exception:
        return float("inf")

def scan_brackets(s0_min, s0_max, n_samples, *resid_args, **resid_kwargs):
    """
    Coarse sweep over sigma0 to find sign-change brackets and 'near-zero' dips.
    Returns (brackets, dips).
    """
    grid = [s0_min + i * (s0_max - s0_min) / (n_samples - 1) for i in range(n_samples)]
    vals = [_safe_residual(s0, *resid_args, **resid_kwargs) for s0 in grid]

    brackets = []
    for (x0, x1), (f0, f1) in pairwise(zip(grid, vals)):
        if math.isfinite(f0) and math.isfinite(f1) and f0 * f1 < 0.0:
            brackets.append((x0, x1))

    # near-zero dips (handles tangential roots where sign doesn't flip)
    dips = []
    
    vals_finite = [v for v in vals if math.isfinite(v)]
    if not vals_finite:
        raise RuntimeError(
        "scan_brackets: no finite residuals. Check ODE setup, make_tov_system "
        "signature/args, EOS domain (p>0), r0, and that integrate_sigma0 returns sol.success=True."
    )
    thr = max(1e-6, 1e-3 * max(1.0, max(abs(v) for v in vals_finite)))
    
    for i in range(1, n_samples - 1):
        f = vals[i]
        if math.isfinite(f) and abs(f) < thr and f <= vals[i - 1] and f <= vals[i + 1]:
            dips.append(grid[i])

    return brackets, dips

def refine_near_zero(s0, width, *resid_args, **resid_kwargs):
    left, right = s0 - width, s0 + width
    fL = _safe_residual(left, *resid_args, **resid_kwargs)
    fR = _safe_residual(right, *resid_args, **resid_kwargs)
    return (left, right) if math.isfinite(fL) and math.isfinite(fR) and fL * fR < 0.0 else None


# -------------------------
# Robust hybrid root finder inside a bracket
# -------------------------

def shoot_sigma0(bracket=None, p_eqState=None, rho_eqState=None, xi=None,
                 r0=None, r_max=None, frac_pc=1e-15, target=0.0, tol=1e-8, max_iter=100,
                 extractor=_sigma_at_rmax, extractor_kwargs=None,
                 use_secant_first=True, scan_if_needed=True,
                 scan_range=(-1e-2, 1e-2), n_samples=401, return_all=False):
    """
    Robust root finder for non‑monotonic residuals.
    - If 'bracket' is valid and straddles a root: secant-in-bracket + bisection fallback.
    - Else (or bracket=None) and scan_if_needed=True: coarse scan via scan_brackets/dips,
      then solve each bracket and pick the best root.
    Returns:
      (sigma0_root, f_val, iters) by default; if return_all=True also returns a list of
      per-root dicts like find_all_roots().
    """
    extractor_kwargs = extractor_kwargs or {}

    def f(s0):
        return residual(s0, p_eqState, rho_eqState, xi, r0, r_max,frac_pc, target=target,
                        extractor=extractor, extractor_kwargs=extractor_kwargs)

    def solve_bracket(a, b):
        fa, fb = f(a), f(b)
        if not (math.isfinite(fa) and math.isfinite(fb)):
            raise RuntimeError("Non-finite residual at bracket ends.")
        if fa == 0.0:
            return a, 0.0, 0
        if fb == 0.0:
            return b, 0.0, 0
        if fa * fb > 0.0:
            raise ValueError("Bracket does not straddle a root (f(a)*f(b)>0).")

        A, B, fA, fB = a, b, fa, fb
        for it in range(1, max_iter + 1):
            # Secant step confined to [A,B]
            if use_secant_first and (fB - fA) != 0.0:
                s = B - fB * (B - A) / (fB - fA)
                if not (min(A, B) < s < max(A, B)) or not math.isfinite(s):
                    s = 0.5 * (A + B)
            else:
                s = 0.5 * (A + B)

            fs = f(s)
            if not math.isfinite(fs):
                s = 0.5 * (A + B)
                fs = f(s)

            if abs(fs) <= tol:
                return s, fs, it
            if abs(B - A) <= tol * max(1.0, abs(s)):
                return s, fs, it

            if fA * fs < 0.0:
                B, fB = s, fs
            else:
                A, fA = s, fs

        s_mid = 0.5 * (A + B)
        return s_mid, f(s_mid), max_iter

    # Path 1: Try provided bracket (if any) first
    if bracket is not None:
        a, b = bracket
        try:
            root, fval, iters = solve_bracket(a, b)
            return (root, fval, iters)
        except Exception:
            if not scan_if_needed:
                raise

    # Path 2: Scan + solve all brackets (handles non‑monotonicity / multiple roots)
    s0_min, s0_max = scan_range
    resid_kwargs = dict(p_eqState=p_eqState, r0=r0, r_max=r_max,frac_pc=frac_pc, target=target,
                        extractor=extractor, extractor_kwargs=extractor_kwargs)
    brackets, dips = scan_brackets(s0_min, s0_max, n_samples, rho_eqState=rho_eqState, xi=xi, **resid_kwargs)

    # Upgrade dips into brackets where possible
    width = 0.1 * (s0_max - s0_min) / n_samples
    for s0 in dips:
        b = refine_near_zero(s0, width, rho_eqState=rho_eqState, xi=xi, **resid_kwargs)
        if b is not None:
            brackets.append(b)

    # Deduplicate/sort
    uniq = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in brackets):
        if not uniq or (a - uniq[-1][1]) > 1e-12 * (1 + abs(a)):
            uniq.append((a, b))

    roots = []
    for a, b in uniq:
        try:
            r, fv, it = solve_bracket(a, b)
            if math.isfinite(r) and math.isfinite(fv):
                roots.append(dict(sigma0=r, f=fv, iters=it))
        except Exception:
            continue

    # Also accept exact near-zeros we didn’t bracket (rare)
    for s0 in dips:
        try:
            fv = residual(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc, target=target,
                          extractor=extractor, extractor_kwargs=extractor_kwargs)
            if math.isfinite(fv) and abs(fv) <= tol:
                roots.append(dict(sigma0=s0, f=fv, iters=0))
        except Exception:
            pass

    if not roots:
        raise RuntimeError("No roots found in scan range; widen scan_range or n_samples.")

    # Pick the “best” root = min |f| (tie-breaker: smallest |sigma0|)
    best = min(roots, key=lambda d: (abs(d['f']), abs(d['sigma0'])))
    return (best['sigma0'], best['f'], best['iters']) if not return_all else ((best['sigma0'], best['f'], best['iters']), roots)

