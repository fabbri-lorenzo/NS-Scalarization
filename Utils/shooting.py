import math
import numpy as np
from itertools import pairwise
from Utils.TOV_EMG import psi2_center


# -------------------------
# Endpoint extraction / mismatch
# -------------------------
def _sigma_rmax(s0, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, idx_sigma=2):
    """
    Integrate the system with central scalar amplitude ``s0`` and return the value
    of the scalar field at the outer boundary ``r_max``.  The index ``idx_sigma``
    selects which component of the solution vector corresponds to the scalar
    field.  If the integration fails or returns non‑finite values, a large
    number is returned so that the root‐finding routines can ignore it.
    """
    sol, _, _ = integrate_fn(
        s0, p_eqState, rho_eqState, r0, r_max, xi, stop_at_2r=False, record_mu2=False
    )
    return float(sol.y[idx_sigma, -1])


# -------------------------
# Residual wrapper
# -------------------------
def residual(sigma0, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi, target=0.0):
    """
    Map sigma0 -> mismatch f(sigma0). Forwards all args to your integrator.
    """
    s0 = _sigma_rmax(sigma0, integrate_fn, p_eqState, rho_eqState, r0, r_max, xi)
    f = s0 - target
    # Guard against NaNs/inf from bad integrations
    if not math.isfinite(f):
        return float("inf")

    return f


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
    return (
        (left, right)
        if math.isfinite(fL) and math.isfinite(fR) and fL * fR < 0.0
        else None
    )


# -------------------------
# Robust hybrid root finder inside a bracket
# -------------------------


def shoot_sigma0(
    bracket=None,
    integrate_fn=None,
    p_eqState=None,
    rho_eqState=None,
    eps0=None,
    r0=None,
    r_max=None,
    xi=None,
    lmbda=None,
    target=0.0,
    abs_tol=1e-8,
    rel_tol=1e-2,
    max_iter=100,
    use_secant_first=True,
    scan_if_needed=True,
    scan_range=(-1e-2, 1e-2),
    n_samples=401,
    return_all=False,
):
    """
    Robust root finder for non‑monotonic residuals.
    - If 'bracket' is valid and straddles a root: secant-in-bracket + bisection fallback.
    - Else (or bracket=None) and scan_if_needed=True: coarse scan via scan_brackets/dips,
      then solve each bracket and pick the best root.
    Returns:
      (sigma0_root, f_val, iters) by default; if return_all=True also returns a list of
      per-root dicts like find_all_roots().
    """

    def f(s0):
        return residual(
            s0,
            integrate_fn,
            p_eqState,
            rho_eqState,
            r0,
            r_max,
            xi,
            target=target,
        )

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

            if abs(fs) <= abs_tol:
                return s, fs, it
            if abs(B - A) <= abs_tol * max(1.0, abs(s)):
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
    resid_kwargs = dict(
        integrate_fn=integrate_fn,
        p_eqState=p_eqState,
        rho_eqState=rho_eqState,
        r0=r0,
        r_max=r_max,
        xi=xi,
        target=target,
    )
    brackets, dips = scan_brackets(
        s0_min,
        s0_max,
        n_samples,
        integrate_fn=integrate_fn,
        p_eqState=p_eqState,
        rho_eqState=rho_eqState,
        xi=xi,
        **resid_kwargs
    )

    # Upgrade dips into brackets where possible
    width = 0.1 * (s0_max - s0_min) / n_samples
    for s0 in dips:
        b = refine_near_zero(
            s0,
            width,
            integrate_fn=integrate_fn,
            p_eqState=p_eqState,
            rho_eqState=rho_eqState,
            xi=xi,
            **resid_kwargs
        )
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
            fv = residual(
                s0,
                integrate_fn,
                p_eqState,
                rho_eqState,
                r0,
                r_max,
                xi,
                target=target,
            )
            if math.isfinite(fv) and abs(fv) <= abs_tol:
                roots.append(dict(sigma0=s0, f=fv, iters=0))
        except Exception:
            pass

    if not roots:
        raise RuntimeError(
            "No roots found in scan range; widen scan_range or n_samples."
        )

    # Pick the “best” root = min |f| (tie-breaker: smallest |sigma0|)
    best = min(roots, key=lambda d: (abs(d["f"]), abs(d["sigma0"])))
    return (
        (best["sigma0"], best["f"], best["iters"])
        if not return_all
        else ((best["sigma0"], best["f"], best["iters"]), roots)
    )


## Ψ2>0 post‑check
#    final = []
#    for r in chosen:
#        psi2 = psi2_center(r, p0, eps0, xi, lmbda)
#        if np.isfinite(psi2) and (psi2 > 0.0):
#            final.append(r)
