import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from Utils.TOV_EMG import make_tov_system, initial_conditions

# -------------------------
# Integrate for a given central sigma0
# -------------------------
def integrate_sigma0(sigma0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc):
    r_span = (r0, r_max)
    y0 = initial_conditions(r0, sigma0, p_eqState)
    p_c = y0[0]
    tov = make_tov_system(p_c, frac_pc, rho_eqState, xi)
    sol = solve_ivp(tov, r_span, y0, method='RK45', rtol=1e-6, atol=1e-9, dense_output=True)
    return sol

# -------------------------
# Endpoint extractors / residuals
# -------------------------
def _sigma_at_rmax(sol, idx_sigma=3):
    if (not hasattr(sol, "y")) or sol.y is None or sol.y.size == 0:
        return float("nan")
    return float(sol.y[idx_sigma, -1])

def _sigma_ratio_at_rmax(sol, sigma0, idx_sigma=3):
    s_inf = _sigma_at_rmax(sol, idx_sigma=idx_sigma)
    if sigma0 == 0.0:
        return float("inf") if s_inf != 0.0 else 0.0
    return s_inf / sigma0

def endpoint_mismatch(sol, target=0.0, extractor=_sigma_at_rmax, **extractor_kwargs):
    if (not hasattr(sol, "y")) or sol.y is None or sol.y.size == 0:
        return float("inf")
    if hasattr(sol, "success") and not sol.success:
        return float("inf")
    val = extractor(sol, **extractor_kwargs)
    f = val - target
    return f if math.isfinite(f) else float("inf")

def residual(sigma0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
             target=0.0, extractor=_sigma_ratio_at_rmax, extractor_kwargs=None):
    ek = dict(extractor_kwargs or {})
    ek.setdefault("sigma0", sigma0)
    sol = integrate_sigma0(sigma0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc)
    return endpoint_mismatch(sol, target=target, extractor=extractor, **ek)

def _estimate_R_star(sol, idx_p=0, frac_pc=None):
    """
    Return R_* (in the solver's r-units) as the last radius where p > frac_pc * p_c.
    If unavailable, returns None.
    """
    if frac_pc is None:
        return None
    p = sol.y[idx_p]
    pc = p[0]
    above = p > (frac_pc * pc)
    if not np.any(above):
        return None
    last_idx = np.where(above)[0][-1]
    return float(sol.t[last_idx])

def _extract_series(sol, idx_sigma=3, idx_p=0, frac_pc=None, domain="full", n_samples=4096):
    r_all = sol.t
    y_all = sol.y[idx_sigma]
    r0, r1 = r_all[0], r_all[-1]

    if domain == "inside_star" and frac_pc is not None:
        p = sol.y[idx_p]; pc = p[0]
        mask = p > (frac_pc * pc)
        if np.any(mask):
            r_raw, y_raw = r_all[mask], y_all[mask]
        else:
            r_raw, y_raw = r_all, y_all

    elif domain in ("to_2R", "star_to_2R") and frac_pc is not None:
        R = _estimate_R_star(sol, idx_p=idx_p, frac_pc=frac_pc)
        if R is None:
            r_raw, y_raw = r_all, y_all
        else:
            r_end = min(2.0 * R, r1)
            if r_end >= r1:
                r_raw, y_raw = r_all, y_all
            else:
                last = max(1, np.searchsorted(r_all, r_end, side="right") - 1)
                r_raw, y_raw = r_all[:last+1], y_all[:last+1]
    else:
        # domain == "full" (default)
        r_raw, y_raw = r_all, y_all

    # uniform resampling over [r_raw[0], r_raw[-1]]
    r_u = np.linspace(r_raw[0], r_raw[-1], min(n_samples, max(64, r_raw.size)))
    if hasattr(sol, "sol") and callable(sol.sol):
        y_u = sol.sol(r_u)[idx_sigma]
    else:
        y_u = np.interp(r_u, r_raw, y_raw)
    return r_u, y_u

# -------------------------
# Node counter (debounced zero crossings)
# -------------------------
def _phase_normalize(r, y):
    n = len(r)
    m = max(5, int(0.02 * n))
    m = min(m, n - 3)
    if m <= 3:
        return y, 1.0
    dy0 = np.gradient(y[:m], r[:m])
    s0 = np.sign(np.nanmean(dy0)) if np.any(np.isfinite(dy0)) else 0.0
    # if starts flat/upwards, flip so that the start is effectively "going down"
    return (-y, -1.0) if s0 >= 0 else (y, 1.0)

def _count_minima_savgol(
    r, y,
    window_frac=0.05, poly=3,
    dy_thresh_rel=1e-3, depth_rel=2e-3,
    guard_frac=0.03
):
    n = len(r)
    if n < 15:
        return 0, []

    # SG window (odd, within bounds)
    w = max(7, int(window_frac * n) | 1)
    if w >= n:  # clamp if series is short
        w = n - 1 if (n - 1) % 2 == 1 else n - 2

    # smooth σ, its first and second derivatives
    dr = (r[-1] - r[0]) / (n - 1) if n > 1 else 1.0
    y_s  = savgol_filter(y, w, poly, mode="interp")
    dy_s = savgol_filter(y, w, poly, deriv=1, delta=dr, mode="interp")
    d2_s = savgol_filter(y, w, poly, deriv=2, delta=dr, mode="interp")

    # derivative threshold + sign
    thr = dy_thresh_rel * (np.nanmax(np.abs(dy_s)) if np.any(np.isfinite(dy_s)) else 0.0)
    s = np.sign(dy_s)
    s[np.abs(dy_s) <= thr] = 0

    lo = int(guard_frac * n)
    hi = n - 1 - lo
    span = np.nanmax(y_s) - np.nanmin(y_s) if np.any(np.isfinite(y_s)) else 0.0
    depth_thr = depth_rel * span
    half = max(2, w // 4)

    idxs = []
    i = lo
    while i < hi:
        # look for - to + crossing of smoothed derivative
        if s[i] < 0 and s[i+1] > 0:
            a = max(lo, i - half)
            b = min(hi, i + half)
            # refine min position by argmin of smoothed σ
            m = a + int(np.argmin(y_s[a:b+1]))

            # must be a convex minimum (σ'' > 0)
            if d2_s[m] <= 0:
                i += 1
                continue

            # shoulders & depth check to suppress micro-ripples
            L = max(lo, m - half); R = min(hi, m + half)
            left  = np.nanmedian(y_s[L:m]) if m - L >= 2 else y_s[L]
            right = np.nanmedian(y_s[m+1:R+1]) if R - (m + 1) >= 1 else y_s[R]
            depth = min(left - y_s[m], right - y_s[m])
            if depth >= depth_thr:
                idxs.append(m)
                i = m + half  # skip past this valley
                continue
        i += 1

    return len(idxs), idxs

def node_count_from_sol(
    sol, idx_sigma=3, idx_p=0, frac_pc=None, domain="to_2R",
    n_samples=4096, trim_margin=0.01,
    # tuning knobs (loosen/tighten if needed)
    window_frac=0.05, poly=3,
    dy_thresh_rel=1e-3, depth_rel=2e-3,
    guard_frac=0.03
):
    """
    Interpret 'node number' as the number of local MINIMA of σ(r) on the chosen domain.
    Phase-normalizes σ so it starts decreasing; then finds minima via SG-smoothed dσ/dr
    zero-crossings with a curvature and depth check.
    """
    r, y = _extract_series(sol, idx_sigma=idx_sigma, idx_p=idx_p,
                           frac_pc=frac_pc, domain=domain, n_samples=n_samples)
    if r.size < 15:
        return 0

    # light tail trim
    cut = max(0, int(trim_margin * r.size))
    if cut > 0 and r.size - cut >= 15:
        r, y = r[:-cut], y[:-cut]

    # ensure the first segment is "descending" to define minima consistently
    y_norm, _ = _phase_normalize(r, y)

    cnt, _ = _count_minima_savgol(
        r, y_norm,
        window_frac=window_frac, poly=poly,
        dy_thresh_rel=dy_thresh_rel, depth_rel=depth_rel,
        guard_frac=guard_frac
    )
    return int(cnt)



# -------------------------
# Bracketed solver (secant inside [a,b] + bisection fallback)
# -------------------------
def _solve_in_bracket(a, b, f, tol=1e-2, max_iter=100):
    fa, fb = f(a), f(b)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise RuntimeError("Non-finite residual at bracket ends.")
    if fa == 0.0:
        return a, 0.0, 0
    if fb == 0.0:
        return b, 0.0, 0
    if fa * fb > 0.0:
        raise ValueError("Bracket does not straddle a root.")

    A, B, fA, fB = a, b, fa, fb
    for it in range(1, max_iter + 1):
        # secant step confined to [A,B]
        s = B - fB * (B - A) / (fB - fA) if (fB - fA) != 0.0 else 0.5*(A+B)
        if not (min(A, B) < s < max(A, B)) or not math.isfinite(s):
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

    m = 0.5 * (A + B)
    return m, f(m), max_iter

# --- force a user bracket into a constant-n sub-bracket ---
def _constant_n_subbracket(a, b, n, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                           idx_sigma=3, bins=24):
    a, b = (a, b) if a < b else (b, a)
    if a == b:
        raise RuntimeError("Provided bracket has zero width.")
    bins = max(24, int(bins))
    grid = np.linspace(a, b, bins + 1)
    vals, nodes = [], []
    for s0 in grid:
        sol = integrate_sigma0(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc)
        nodes.append(node_count_from_sol(sol, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc, domain="to_2R"))
        vals.append(_sigma_ratio_at_rmax(sol, s0, idx_sigma=idx_sigma))

    for i in range(len(grid) - 1):
        if nodes[i] == n and nodes[i+1] == n:
            fi, fj = vals[i], vals[i+1]
            if math.isfinite(fi) and math.isfinite(fj) and fi * fj < 0.0:
                return (grid[i], grid[i+1])
    raise RuntimeError("Provided bracket crosses a node-transition; no constant-n sub-bracket found.")

# -------------------------
# Public API: shooter for a specific node number n
# -------------------------
def shoot_sigma0_for_mode(n, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                          bracket=None, scan_range=(1e-10, 1e-2), n_per_decade=12,
                          idx_sigma=3, tol_ratio=1e-2, max_iter=100):
    """
    Find sigma0 such that sigma(r_max)/sigma0 ≈ 0 AND the solution has exactly n nodes.
    Strategy: log scan of |sigma0| (both signs), keep consecutive pairs with the SAME node
    count n and opposite signs of the ratio; pick the smallest-|sigma0| bracket and refine.
    """
    # build scan grid
    lo, hi = scan_range
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError("scan_range must be positive with hi>lo.")
    mags = np.geomspace(lo, hi, max(2, int(n_per_decade * (np.log10(hi) - np.log10(lo)) + 1)))
    grid = np.ravel(np.column_stack((mags, -mags)))  # interleaved ±

    def ratio(s0):
        return residual(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                        target=0.0, extractor=_sigma_ratio_at_rmax, extractor_kwargs={"idx_sigma": idx_sigma})

    # evaluate nodes and ratios
    vals = []
    nodes = []
    for s0 in grid:
        try:
            sol = integrate_sigma0(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc)
            nodes.append(node_count_from_sol(sol, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc, domain="to_2R"))
            vals.append(_sigma_ratio_at_rmax(sol, s0, idx_sigma=idx_sigma))
        except Exception:
            nodes.append(None)
            vals.append(float("inf"))

    if bracket is not None:
        a, b = bracket

        # try to enforce constant-n by subdividing the bracket
        try:
            a, b = _constant_n_subbracket(a, b, n, p_eqState, rho_eqState, xi,
                                          r0, r_max, frac_pc, idx_sigma=idx_sigma, bins=48)
        except RuntimeError as e:
            # fall back to strict check to explain what's wrong
            sol_a = integrate_sigma0(a, p_eqState, rho_eqState, xi, r0, r_max, frac_pc)
            sol_b = integrate_sigma0(b, p_eqState, rho_eqState, xi, r0, r_max, frac_pc)
            na = node_count_from_sol(sol_a, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc, domain="to_2R")
            nb = node_count_from_sol(sol_b, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc, domain="to_2R")
            raise RuntimeError(f"Bracket node mismatch: n(a)={na}, n(b)={nb}, expected {n}. {e}")

        # refine inside the constant-n sub-bracket on the ratio observable
        def ratio(s0):
            return residual(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                            target=0.0, extractor=_sigma_ratio_at_rmax,
                            extractor_kwargs={"idx_sigma": idx_sigma})

        root, fval, iters = _solve_in_bracket(a, b, ratio, tol=tol_ratio, max_iter=max_iter)

        # sanity: verify node count at the refined root
        sol = integrate_sigma0(root, p_eqState, rho_eqState, xi, r0, r_max, frac_pc)
        k = node_count_from_sol(sol, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc, domain="to_2R")
        if k != n:
            raise RuntimeError(f"Refined root lost node count: expected {n}, got {k}.")
        return root
    else:
        # collect brackets ONLY inside constant-n runs
        candidates = []
        run_start = None
        for i in range(len(grid)):
            if nodes[i] == n:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    # process run [run_start, i-1]
                    for j in range(run_start, i - 1):
                        fj, fk = vals[j], vals[j+1]
                        if math.isfinite(fj) and math.isfinite(fk) and fj * fk < 0.0:
                            candidates.append((grid[j], grid[j+1]))
                    run_start = None
        if run_start is not None:  # tail run
            i = len(grid)
            for j in range(run_start, i - 1):
                fj, fk = vals[j], vals[j+1]
                if math.isfinite(fj) and math.isfinite(fk) and fj * fk < 0.0:
                    candidates.append((grid[j], grid[j+1]))
    
        if not candidates:
            raise RuntimeError("No constant-n sign flip found in scan_range; widen the range or increase n_per_decade.")
    
        # refine the smallest-|sigma0| candidate
        a, b = min(candidates, key=lambda ab: max(abs(ab[0]), abs(ab[1])))
        root, fval, iters = _solve_in_bracket(a, b,
                                              lambda s0: residual(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                                                                  target=0.0, extractor=_sigma_ratio_at_rmax,
                                                                  extractor_kwargs={"idx_sigma": idx_sigma}),
                                              tol=tol_ratio, max_iter=max_iter)
    
        sol = integrate_sigma0(root, p_eqState, rho_eqState, xi, r0, r_max, frac_pc)
        k = node_count_from_sol(sol, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc)
        if k != n:
            raise RuntimeError(f"Refined root lost node count: expected {n}, got {k}.")
        return root
