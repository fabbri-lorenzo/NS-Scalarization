import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from Utils.TOV_EMG import make_tov_EMG, initial_conditions, StoppingConditions
from concurrent.futures import ProcessPoolExecutor, as_completed

# GLOBAL SMALL CACHE
_SOL_CACHE = {}  # key: (round(sigma0, sig_nd), mode, rtol, atol, Rprobe_m) -> sol
def _cache_key(sigma0, mode, rtol, atol, Rprobe_m):
    # quantize σ0 so tiny float noise doesn't explode the cache
    sig_nd = 14  # tweak if needed
    return (round(float(sigma0), sig_nd), mode, float(rtol), float(atol), float(Rprobe_m) if Rprobe_m else None)
# Stricter, consistent settings for minima-based node counting
_STRICT_NC_KW = dict(
    domain="to_2R",   # always count only to 2 R*
    n_samples=8192,   # denser resample
    trim_margin=0.05, # trim 5% of the tail where tiny wiggles appear
    window_frac=0.08, # smoother filter
    dy_thresh_rel=5e-3,
    depth_rel=1e-2,   # demand deeper minima (suppresses micro-ripples)
    guard_frac=0.08   # ignore ends more aggressively
)

# -------------------------
# Integrate for a given central sigma0
# -------------------------
def integrate_sigma0(
    sigma0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
    *,
    rtol=1e-6, atol=1e-9,
    mode="full",          # "probe", "to_2R", or "full"
    Rprobe_m=None,        # used only when mode=="probe"
    dense=False           # ask for dense_output only when you really need it
):
    """
    mode="probe": stop at r = Rprobe_m (fast for stage-1 / relaxed scans)
    mode="to_2R": stop at 2*R⋆ using StoppingConditions (fast for node counting)
    mode="full" : no early stop (use only for final/plots)
    """
    r_span = (r0, r_max)
    y0 = initial_conditions(r0, sigma0, p_eqState)
    p_c = y0[0]

    tov = make_tov_EMG(p_c, frac_pc, rho_eqState, xi)
    events = []

    if mode == "probe":
        if Rprobe_m is None:
            raise ValueError("mode='probe' needs Rprobe_m")
        def hit_Rprobe(r, y):
            return r - Rprobe_m
        hit_Rprobe.terminal = True
        hit_Rprobe.direction = 1.0
        events.append(hit_Rprobe)

    elif mode == "to_2R":
        sc = StoppingConditions(p_c, frac_pc)
        events = [sc.pressure_limit(), sc.double_radius()]

    sol = solve_ivp(
        tov, r_span, y0,
        method='RK45',
        rtol=rtol, atol=atol,
        dense_output=bool(dense),
        events=events if events else None
    )
    return sol

# -------------------------
# Endpoint extractors / residuals
# -------------------------
def _sigma_at_rmax(sol, idx_sigma=3, **_):
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
             target=0.0, extractor=_sigma_ratio_at_rmax, extractor_kwargs=None,
             integ_kwargs=None):
    ek = dict(extractor_kwargs or {})
    ek.setdefault("sigma0", sigma0)
    ik = dict(integ_kwargs or {})
    sol = integrate_sigma0(sigma0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc, **ik)
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

# To estimate the value at the boundary (the first try for the shooting process)
def _value_at_r(sol, r_query, idx, default=float("nan")):
    """
    Robust point sampler:
    - tolerates tiny drift when an event hits right at r_query
    - uses dense_output if available, else linear interp
    - never returns None; falls back to `default` or float values
    """
    if (not hasattr(sol, "y")) or sol.y is None or sol.y.size == 0:
        return default
    r_all = sol.t
    if r_query < r_all[0] or r_query > r_all[-1] + 1e-12 * max(1.0, r_all[-1]):
        return default
    if abs(r_all[-1] - r_query) <= 1e-10 * max(1.0, r_query):
        return float(sol.y[idx, -1])
    if hasattr(sol, "sol") and callable(sol.sol):
        try:
            return float(sol.sol(r_query)[idx])
        except Exception:
            pass
    return float(np.interp(r_query, r_all, sol.y[idx]))

def _sigma_at_Rprobe(sol, Rprobe_m, idx_sigma=3, **_):
    return _value_at_r(sol, Rprobe_m, idx_sigma)

def _sigma_ratio_at_Rprobe(sol, sigma0, Rprobe_m, idx_sigma=3):
    sR = _sigma_at_Rprobe(sol, Rprobe_m, idx_sigma=idx_sigma)
    if sigma0 == 0.0:
        return float("inf") if sR != 0.0 else 0.0
    return sR / sigma0

def _get_sol_cached(
    sigma0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
    *, mode="probe", Rprobe_m=None, rtol=1e-5, atol=1e-8, dense=False
):
    key = _cache_key(sigma0, mode, rtol, atol, Rprobe_m)
    if key in _SOL_CACHE:
        return _SOL_CACHE[key]
    sol = integrate_sigma0(
        sigma0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
        rtol=rtol, atol=atol, mode=mode, Rprobe_m=Rprobe_m, dense=dense
    )
    _SOL_CACHE[key] = sol
    return sol

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

def _count_nodes_strict(sol, *, idx_sigma=3, idx_p=0, frac_pc=None):
    return node_count_from_sol(
        sol, idx_sigma=idx_sigma, idx_p=idx_p, frac_pc=frac_pc, **_STRICT_NC_KW
    )


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

# -------------------------
# Public API: shooter for a specific node number n
# -------------------------
def shoot_sigma0_for_mode(
    n,
    p_eqState, rho_eqState, xi,
    r0, r_max, frac_pc,
    bracket=None,
    scan_range=(1e-10, 1e-2), n_per_decade=48,
    idx_sigma=3,
    use_probe_first=True, Rprobe_m=300e3,      # 300 km
    tol_probe_abs=1e-10,                       # |σ(Rprobe)| target
    tol_ratio_probe=1e-2,                      # |σ(Rprobe)/σ0| relaxed target
    sigma0_min_abs=0.0,                        # e.g. 1e-12*M in caller
    fallback_to_rmax=True,
    max_iter=100,
    clear_cache_after=True,
    rtol_probe=1e-5, atol_probe=1e-8,          # fast-but-safe probe tolerances
    rtol_2R=1e-5,   atol_2R=1e-8,              # fast-but-safe to-2R tolerances
    assume_Z2=True,                             # exploit σ→−σ symmetry
    early_exit=True,                            # stop as soon as first constant-n bracket is found
    parallel_workers=2,                         # >0 enables parallel stage for coarse scan
    n_per_decade_refine=96                      # refinement bins around first bracket
):

    try:
        # ---------- adaptive grids over σ0 (coarse + refine) ----------
        lo, hi = scan_range
        if lo <= 0 or hi <= 0 or hi <= lo:
            raise ValueError("scan_range must be positive with hi>lo.")
        decades = (np.log10(hi) - np.log10(lo))
    
        nd_coarse = max(8, int(max(12, n_per_decade // 2) * decades) + 1)  # coarse
        nd_refine = max(n_per_decade_refine, int(64 * decades))            # fine
    
        mags_coarse = np.geomspace(lo, hi, nd_coarse)

        # PROBE BRACKETING NEEDS BOTH SIGNS. Do not compress to σ0>0 only.
        if use_probe_first:
            grid_coarse = np.ravel(np.column_stack((mags_coarse, -mags_coarse)))
        else:
            grid_coarse = (mags_coarse if assume_Z2 else
                           np.ravel(np.column_stack((mags_coarse, -mags_coarse))))
        grid_coarse = np.asarray(grid_coarse, dtype=float)

        grid_coarse = np.asarray(grid_coarse, dtype=float)
    
        # ---------- helpers that use the faster tolerances ----------
        def _verify_mode(s0):
            sol_2R = _get_sol_cached(
                s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                mode="to_2R", Rprobe_m=None, rtol=rtol_2R, atol=atol_2R, dense=False
            )
            k = _count_nodes_strict(sol_2R, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc)
            return (k == n)
    
        def _probe_and_nodes(s0):
            """Return (s0, nn, sigR) for this σ0 or (s0, None, nan) on failure."""
            try:
                sol_2R = _get_sol_cached(
                    s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                    mode="to_2R", Rprobe_m=None, rtol=rtol_2R, atol=atol_2R, dense=False
                )
                nn = _count_nodes_strict(sol_2R, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc)

                if nn != n or abs(s0) < sigma0_min_abs:
                    return (s0, nn, float("nan"))
    
                sol_probe = _get_sol_cached(
                    s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                    mode="probe", Rprobe_m=Rprobe_m, rtol=rtol_probe, atol=atol_probe, dense=False
                )
                sigR = _sigma_at_Rprobe(sol_probe, Rprobe_m, idx_sigma=idx_sigma)
                return (s0, nn, sigR if math.isfinite(sigR) else float("nan"))
            except Exception:
                return (s0, None, float("nan"))
            
        def _constant_n_subbracket_local(a, b, require_same_sign=True, bins=64):
            a, b = (a, b) if a < b else (b, a)
            if a == b:
                raise RuntimeError("Provided bracket has zero width.")
            x = np.linspace(a, b, int(bins) + 1)
            prev_s0, prev_val, prev_n = None, None, None
            for s0 in x:
                if abs(s0) < sigma0_min_abs:
                    prev_s0, prev_val, prev_n = s0, None, None
                    continue
    
                # value at probe (fast, cached)
                sol_probe = _get_sol_cached(
                    s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                    mode="probe", Rprobe_m=Rprobe_m, rtol=rtol_probe, atol=atol_probe, dense=False
                )
                val = _sigma_at_Rprobe(sol_probe, Rprobe_m, idx_sigma=idx_sigma)
    
                # node count to 2R* (fast, cached)
                sol_2R = _get_sol_cached(
                    s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                    mode="to_2R", Rprobe_m=None, rtol=rtol_2R, atol=atol_2R, dense=False
                )
                nn = _count_nodes_strict(sol_2R, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc)

    
                if prev_s0 is not None and prev_val is not None and prev_n is not None:
                    if nn == n and prev_n == n:
                        if (not require_same_sign) or (prev_s0 * s0 > 0):
                            if math.isfinite(val) and math.isfinite(prev_val) and (val * prev_val < 0.0):
                                return (prev_s0, s0)
    
                prev_s0, prev_val, prev_n = s0, val, nn
    
            raise RuntimeError("Provided bracket crosses a node-transition; no constant-n sub-bracket found.")    
    
        # ---------- user-provided bracket path (unchanged logic except tolerances) ----------
        if bracket is not None:
            a, b = _constant_n_subbracket_local(bracket[0], bracket[1], require_same_sign=False,     bins=nd_refine)
    
            if use_probe_first:
                def f_probe(s0):
                    return residual(
                        s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                        target=0.0,
                        extractor=_sigma_at_Rprobe,
                        extractor_kwargs={"Rprobe_m": Rprobe_m, "idx_sigma": idx_sigma},
                        integ_kwargs={"mode": "probe", "Rprobe_m": Rprobe_m,
                                      "rtol": rtol_probe, "atol": atol_probe, "dense": False}
                    )
                root, fval, iters = _solve_in_bracket(a, b, f_probe, tol=max(tol_probe_abs, 1e-18),     max_iter=max_iter)
                if abs(root) >= sigma0_min_abs and _verify_mode(root):
                    return root
    
                # relaxed σ(Rprobe)/σ0
                best = None
                for s0 in np.linspace(min(a, b), max(a, b), 128):
                    if abs(s0) < sigma0_min_abs:
                        continue
                    try:
                        sol = integrate_sigma0(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                                               rtol=rtol_probe, atol=atol_probe, mode="probe",     Rprobe_m=Rprobe_m)
                        if node_count_from_sol(sol, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc,     domain="to_2R") != n:
                            continue
                        rati = _sigma_ratio_at_Rprobe(sol, s0, Rprobe_m, idx_sigma=idx_sigma)
                        if math.isfinite(rati):
                            val = abs(rati)
                            if (best is None) or (val < best[0]):
                                best = (val, s0)
                    except Exception:
                        pass
                if best is not None and best[0] <= tol_ratio_probe:
                    return best[1]
    
                if not fallback_to_rmax:
                    raise RuntimeError("Probe criteria failed in user bracket.")
    
            # fallback legacy at r_max
            def f_ratio_rmax(s0):
                return residual(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                                target=0.0,
                                extractor=_sigma_ratio_at_rmax,
                                extractor_kwargs={"idx_sigma": idx_sigma})
            root, fval, iters = _solve_in_bracket(a, b, f_ratio_rmax, tol=1e-2, max_iter=max_iter)
            if abs(root) >= sigma0_min_abs and _verify_mode(root):
                return root
            raise RuntimeError("Refined root invalid or lost node count in legacy path.")
    
        # ---------- no user bracket: incremental coarse scan (optional parallel) ----------
        found_bracket = None
    
        if parallel_workers and not early_exit:
            # parallel precomputation over the whole coarse grid
            results = []
            with ProcessPoolExecutor(max_workers=int(parallel_workers)) as ex:
                futs = {ex.submit(_probe_and_nodes, float(s0)): s0 for s0 in grid_coarse}
                for fut in as_completed(futs):
                    s0, nn, sigR = fut.result()
                    results.append((s0, nn, sigR))
            # sort by |σ0| so neighbors make sense
            results.sort(key=lambda t: abs(t[0]))
            prev = None
            for s0, nn, sigR in results:
                if nn != n or not math.isfinite(sigR) or abs(s0) < sigma0_min_abs:
                    prev = None
                    continue
                if prev is not None:
                    ps0, psigR = prev
                    if psigR * sigR < 0.0:
                        found_bracket = (ps0, s0)
                        break
                prev = (s0, sigR)
        else:
            # sequential, early-exit friendly
            prev_s0, prev_sigR = None, None
            for s0 in grid_coarse:
                s0, nn, sigR = _probe_and_nodes(float(s0))
                if nn != n or not math.isfinite(sigR) or abs(s0) < sigma0_min_abs:
                    prev_s0, prev_sigR = None, None
                    continue
                if (prev_s0 is not None) and (prev_sigR * sigR < 0.0):
                    found_bracket = (prev_s0, s0)
                    if early_exit:
                        break
                prev_s0, prev_sigR = s0, sigR
    
        # ---------- refine bracket locally with higher density ----------
        if found_bracket is not None:
            a, b = found_bracket
            a, b = (a, b) if a < b else (b, a)
            # tighten to a constant-n sub-bracket with more bins
            a, b = _constant_n_subbracket_local(a, b, require_same_sign=True, bins=nd_refine)
    
            if use_probe_first:
                def f_probe(s0):
                    return residual(
                        s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                        target=0.0,
                        extractor=_sigma_at_Rprobe,
                        extractor_kwargs={"Rprobe_m": Rprobe_m, "idx_sigma": idx_sigma},
                        integ_kwargs={"mode": "probe", "Rprobe_m": Rprobe_m,
                                      "rtol": rtol_probe, "atol": atol_probe, "dense": False}
                    )
                root, fval, iters = _solve_in_bracket(a, b, f_probe, tol=max(tol_probe_abs, 1e-18),     max_iter=max_iter)
                if abs(root) >= sigma0_min_abs and _verify_mode(root):
                    return root
    
                # relaxed ratio at probe
                best = None
                for s0 in np.linspace(a, b, 128):
                    if abs(s0) < sigma0_min_abs:
                        continue
                    s0, nn, sigR = _probe_and_nodes(float(s0))
                    if nn != n or not math.isfinite(sigR):
                        continue
                    rati = sigR / s0 if s0 != 0.0 else float("inf")
                    if math.isfinite(rati):
                        val = abs(rati)
                        if (best is None) or (val < best[0]):
                            best = (val, s0)
                if best is not None and best[0] <= tol_ratio_probe and _verify_mode(best[1]):
                    return best[1]
    
                if not fallback_to_rmax:
                    raise RuntimeError("Probe criteria failed in found bracket.")
    
            # legacy ratio at r_max
            def f_ratio_rmax(s0):
                return residual(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                                target=0.0,
                                extractor=_sigma_ratio_at_rmax,
                                extractor_kwargs={"idx_sigma": idx_sigma})
            root, fval, iters = _solve_in_bracket(a, b, f_ratio_rmax, tol=1e-2, max_iter=max_iter)
            if abs(root) >= sigma0_min_abs and _verify_mode(root):
                return root

            # ---------- ultimate fallback: legacy full-grid scan at r_max (kept, but thinned if we refined)         ----------
            vals, nodes_vals = [], []
            scan_grid = grid_coarse if found_bracket is None else np.linspace(a, b, 64)
            for s0 in scan_grid:
                try:
                    sol = integrate_sigma0(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc)
                    nodes_vals.append(node_count_from_sol(sol, idx_sigma=idx_sigma, idx_p=0, frac_pc=frac_pc,         domain="to_2R"))
                    vals.append(_sigma_ratio_at_rmax(sol, s0, idx_sigma=idx_sigma))
                except Exception:
                    nodes_vals.append(None)
                    vals.append(float("inf"))
        
            def _collect_constant_n_brackets(vals, nodes_arr, require_same_sign=False):
                cands = []
                run_start = None
                grid = scan_grid
                for i in range(len(grid)):
                    s0 = grid[i]
                    if nodes_arr[i] == n and math.isfinite(vals[i]) and abs(s0) >= sigma0_min_abs:
                        if run_start is None:
                            run_start = i
                    else:
                        if run_start is not None:
                            for j in range(run_start, i - 1):
                                a, b = grid[j], grid[j+1]
                                if abs(a) < sigma0_min_abs or abs(b) < sigma0_min_abs:
                                    continue
                                if require_same_sign and (a * b <= 0):
                                    continue
                                vj, vk = vals[j], vals[j+1]
                                if math.isfinite(vj) and math.isfinite(vk) and (vj * vk < 0.0):
                                    cands.append((a, b))
                            run_start = None
                if run_start is not None:
                    i = len(grid)
                    for j in range(run_start, i - 1):
                        a, b = grid[j], grid[j+1]
                        if abs(a) < sigma0_min_abs or abs(b) < sigma0_min_abs:
                            continue
                        if require_same_sign and (a * b <= 0):
                            continue
                        vj, vk = vals[j], vals[j+1]
                        if math.isfinite(vj) and math.isfinite(vk) and (vj * vk < 0.0):
                            cands.append((a, b))
                return cands
        
            cand = _collect_constant_n_brackets(vals, nodes_vals, require_same_sign=False)
            if not cand:
                raise RuntimeError("No constant-n sign flip found in scan_range (legacy path).")
            for a, b in sorted(cand, key=lambda ab: max(abs(ab[0]), abs(ab[1]))):
                def f_ratio_rmax(s0):
                    return residual(s0, p_eqState, rho_eqState, xi, r0, r_max, frac_pc,
                                    target=0.0,
                                    extractor=_sigma_ratio_at_rmax,
                                    extractor_kwargs={"idx_sigma": idx_sigma})
                root, fval, iters = _solve_in_bracket(a, b, f_ratio_rmax, tol=1e-2, max_iter=max_iter)
                if abs(root) >= sigma0_min_abs and _verify_mode(root):
                    return root
        
            raise RuntimeError("All legacy brackets converged to σ0 below threshold or lost node count.")
    finally:
           if clear_cache_after:
            _SOL_CACHE.clear()