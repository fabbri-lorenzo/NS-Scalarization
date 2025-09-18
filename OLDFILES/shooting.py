# shooting.py
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from Utils.TOV_EMG import make_tov_EMG, initial_conditions
import copy

# =========================================================
#            CORE INTEGRATORS (full + probe)
# =========================================================

def integrate_sigma0(sigma0, p_eqState, rho_eqState, xi, rho0, r0, r_max, frac_pc):
    """
    Full integration to r_max with dense_output enabled.
    Slow but robust; used for verification and legacy fallbacks.
    """
    r_span = (r0, r_max)
    y0 = initial_conditions(r0, sigma0, p_eqState,  xi, rho0)
    p_c = y0[0]
    tov = make_tov_EMG(p_c, frac_pc, rho_eqState, xi)
    sol = solve_ivp(
        tov, r_span, y0,
        method='RK45',
        rtol=1e-6, atol=1e-9,
        dense_output=True
    )
    return sol


def sigma_infty_residual_S6(sol, k_tail=9):
    """
    Sixth-order tail residual for σ∞ (should be ~0 at infinity).
    Uses three late radii r1<r2<r3 and cancels 1/r^3 and 1/r^5 terms.

    R(r) := r σ'(r) + σ(r) = σ∞ - 2 a3/r^3 - 4 a5/r^5 + O(r^-7)
    Choose weights w1,w2,w3 s.t.:
      w1+w2+w3=1,  Σ wi/ri^3 = 0,  Σ wi/ri^5 = 0  ⇒  S6 := Σ wi R(ri) = σ∞ + O(r^-7)

    Returns a scalar residual S6; sign should flip across a good bracket.
    """
    import numpy as np

    if (not hasattr(sol, "y")) or sol.y is None or sol.y.size == 0:
        return float("nan")

    r = sol.t
    sigma = sol.y[2]
    dsigma = sol.y[3]
    n = r.size
    if n < 6:
        # degenerate fallback: last-point residual (still finite, low order)
        return float(r[-1] * dsigma[-1] + sigma[-1])

    k_tail = max(5, min(int(k_tail), n-1))
    r_tail    = r[-k_tail:]
    sigma_tail= sigma[-k_tail:]
    dsig_tail = dsigma[-k_tail:]

    # pick 3 well-spaced indices in the tail window
    i1 = int(0.60*(k_tail-1))
    i2 = int(0.80*(k_tail-1))
    i3 = k_tail-1

    r1, r2, r3 = r_tail[[i1, i2, i3]]
    R1 = r1 * dsig_tail[i1] + sigma_tail[i1]
    R2 = r2 * dsig_tail[i2] + sigma_tail[i2]
    R3 = r3 * dsig_tail[i3] + sigma_tail[i3]

    A = np.array([
        [1.0,          1.0,          1.0         ],
        [r1**-3.0,     r2**-3.0,     r3**-3.0    ],
        [r1**-5.0,     r2**-5.0,     r3**-5.0    ],
    ], dtype=float)
    b = np.array([1.0, 0.0, 0.0], dtype=float)

    try:
        w1, w2, w3 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # if radii are nearly collinear in this space, fall back to simple average
        w1 = w2 = w3 = 1.0/3.0

    S6 = w1*R1 + w2*R2 + w3*R3
    return float(S6)

# =========================================================
#            SIGN SYMMETRY HELPERS
# =========================================================

def _sigma_sign(sigma0: float):
    """Return (sgn, abs_sigma0) with sgn in {+1,-1}."""
    return (-1.0 if sigma0 < 0 else 1.0), abs(float(sigma0))

def _mirror_sigma_sign(sol, sgn: float):
    """
    If sgn<0, return a shallow-copied OdeResult where only the scalar field
    components are sign-flipped. Assumes y[2]=σ and y[3]=σ'.
    Preserves dense_output via a thin callable wrapper.
    """
    if sgn >= 0:
        return sol

    new = copy.copy(sol)
    new_y = sol.y.copy()
    new_y[2] *= -1.0
    new_y[3] *= -1.0
    new.y = new_y

    # Keep dense_output behavior, but mirror σ and σ' on the fly
    if getattr(sol, "sol", None) is not None:
        orig = sol.sol

        class _MirroredDense:
            def __init__(self, _orig):
                self._orig = _orig
            def __call__(self, r):
                Y = self._orig(r)
                Y[2] = -Y[2]
                Y[3] = -Y[3]
                return Y

        new.sol = _MirroredDense(orig)

    return new

# =========================================================
#                 SMALL SOLUTION CACHE
# =========================================================

_SOL_CACHE = {}  # key: (round(sigma0, 14), mode, rtol, atol, Rprobe_m) -> sol

def _cache_key(sigma0, rtol, atol, dense_output=True):
    return (
        round(abs(float(sigma0)), 12),
        float(rtol),
        float(atol),
        bool(dense_output),
    )
    
def _get_sol_cached(
    sigma0, p_eqState, rho_eqState, xi, rho0, r0, r_max, frac_pc,
    *, rtol=1e-6, atol=1e-9, dense_output=True
):
    """
    Always perform a full integration to r_max (no probe mode).
    Caches by (|sigma0|, rtol, atol, dense_output).
    """
    key = _cache_key(sigma0, rtol, atol, dense_output)
    if key in _SOL_CACHE:
        return _SOL_CACHE[key]

    sol = integrate_sigma0(sigma0, p_eqState, rho_eqState, xi, rho0, r0, r_max, frac_pc)
    _SOL_CACHE[key] = sol
    return sol
    

def clear_shoot_cache():
    """Public helper to clear the internal solution cache."""
    _SOL_CACHE.clear()


# =========================================================
#             STAR RADIUS ESTIMATION & SAMPLERS
# =========================================================

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

def _extract_series(sol, idx_sigma, idx_p, frac_pc=None, domain="full", n_samples=4096):
    """
    Extract σ(r) on a uniform r-grid, optionally restricted to inside the star or up to 2R_*.
    Uses dense_output if available; otherwise linear interpolation on (sol.t, sol.y).
    """
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


# =========================================================
#          NODE COUNTER = ZERO-CROSSINGS OF σ(r)
# =========================================================

def _count_zero_crossings(
    r, y,
    *,
    smooth=True,
    window_frac=0.05,
    poly=3,
    zero_band_rel=3e-3,
    min_persist_frac=0.01,
    peak_rel=5e-3,
    guard_frac=0.03,
    merge_dr_frac=0.01,
    # NEW: optional absolute cutoff radius; only crossings with r <= r_cut are considered
    r_cut=None,
):
    """
    Count robust ZERO-CROSSINGS of y(r) while suppressing tiny fluctuations.

    If r_cut is provided (float), the series is truncated to r <= r_cut BEFORE
    applying guards/hysteresis/persistence checks.
    """
    n = len(r)
    if n < 3:
        return 0

    if r_cut is not None and np.isfinite(r_cut):
        # keep up to the last index where r <= r_cut
        last = np.searchsorted(r, r_cut, side="right") - 1
        if last < 2:
            return 0
        r = r[:last+1]
        y = y[:last+1]
        n = len(r)

    # (0) smoothing (optional)
    y_s = y
    if smooth:
        w = max(7, int(window_frac * n) | 1)
        if w >= n:
            w = n - 1 if (n - 1) % 2 == 1 else n - 2
        y_s = savgol_filter(y, w, poly, mode="interp")

    # (1) amplitude and guards
    amp = np.nanmax(np.abs(y_s)) if np.any(np.isfinite(y_s)) else 0.0
    if not np.isfinite(amp) or amp == 0.0:
        return 0

    lo = int(guard_frac * n)
    hi = n - 1 - lo
    if hi - lo < 2:
        return 0
    r_seg = r[lo:hi+1]
    y_seg = y_s[lo:hi+1]
    m = len(r_seg)

    # (2) hysteresis band & thresholds
    band = float(zero_band_rel) * amp
    peak_thr = float(peak_rel) * amp
    min_persist_pts = max(1, int(min_persist_frac * m))
    merge_dr = merge_dr_frac * (r[-1] - r[0]) if n > 1 else 0.0

    def hyst_sign(val, prev):
        if val > +band:
            return +1
        if val < -band:
            return -1
        # inside band: keep previous sign (prevents chatter)
        return prev

    cnt = 0
    s_prev = 0
    last_flip_idx = 0
    last_flip_r = r_seg[0] - 2.0 * merge_dr
    run_peak = abs(y_seg[0])

    for i in range(1, m):
        s_cur = hyst_sign(y_seg[i], s_prev)
        run_peak = max(run_peak, abs(y_seg[i]))

        if s_cur != s_prev and s_cur != 0 and s_prev != 0:
            # persistence check on previous run
            run_len = i - last_flip_idx
            if run_len < min_persist_pts:
                s_prev = s_cur
                continue

            # peak excursion check
            if run_peak < peak_thr:
                s_prev = s_cur
                continue

            # merge by r-separation
            if (r_seg[i] - last_flip_r) < merge_dr:
                s_prev = s_cur
                continue

            # valid flip
            cnt += 1
            last_flip_idx = i
            last_flip_r = r_seg[i]
            run_peak = 0.0

        s_prev = s_cur

    return int(cnt)


def node_count_from_sol(
    sol, idx_sigma, idx_p, frac_pc=None, domain="to_2R",
    n_samples=4096, trim_margin=0.01,
    # smoothing/robustness knobs
    window_frac=0.05, poly=3,
    zero_band_rel=3e-3,       # hysteresis band around 0 (relative to max|σ|)
    min_persist_frac=0.01,    # min fraction of points a sign must persist
    peak_rel=5e-3,            # min peak between flips (relative to max|σ|)
    guard_frac=0.03,          # ignore ends
    merge_dr_frac=0.01        # merge flips closer than this r-fraction
):
    """
    Node number = debounced ZERO-CROSSINGS of σ(r), counted ONLY up to 2*R_*.

    We compute R_* from the pressure threshold (frac_pc * p_c). The time/radial series
    is truncated at min(2*R_*, r_end) before counting, irrespective of the resampling
    domain (still using domain to shape the interpolation density).
    """
    # Resample (optionally already trimmed by domain), but we will enforce a hard cutoff below.
    r, y = _extract_series(sol, idx_sigma=idx_sigma, idx_p=idx_p,
                           frac_pc=frac_pc, domain=domain, n_samples=n_samples)
    if r.size < 3:
        return 0

    # Determine hard cutoff at 2 * R_*
    r_cut = None
    if frac_pc is not None:
        R = _estimate_R_star(sol, idx_p=idx_p, frac_pc=frac_pc)
        if R is not None and np.isfinite(R):
            r_cut = min(2.0 * R, r[-1])

    # light tail trim first (on the resampled series), then apply r_cut inside the counter
    cut = max(0, int(trim_margin * r.size))
    if cut > 0 and r.size - cut >= 3:
        r, y = r[:-cut], y[:-cut]

    return _count_zero_crossings(
        r, y,
        smooth=True,
        window_frac=window_frac,
        poly=poly,
        zero_band_rel=zero_band_rel,
        min_persist_frac=min_persist_frac,
        peak_rel=peak_rel,
        guard_frac=guard_frac,
        merge_dr_frac=merge_dr_frac,
        r_cut=r_cut,  
    )


# =========================================================
#     BRACKETED SOLVER (secant-in-interval + bisection)
# =========================================================

def _solve_in_bracket(a, b, f, tol=1e-2, max_iter=100):
    """
    Confined secant iteration inside [a,b] with robust fallbacks.
    Returns (root_approx, f(root_approx), iterations).
    """
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


# =========================================================
#        PUBLIC API: SHOOT FOR A GIVEN NODE NUMBER n
# =========================================================

def shoot_sigma0_for_mode(
    n, p_eqState, rho_eqState, xi, rho0, r0, r_max, frac_pc,
    *,
    bracket,                     # REQUIRED: (a,b) with a<b
    idx_p=0, idx_sigma=2,
    bracket_logspace=True,       # only used if a>0 and b>0 (else linear)
    bracket_npts=128,
    s6_tail_pts=9,
    s6_abs_tol=1e-12,            # solver tolerance when we *do* solve S6=0
    sigma_abs_tol=1e-12,         # acceptance: |σ∞| <= sigma_abs_tol
    sigma_rel_tol=1e-2,          # fallback acceptance: |σ∞|/|σ0| <= sigma_rel_tol
    max_iter=100,
    sigma0_min_abs=0.0,
    clear_cache_after=True,
):
    """
    S6 shooter with *article-style acceptance*:
      - primary: |σ∞| <= sigma_abs_tol
      - fallback: |σ∞|/|σ0| <= sigma_rel_tol
    where σ∞ is estimated via the sixth-order tail residual S6 (cancels 1/r^3 and 1/r^5).

    Steps per constant-n slice inside the user bracket:
      (A) Endpoint accept: if either endpoint already satisfies the acceptance checks, return it.
      (B) Otherwise, if S6 changes sign in the slice, solve S6(σ0)=0 (tol = s6_abs_tol),
          then accept the root only if it passes the same acceptance checks.
    No probe, no ratio-at-rmax, no global scan.

    Raises RuntimeError if nothing satisfies node==n *and* the acceptance logic.
    """
    import math
    import numpy as np
    try:
        # -------- helper with σ→−σ symmetry (full integration only) --------
        def _sol_signed(s0, *, rtol=1e-6, atol=1e-9, dense_output=True):
            sgn, s0_abs = _sigma_sign(s0)
            sol = _get_sol_cached(
                s0_abs, p_eqState, rho_eqState, xi, rho0, r0, r_max, frac_pc,
                rtol=rtol, atol=atol, dense_output=dense_output
            )
            return _mirror_sigma_sign(sol, sgn)

        def _verify_nodes_full(s0: float) -> int:
            sf = _sol_signed(s0, dense_output=True)
            return node_count_from_sol(sf, idx_sigma, idx_p, frac_pc=frac_pc, domain="to_2R")

        def _S6(s0: float) -> float:
            sf = _sol_signed(s0, dense_output=True)
            return sigma_infty_residual_S6(sf, k_tail=s6_tail_pts)

        def _accept(s0: float, S6_val: float) -> bool:
            if not math.isfinite(S6_val):
                return False
            if abs(S6_val) <= sigma_abs_tol:
                return True
            denom = max(abs(s0), sigma0_min_abs if sigma0_min_abs > 0.0 else 1.0)
            return (abs(S6_val) / denom) <= sigma_rel_tol

        # -------- bracket & sampling --------
        if bracket is None or len(bracket) != 2:
            raise RuntimeError("This shooter requires a user-supplied bracket=(a,b).")
        a_raw, b_raw = bracket
        a, b = (a_raw, b_raw) if a_raw < b_raw else (b_raw, a_raw)
        if abs(a) < sigma0_min_abs or abs(b) < sigma0_min_abs:
            raise RuntimeError("Bracket endpoints violate sigma0_min_abs.")

        xs = np.geomspace(a, b, int(bracket_npts)) if (bracket_logspace and a > 0.0 and b > 0.0) \
             else np.linspace(a, b, int(bracket_npts))

        # -------- node counts on samples; build constant-n slices --------
        nodes = np.empty(xs.size, dtype=float)
        for i, x in enumerate(xs):
            if abs(x) < sigma0_min_abs:
                nodes[i] = np.nan
                continue
            nodes[i] = _verify_nodes_full(x)

        const_n = [(xs[i], xs[i+1]) for i in range(xs.size - 1) if nodes[i] == n and nodes[i+1] == n]
        if not const_n:
            raise RuntimeError("No constant-n sub-interval inside the bracket.")

        # -------- try acceptance at endpoints, then root solve if needed --------
        for (pa, pb) in const_n:
            if abs(pa) >= sigma0_min_abs:
                Sa = _S6(pa)
                if _accept(pa, Sa):
                    # guard: recheck nodes to be safe
                    if _verify_nodes_full(pa) == n:
                        return pa
            if abs(pb) >= sigma0_min_abs:
                Sb = _S6(pb)
                if _accept(pb, Sb):
                    if _verify_nodes_full(pb) == n:
                        return pb

            # if S6 flips sign inside the slice, solve for root; then apply acceptance on the root
            Sa = _S6(pa) if 'Sa' not in locals() else Sa
            Sb = _S6(pb) if 'Sb' not in locals() else Sb
            if math.isfinite(Sa) and math.isfinite(Sb) and Sa * Sb < 0.0:
                root, fval, iters = _solve_in_bracket(pa, pb, _S6, tol=s6_abs_tol, max_iter=max_iter)
                # apply acceptance logic at the root (use the function value we already have: fval≈S6(root))
                if abs(root) >= sigma0_min_abs and _verify_nodes_full(root) == n and _accept(root, fval):
                    return root

        raise RuntimeError(
            "No σ0 in the bracket satisfies node==n with |σ∞|<=abs_tol or |σ∞|/|σ0|<=rel_tol."
        )

    finally:
        if clear_cache_after:
            _SOL_CACHE.clear()



