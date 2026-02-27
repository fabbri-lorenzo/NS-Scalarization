import numpy as np
from Utils.params import c, M
from Utils.TOV import _sigma2_psi2
from Utils.analysis import integrate_star
from Utils.graphics_single import custom_print
from scipy.optimize import brentq
from concurrent.futures import ThreadPoolExecutor
import os


def _sigma_residual_worker(args):
    """Worker for parallel evaluation of sigma_residual across a grid.

    Parameters are packed to satisfy the pickler used by ThreadPoolExecutor.
    On failure the worker returns NaN for the residual.
    """
    (
        s,
        r0,
        r_max,
        p_eqState,
        rho_eqState,
        xi,
        m2,
        lmbda,
        nu_val,
        rho0,
        frac_pc,
        method,
        idx_sigma,
        target,
    ) = args
    try:
        return sigma_residual(
            float(s),
            r0,
            r_max,
            p_eqState,
            rho_eqState,
            xi,
            m2,
            lmbda,
            nu_val,
            rho0,
            frac_pc,
            method,
            idx_sigma=idx_sigma,
            target=target,
        )
    except Exception:
        return np.nan


def _bracket_root_worker(args):
    """Worker for parallel evaluation of find_root_brent across brackets.
    This simply wraps find_root_brent with its arguments.
    """
    (
        u,
        v,
        r0,
        r_max,
        p_eqState,
        rho_eqState,
        xi_val,
        m2_val,
        lmbda_val,
        nu_val,
        rho0,
        frac_pc,
        method,
        idx_sigma,
        target,
    ) = args
    return find_root_brent(
        u,
        v,
        r0,
        r_max,
        p_eqState,
        rho_eqState,
        xi_val,
        m2_val,
        lmbda_val,
        nu_val,
        rho0,
        frac_pc,
        method,
        idx_sigma=idx_sigma,
        target=target,
    )


def sigma_residual(
    s0,
    r0,
    r_max,
    p_eqState,
    rho_eqState,
    xi,
    m2,
    lmbda,
    nu,
    rho0,
    frac_pc,
    method,
    idx_sigma=2,
    idx_sigma_p=3,
    target=0.0,
    *,
    return_details=False,
):
    """
    Residual used for shooting, chosen by asymptotic regime:

      1) Massive around zero (nu == 0, m > 0):
         Robin on sigma:
            sigma' + (m + 1/r) sigma = 0

      2) Shifted vacuum / double-well (nu != 0):
         Robin on delta sigma = sigma - sigma_inf, where sigma_inf = nearest target in {+|nu|,-|nu|}:
            sigma' + (m_vac + 1/r) (sigma - sigma_inf) = 0
         with m_vac = sqrt( V''(sigma_inf) ) = sqrt( m2 + 2 lambda nu^2 )

      3) Massless pure quartic (nu == 0, m == 0):
         fallback to target matching sigma(r_max) -> target (typically 0)
    """
    sol, _, _ = integrate_star(
        s0,
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
        method=method,
        stop_at_2r=False,
        record_mu2=False,
    )

    sigma_end = float(sol.y[idx_sigma, -1])
    sp_end = float(sol.y[idx_sigma_p, -1])

    # physical "bare" mass around sigma=0
    m0 = float(np.sqrt(max(0.0, m2)))

    # ------------------------------------------------------------------
    # (A) Massive around zero: nu == 0 and m0 > 0
    # ------------------------------------------------------------------
    if (nu == 0.0) and (m0 > 0.0):
        delta = sp_end + (m0 + 1.0 / r_max) * sigma_end
        t_eff = 0.0
        return (delta, sigma_end, t_eff) if return_details else delta

    # ------------------------------------------------------------------
    # (B) Shifted vacuum (double-well-type asymptotics): nu != 0
    #     Apply Robin to delta sigma = sigma - sigma_inf
    # ------------------------------------------------------------------
    if nu != 0.0:
        # choose the asymptotic vacuum branch (+|nu| or -|nu|) closest to sigma_end
        t_eff = _nearest_target_value(sigma_end, target)  # usually target=[+|nu|,-|nu|]

        # asymptotic mass around the vacuum:
        # V = 1/2 m2 sigma^2 + lambda/4 (sigma^2 - nu^2)^2
        # V''(±nu) = m2 + 2 lambda nu^2
        m_vac_sq = float(m2 + 2.0 * lmbda * (nu**2))

        # If m_vac^2 <= 0, Yukawa Robin is not valid (tachyonic/critical tail);
        # fallback to target matching.
        if m_vac_sq > 0.0:
            m_vac = float(np.sqrt(m_vac_sq))
            delta = sp_end + (m_vac + 1.0 / r_max) * (sigma_end - t_eff)
            return (delta, sigma_end, t_eff) if return_details else delta

        # fallback if no positive asymptotic mass
        delta = sigma_end - t_eff
        return (delta, sigma_end, t_eff) if return_details else delta

    # ------------------------------------------------------------------
    # (C) Massless pure quartic / generic non-Yukawa tail fallback
    # ------------------------------------------------------------------
    t_eff = _nearest_target_value(sigma_end, target)
    delta = sigma_end - t_eff
    return (delta, sigma_end, t_eff) if return_details else delta


def _eval_sigma_residuals_on_grid(
    S,
    r0,
    r_max,
    p_eqState,
    rho_eqState,
    xi,
    m2,
    lmbda,
    nu_val,
    rho0,
    frac_pc,
    method,
    target=0.0,
    *,
    parallel=False,
    max_workers=None,
    idx_sigma=2,
):
    """Evaluate sigma_residual on an array of s0 values, returning F.

    If ``parallel`` is True, this will evaluate the residuals concurrently
    using a ThreadPoolExecutor. Each call to ``sigma_residual`` is independent,
    so the speedup can be significant when the integrator is expensive.
    On failure the residual value is set to NaN.
    """
    if not parallel:
        F = np.empty(len(S), dtype=float)
        for i, s in enumerate(S):
            try:
                F[i] = sigma_residual(
                    float(s),
                    r0,
                    r_max,
                    p_eqState,
                    rho_eqState,
                    xi,
                    m2,
                    lmbda,
                    nu_val,
                    rho0,
                    frac_pc,
                    method,
                    idx_sigma=idx_sigma,
                    target=target,
                )
            except Exception:
                F[i] = np.nan
        return F
    # parallel case
    tasks = [
        (
            float(s),
            r0,
            r_max,
            p_eqState,
            rho_eqState,
            xi,
            m2,
            lmbda,
            nu_val,
            rho0,
            frac_pc,
            method,
            idx_sigma,
            target,
        )
        for s in S
    ]
    F = np.empty(len(S), dtype=float)
    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        for i, res in enumerate(executor.map(_sigma_residual_worker, tasks)):
            F[i] = res
    return F


def _detect_brackets_from_grid(S, F):
    """
    Return sign-change brackets from sampled grid points.
    Each bracket is (u, v, score), where score helps rank/compress:
      score = min(|f(u)|, |f(v)|)  (lower is usually better)
    """
    brackets = []
    for i in range(len(S) - 1):
        f1, f2 = F[i], F[i + 1]
        if (
            np.isfinite(f1)
            and np.isfinite(f2)
            and (f1 == 0.0 or f2 == 0.0 or (f1 * f2 < 0.0))
        ):
            score = float(min(abs(f1), abs(f2)))
            brackets.append((float(S[i]), float(S[i + 1]), score))
    return brackets


def _merge_index_ranges(ranges, n_points, expand_points=0):
    """
    Merge list of (i0, i1) point-index ranges (inclusive) into disjoint ranges.
    Expands each range by expand_points before merging.
    """
    if not ranges:
        return []

    out = []
    clipped = []
    for i0, i1 in ranges:
        a = max(0, int(i0) - int(expand_points))
        b = min(n_points - 1, int(i1) + int(expand_points))
        if a > b:
            a, b = b, a
        clipped.append((a, b))

    clipped.sort(key=lambda x: x[0])
    cur_a, cur_b = clipped[0]
    for a, b in clipped[1:]:
        if a <= cur_b + 1:
            cur_b = max(cur_b, b)
        else:
            out.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    out.append((cur_a, cur_b))
    return out


def _compress_brackets_by_center(brackets, cluster_tol):
    """
    Compress near-duplicate brackets by clustering on bracket center.
    Keeps the bracket with the lowest score in each cluster.

    Input brackets: list of (u, v, score)
    Output: list of (u, v)
    """
    if not brackets:
        return []

    # Sort by center
    items = sorted(brackets, key=lambda b: 0.5 * (b[0] + b[1]))

    clusters = []
    cur = [items[0]]
    cur_center = 0.5 * (items[0][0] + items[0][1])

    for b in items[1:]:
        c = 0.5 * (b[0] + b[1])
        if abs(c - cur_center) <= cluster_tol:
            cur.append(b)
            # update representative center (mean center)
            cur_center = np.mean([0.5 * (x[0] + x[1]) for x in cur])
        else:
            clusters.append(cur)
            cur = [b]
            cur_center = c
    clusters.append(cur)

    # Keep best score from each cluster
    out = []
    for cl in clusters:
        best = min(cl, key=lambda x: x[2])  # smallest score
        out.append((best[0], best[1]))

    return out


def diagnostic_scan(
    r0,
    r_max,
    p_eqState,
    rho_eqState,
    xi,
    m2,
    lmbda,
    nu_val,
    rho0,
    frac_pc,
    method,
    a,
    b,
    *,
    n_coarse=41,
    n_refine=81,
    target=0.0,
    expand_coarse_points=1,
    detect_near_zero=True,
    near_zero_factor=5.0,
    compress_brackets=True,
    compress_tol=None,
    parallel=False,
    max_workers=None,
):
    """
    Multi-resolution diagnostic scan:
      1) Coarse scan over [a,b]
      2) Identify promising regions (sign changes + optional near-zero local minima)
      3) Refine only those regions
      4) Return refined brackets (optionally compressed)

    Parameters
    ----------
    parallel : bool, optional
        If True, evaluates the residuals on the coarse and refined grids concurrently.
        This can reduce wall-clock time when ``sigma_residual`` is expensive.
    max_workers : int or None, optional
        Maximum number of worker threads used when ``parallel`` is True.

    Returns
    -------
    result : dict with keys
      - "brackets": list[(u, v)] for Brent
      - "S_coarse", "F_coarse"
      - "S_refined", "F_refined"   (concatenated refined samples, sorted, unique-ish)
      - "regions": list[(s_left, s_right)] refined regions
    """
    n_coarse = int(n_coarse)
    n_refine = int(n_refine)
    if n_coarse < 3:
        raise ValueError("n_coarse must be >= 3")
    if n_refine < 3:
        raise ValueError("n_refine must be >= 3")

    # --- 1) coarse scan ---
    S1 = np.geomspace(a, min(b, 1e-2 * M), n_coarse // 2)
    S2 = np.linspace(min(b, 1e-2 * M), b, n_coarse - len(S1))
    S_coarse = np.unique(np.concatenate([S1, S2]))
    F_coarse = _eval_sigma_residuals_on_grid(
        S_coarse,
        r0,
        r_max,
        p_eqState,
        rho_eqState,
        xi,
        m2,
        lmbda,
        nu_val,
        rho0,
        frac_pc,
        method,
        target=target,
        parallel=parallel,
        max_workers=max_workers,
        idx_sigma=2,
    )

    # --- 2) find promising coarse ranges (as point-index ranges) ---
    candidate_ranges = []

    # (i) Sign-change coarse cells
    for i in range(n_coarse - 1):
        f1, f2 = F_coarse[i], F_coarse[i + 1]
        if (
            np.isfinite(f1)
            and np.isfinite(f2)
            and (f1 == 0.0 or f2 == 0.0 or (f1 * f2 < 0.0))
        ):
            candidate_ranges.append((i, i + 1))

    # (ii) Optional: near-zero local minima to catch missed sign flips at coarse resolution
    if detect_near_zero:
        finite = np.isfinite(F_coarse)
        if np.any(finite):
            absF = np.abs(F_coarse[finite])
            med = float(np.median(absF))
            # fallback if median is tiny/zero
            scale = med if med > 0.0 else float(np.nanmax(absF)) if absF.size else 1.0
            if scale <= 0.0:
                scale = 1.0
            thresh = near_zero_factor * scale / max(1, n_coarse // 10)

            # local minima in |F|
            for i in range(1, n_coarse - 1):
                f0, fm, fp = F_coarse[i], F_coarse[i - 1], F_coarse[i + 1]
                if not (np.isfinite(f0) and np.isfinite(fm) and np.isfinite(fp)):
                    continue
                af0, afm, afp = abs(f0), abs(fm), abs(fp)
                if af0 <= afm and af0 <= afp and af0 <= thresh:
                    # refine around this minimum (one cell on each side)
                    candidate_ranges.append((i - 1, i + 1))

    # If nothing looks promising, return early
    if not candidate_ranges:
        return {
            "brackets": [],
            "S_coarse": S_coarse,
            "F_coarse": F_coarse,
            "S_refined": np.array([], dtype=float),
            "F_refined": np.array([], dtype=float),
            "regions": [],
        }

    # Merge/expand candidate coarse ranges
    merged_ranges_idx = _merge_index_ranges(
        candidate_ranges,
        n_points=n_coarse,
        expand_points=expand_coarse_points,
    )

    # --- 3) refine only merged regions ---
    refined_brackets_scored = []
    S_refined_all = []
    F_refined_all = []
    regions = []

    for i0, i1 in merged_ranges_idx:
        sL = float(S_coarse[i0])
        sR = float(S_coarse[i1])
        if sR <= sL:
            continue

        S_loc = np.linspace(sL, sR, n_refine)
        F_loc = _eval_sigma_residuals_on_grid(
            S_loc,
            r0,
            r_max,
            p_eqState,
            rho_eqState,
            xi,
            m2,
            lmbda,
            nu_val,
            rho0,
            frac_pc,
            method,
            target=target,
            parallel=parallel,
            max_workers=max_workers,
            idx_sigma=2,
        )

        local_brackets = _detect_brackets_from_grid(S_loc, F_loc)
        refined_brackets_scored.extend(local_brackets)

        S_refined_all.append(S_loc)
        F_refined_all.append(F_loc)
        regions.append((sL, sR))

    if S_refined_all:
        S_refined = np.concatenate(S_refined_all)
        F_refined = np.concatenate(F_refined_all)
        # sort for nicer diagnostics
        order = np.argsort(S_refined)
        S_refined = S_refined[order]
        F_refined = F_refined[order]
    else:
        S_refined = np.array([], dtype=float)
        F_refined = np.array([], dtype=float)

    # --- 4) optional bracket compression (fewer Brent calls) ---
    if compress_brackets and refined_brackets_scored:
        coarse_step = (
            abs(S_coarse[1] - S_coarse[0]) if len(S_coarse) > 1 else abs(b - a)
        )
        tol = float(compress_tol) if (compress_tol is not None) else 0.35 * coarse_step
        brackets = _compress_brackets_by_center(
            refined_brackets_scored, cluster_tol=tol
        )
    else:
        brackets = [(u, v) for (u, v, _score) in refined_brackets_scored]

    # Final sort by |center| or by center; center is usually more intuitive
    brackets = sorted(brackets, key=lambda uv: 0.5 * (uv[0] + uv[1]))

    return {
        "brackets": brackets,
        "S_coarse": S_coarse,
        "F_coarse": F_coarse,
        "S_refined": S_refined,
        "F_refined": F_refined,
        "regions": regions,
    }


def find_root_brent(
    u,
    v,
    r0,
    r_max,
    p_eqState,
    rho_eqState,
    xi,
    m2,
    lmbda,
    nu,
    rho0,
    frac_pc,
    method,
    idx_sigma: int = 2,
    rtol: float = 1e-12,
    maxiter: int = 200,
    target=0.0,
):
    def f(s0):
        return sigma_residual(
            s0,
            r0,
            r_max,
            p_eqState,
            rho_eqState,
            xi,
            m2,
            lmbda,
            nu,
            rho0,
            frac_pc,
            method=method,
            idx_sigma=idx_sigma,
            target=target,
        )

    try:
        s_star = brentq(f, float(u), float(v), rtol=rtol, maxiter=maxiter)
        # one more eval to get (sigma_end, t_eff) without a second integrate later
        delta, sigma_end, t_eff = sigma_residual(
            s_star,
            r0,
            r_max,
            p_eqState,
            rho_eqState,
            xi,
            m2,
            lmbda,
            nu,
            rho0,
            frac_pc,
            method=method,
            idx_sigma=idx_sigma,
            target=target,
            return_details=True,
        )
        return float(s_star), float(delta), float(sigma_end), float(t_eff)
    except Exception:
        return None, None, None, None


def residual_accept(sigma_rmax, sigma0, target, abs_threshold, tol_relative):
    """
    Check whether the residual at r_max is acceptable given the target.

    - If target == 0: use |σ(r_max)| / |σ0| < tol_relative
    - If target != 0: use |σ(r_max) - target| / |target| < tol_relative
    Absolute threshold always applies first.
    """
    # absolute check
    if abs(sigma_rmax - target) <= abs_threshold:
        return True, "abs"

    # relative check
    if target == 0.0:
        if sigma0 != 0.0 and abs(sigma_rmax) / abs(sigma0) <= tol_relative:
            return True, "rel"
    else:
        if abs((sigma_rmax - target) / target) <= tol_relative:
            return True, "rel"

    return False, None


def _nearest_target_value(sigma_rmax, target):
    """Return the scalar target t* in `target` closest to sigma_rmax."""
    try:
        return float(target)  # scalar target
    except (TypeError, ValueError):
        arr = np.asarray(target, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("target list must be non-empty and finite")
        idx = np.nanargmin(np.abs(arr - sigma_rmax))
        return float(arr[idx])


def probe_brackets(
    brackets,
    r0,
    r_max,
    p_eqState,
    rho_eqState,
    xi_val,
    rho0,
    frac_pc,
    method,
    m2_val,
    lmbda_val,
    nu_val,
    abs_threshold,
    tol_relative,
    merge_tol,
    target=0.0,
    idx_sigma: int = 2,
    *,
    parallel=False,
    max_workers=None,
):
    """
    Probe each sign-change bracket and return (s0_list, sigma_end_list), where
    sigma_end_list contains σ(r_max) (not residuals). Near-duplicates are
    merged within each vacuum branch when ν≠0.

    Parameters
    ----------
    parallel : bool, optional
        If True, find_root_brent is invoked concurrently for each bracket.
    max_workers : int or None, optional
        Maximum number of worker threads used when ``parallel`` is True.
    """
    candidates = []
    residuals = []  # |σ(r_max) - t_eff|
    sigma_ends = []  # σ(r_max)
    vac_signs = []

    p0 = float(p_eqState(rho0))
    eps0 = float(rho0 * c * c)

    # Helper to process the result for a single bracket
    def process_result(u, v, s_star, delta, sigma_end, t_eff):
        if s_star is None or (not np.isfinite(delta)):
            custom_print(
                f"[brent] bracket [{u/M:.3e},{v/M:.3e}] → FAILED", color="yellow"
            )
            return
        m0 = float(np.sqrt(max(0.0, m2_val)))
        m_vac_sq = float(m2_val + 2.0 * lmbda_val * (nu_val**2))

        uses_robin = ((nu_val == 0.0) and (m0 > 0.0)) or (
            (nu_val != 0.0) and (m_vac_sq > 0.0)
        )

        if uses_robin:
            # accept based on Robin residual (delta)
            accept = (abs(delta) <= abs_threshold) or (
                abs(s_star) > 0.0 and abs(delta) / abs(s_star) <= tol_relative
            )
            reason = "robin"
        else:
            # fallback (pure quartic/log tail, or non-Yukawa asymptotics)
            accept, reason = residual_accept(
                sigma_rmax=sigma_end,
                sigma0=s_star,
                target=t_eff,
                abs_threshold=abs_threshold,
                tol_relative=tol_relative,
            )

        _, psi2 = _sigma2_psi2(s_star, p0, eps0, xi_val, m2_val, lmbda_val, nu_val)
        psi_ok = np.isfinite(psi2) and (psi2 > 0.0)

        if accept and psi_ok:
            custom_print(
                f"ACCEPT σ0/M={s_star/M:.6e}  |residual|/M={abs(delta)/M:.3e}  Ψ2={psi2:.3e}  via={reason}",
                color="green",
            )
            candidates.append(float(s_star))
            residuals.append(float(abs(delta)))
            sigma_ends.append(float(sigma_end))
            if nu_val != 0.0:
                vac_signs.append(1 if t_eff >= 0 else -1)
            else:
                vac_signs.append(0)
        else:
            why = []
            if not accept:
                why.append("residual too large")
            if not psi_ok:
                why.append(f"Ψ2≤0 (Ψ2={psi2:.3e})")
            custom_print(
                f"REJECT σ0/M={s_star/M:.6e}  |residual|/M={abs(delta)/M:.3e} → {'; '.join(why) if why else 'unknown'}",
                color="red",
            )

    if not parallel:
        # serial evaluation
        for u, v in brackets:
            s_star, delta, sigma_end, t_eff = find_root_brent(
                u,
                v,
                r0,
                r_max,
                p_eqState,
                rho_eqState,
                xi_val,
                m2_val,
                lmbda_val,
                nu_val,
                rho0,
                frac_pc,
                method,
                idx_sigma=idx_sigma,
                target=target,
            )
            process_result(u, v, s_star, delta, sigma_end, t_eff)
    else:
        # parallel evaluation: dispatch each bracket concurrently
        tasks = [
            (
                u,
                v,
                r0,
                r_max,
                p_eqState,
                rho_eqState,
                xi_val,
                m2_val,
                lmbda_val,
                nu_val,
                rho0,
                frac_pc,
                method,
                idx_sigma,
                target,
            )
            for (u, v) in brackets
        ]
        with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
            for (u, v), result in zip(
                brackets, executor.map(_bracket_root_worker, tasks)
            ):
                s_star, delta, sigma_end, t_eff = result
                process_result(u, v, s_star, delta, sigma_end, t_eff)

    if merge_tol is not None and merge_tol > 0 and len(candidates) > 1:
        grouped = {}
        for s0, res, se, vs in zip(candidates, residuals, sigma_ends, vac_signs):
            grouped.setdefault(vs, []).append((s0, res, se))

        merged_c, merged_r, merged_se, merged_v = [], [], [], []
        for vs, group in grouped.items():
            group_sorted = sorted(group, key=lambda x: abs(x[0]))
            cur_s0, cur_r, cur_se = group_sorted[0]
            for s0, r0_c, se in group_sorted[1:]:
                if abs(s0 - cur_s0) <= merge_tol:
                    if r0_c < cur_r:
                        cur_s0, cur_r, cur_se = s0, r0_c, se
                else:
                    merged_c.append(cur_s0)
                    merged_r.append(cur_r)
                    merged_se.append(cur_se)
                    merged_v.append(vs)
                    cur_s0, cur_r, cur_se = s0, r0_c, se
            merged_c.append(cur_s0)
            merged_r.append(cur_r)
            merged_se.append(cur_se)
            merged_v.append(vs)

        candidates, residuals, sigma_ends, vac_signs = (
            merged_c,
            merged_r,
            merged_se,
            merged_v,
        )

    if not candidates:
        return [], []

    order = np.argsort(np.abs(candidates))
    s0_sorted = [candidates[i] for i in order]
    sigma_end_sorted = [sigma_ends[i] for i in order]
    return s0_sorted, sigma_end_sorted
