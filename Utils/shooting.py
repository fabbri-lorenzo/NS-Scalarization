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


def _candidate_ranges_from_grid(F, n_points, detect_near_zero, near_zero_factor):
    candidate_ranges = []
    for i in range(n_points - 1):
        f1, f2 = F[i], F[i + 1]
        if (
            np.isfinite(f1)
            and np.isfinite(f2)
            and (f1 == 0.0 or f2 == 0.0 or (f1 * f2 < 0.0))
        ):
            candidate_ranges.append((i, i + 1))

    if detect_near_zero:
        finite = np.isfinite(F)
        if np.any(finite):
            absF = np.abs(F[finite])
            med = float(np.median(absF))
            scale = med if med > 0.0 else float(np.nanmax(absF)) if absF.size else 1.0
            if scale <= 0.0:
                scale = 1.0
            thresh = near_zero_factor * scale / max(1, n_points // 10)

            for i in range(1, n_points - 1):
                f0, fm, fp = F[i], F[i - 1], F[i + 1]
                if not (np.isfinite(f0) and np.isfinite(fm) and np.isfinite(fp)):
                    continue
                if abs(f0) <= abs(fm) and abs(f0) <= abs(fp) and abs(f0) <= thresh:
                    candidate_ranges.append((i - 1, i + 1))
    return candidate_ranges


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
    n_coarse=201,
    target=0.0,
    parallel=False,
    max_workers=None,
):
    """
    Coarse-only diagnostic scan:
      - Build a coarse grid on [-b,-a] U [a,b]
      - Evaluate sigma_residual on the grid
      - Detect sign-change brackets on the coarse grid

    Returns a dict compatible with your existing caller:
      - "brackets": list[(u,v)] for Brent
      - "S_coarse","F_coarse"
      - "regions" empty
    """
    n_coarse = int(n_coarse)
    if n_coarse < 3:
        raise ValueError("n_coarse must be >= 3")

    a = float(a)
    b = float(b)
    if not (0.0 < a < b):
        raise ValueError("Require 0 < a < b for [-b,-a] U [a,b] scanning")

    # split points between negative and positive side
    n_pos = n_coarse // 2
    n_neg = n_coarse - n_pos

    S_pos = np.linspace(a, b, n_pos, endpoint=True)
    S_neg = np.linspace(-b, -a, n_neg, endpoint=True)

    F_pos = _eval_sigma_residuals_on_grid(
        S_pos,
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
    F_neg = _eval_sigma_residuals_on_grid(
        S_neg,
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

    # combine for diagnostics
    S_coarse = np.concatenate([S_neg, S_pos])
    F_coarse = np.concatenate([F_neg, F_pos])
    order = np.argsort(S_coarse)
    S_coarse = S_coarse[order]
    F_coarse = F_coarse[order]

    # detect brackets directly on coarse grid
    coarse_brackets_scored = _detect_brackets_from_grid(S_coarse, F_coarse)
    brackets = [(u, v) for (u, v, _score) in coarse_brackets_scored]
    brackets = sorted(brackets, key=lambda uv: 0.5 * (uv[0] + uv[1]))

    return {
        "brackets": brackets,
        "S_coarse": S_coarse,
        "F_coarse": F_coarse,
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

        # ----------------------------
        # Canonicalize to positive σ0
        # ----------------------------
        s0_phys = float(s_star)
        s0_pos = float(abs(s0_phys))

        if s0_phys < 0.0:
            # Recompute diagnostics at +|σ0| so (σ0, delta, sigma_end, t_eff) match.
            delta_p, sigma_end_p, t_eff_p = sigma_residual(
                s0_pos,
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
                method=method,
                idx_sigma=idx_sigma,
                target=target,
                return_details=True,
            )
            # If symmetry is truly exact, this should be valid.
            # Still guard numerically.
            if not np.isfinite(delta_p):
                custom_print(
                    f"[canon] σ0={s0_phys/M:.3e} → +|σ0| failed (NaN residual); keeping original",
                    color="yellow",
                )
                s0_use, delta_use, sigma_end_use, t_eff_use = (
                    s0_phys,
                    float(delta),
                    float(sigma_end),
                    float(t_eff),
                )
            else:
                s0_use, delta_use, sigma_end_use, t_eff_use = (
                    s0_pos,
                    float(delta_p),
                    float(sigma_end_p),
                    float(t_eff_p),
                )
        else:
            s0_use, delta_use, sigma_end_use, t_eff_use = (
                s0_pos,
                float(delta),
                float(sigma_end),
                float(t_eff),
            )

        # ----------------------------
        # Acceptance test (unchanged)
        # ----------------------------
        m0 = float(np.sqrt(max(0.0, m2_val)))
        m_vac_sq = float(m2_val + 2.0 * lmbda_val * (nu_val**2))
        uses_robin = ((nu_val == 0.0) and (m0 > 0.0)) or (
            (nu_val != 0.0) and (m_vac_sq > 0.0)
        )

        if uses_robin:
            accept = (abs(delta_use) <= abs_threshold) or (
                abs(s0_use) > 0.0 and abs(delta_use) / abs(s0_use) <= tol_relative
            )
            reason = "robin"
        else:
            accept, reason = residual_accept(
                sigma_rmax=sigma_end_use,
                sigma0=s0_use,
                target=t_eff_use,
                abs_threshold=abs_threshold,
                tol_relative=tol_relative,
            )

        # NOTE: psi2 should be computed at the *canonical* σ0 you will store
        _, psi2 = _sigma2_psi2(s0_use, p0, eps0, xi_val, m2_val, lmbda_val, nu_val)
        psi_ok = np.isfinite(psi2) and (psi2 > 0.0)

        if accept and psi_ok:
            custom_print(
                f"ACCEPT σ0/M={s0_use/M:.6e}  |residual|/M={abs(delta_use)/M:.3e}  Ψ2={psi2:.3e}  via={reason}",
                color="green",
            )
            candidates.append(float(s0_use))  # <-- always positive now
            residuals.append(float(abs(delta_use)))
            sigma_ends.append(float(sigma_end_use))
            if nu_val != 0.0:
                vac_signs.append(1 if t_eff_use >= 0 else -1)
            else:
                vac_signs.append(0)
        else:
            why = []
            if not accept:
                why.append("residual too large")
            if not psi_ok:
                why.append(f"Ψ2≤0 (Ψ2={psi2:.3e})")
            custom_print(
                f"REJECT σ0/M={s0_use/M:.6e}  |residual|/M={abs(delta_use)/M:.3e} → {'; '.join(why) if why else 'unknown'}",
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
            group_sorted = sorted(group, key=lambda x: x[0])
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

    order = np.argsort(candidates)
    s0_sorted = [candidates[i] for i in order]
    sigma_end_sorted = [sigma_ends[i] for i in order]
    return s0_sorted, sigma_end_sorted
