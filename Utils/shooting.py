"""Utils.shooting

Adaptive diagnostic scans + shooting utilities.

This file was updated to:
  1) restore the full implementation of :func:`probe_brackets` (it was a stub)
  2) provide a robust wall-clock timeout wrapper for :func:`diagnostic_scan`
     that works on macOS/Windows (spawn start method) by avoiding non-picklable
     local functions/closures.

The timeout wrapper runs the scan in a separate process and hard-terminates
that process if it exceeds the requested limit.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from Utils.analysis import integrate_star
from Utils.graphics_single import custom_print
from Utils.params import M, c
from Utils.TOV import _sigma2_psi2


def sigma_residual(
    s0: float,
    r0: float,
    r_max: float,
    p_eqState: Callable[[float], float],
    rho_eqState: Callable[[float], float],
    xi: float,
    m2: float,
    lmbda: float,
    nu: float,
    rho0: float,
    frac_pc: float,
    method: str,
    idx_sigma: int = 2,
    idx_sigma_p: int = 3,
    target: float | Sequence[float] = 0.0,
    *,
    return_details: bool = False,
):
    """Compute the shooting residual at r_max.

    Returns either a float residual, or (residual, sigma_end, t_eff)
    if ``return_details`` is True.
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

    m0 = float(np.sqrt(max(0.0, m2)))

    # (A) Massive around zero vacuum -> use Robin residual
    if (nu == 0.0) and (m0 > 0.0):
        delta = sp_end + (m0 + 1.0 / r_max) * sigma_end
        t_eff = 0.0
        return (delta, sigma_end, t_eff) if return_details else delta

    # (B) Shifted vacuum (nu != 0)
    if nu != 0.0:
        # choose the closest target branch
        try:
            t_eff = float(target)
        except (TypeError, ValueError):
            arr = np.asarray(target, dtype=float)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                raise ValueError("target list must be non-empty and finite")
            t_eff = float(arr[np.nanargmin(np.abs(arr - sigma_end))])

        m_vac_sq = float(m2 + 2.0 * lmbda * (nu**2))
        if m_vac_sq > 0.0:
            m_vac = float(np.sqrt(m_vac_sq))
            delta = sp_end + (m_vac + 1.0 / r_max) * (sigma_end - t_eff)
            return (delta, sigma_end, t_eff) if return_details else delta

        # tachyonic/flat vacuum: fall back to Dirichlet
        delta = sigma_end - t_eff
        return (delta, sigma_end, t_eff) if return_details else delta

    # (C) Massless and nu=0: Dirichlet residual to target
    try:
        t_eff = float(target)
    except (TypeError, ValueError):
        arr = np.asarray(target, dtype=float)
        t_eff = float(arr[np.nanargmin(np.abs(arr - sigma_end))])
    delta = sigma_end - t_eff
    return (delta, sigma_end, t_eff) if return_details else delta


def residual_accept(
    sigma_rmax: float,
    sigma0: float,
    target: float,
    abs_threshold: float,
    tol_relative: float,
) -> Tuple[bool, str | None]:
    """Acceptance rule for Dirichlet residuals."""

    if abs(sigma_rmax - target) <= abs_threshold:
        return True, "abs"

    if target == 0.0:
        if sigma0 != 0.0 and abs(sigma_rmax) / abs(sigma0) <= tol_relative:
            return True, "rel"
    else:
        if abs((sigma_rmax - target) / target) <= tol_relative:
            return True, "rel"

    return False, None


def _eval_sigma_residuals_on_grid(
    S: np.ndarray,
    r0: float,
    r_max: float,
    p_eqState: Callable[[float], float],
    rho_eqState: Callable[[float], float],
    xi: float,
    m2: float,
    lmbda: float,
    nu_val: float,
    rho0: float,
    frac_pc: float,
    method: str,
    target: float | Sequence[float],
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    idx_sigma: int = 2,
) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    F = np.empty_like(S, dtype=float)

    def worker(s0: float) -> float:
        try:
            return float(
                sigma_residual(
                    float(s0),
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
            )
        except Exception:
            return math.nan

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers or (os.cpu_count() or 8)) as ex:
            for i, val in enumerate(ex.map(worker, S)):
                F[i] = val
    else:
        for i, s0 in enumerate(S):
            F[i] = worker(float(s0))
    return F


def _detect_brackets_from_grid(
    S: np.ndarray, F: np.ndarray
) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for i in range(len(S) - 1):
        u, v = float(S[i]), float(S[i + 1])
        f1, f2 = float(F[i]), float(F[i + 1])
        if not (math.isfinite(f1) and math.isfinite(f2)):
            continue
        if f1 * f2 < 0.0:
            score = float(min(abs(f1), abs(f2)))
            out.append((u, v, score))
    return out


def _candidate_ranges_from_grid(
    F: np.ndarray,
    *,
    detect_near_zero: bool,
    near_zero_factor: float,
) -> List[Tuple[int, int]]:
    """Return index ranges on a positive grid to refine."""
    n = len(F)
    ranges: List[Tuple[int, int]] = []

    # (i) sign-change cells
    for i in range(n - 1):
        f1, f2 = float(F[i]), float(F[i + 1])
        if math.isfinite(f1) and math.isfinite(f2) and (f1 * f2 < 0.0):
            ranges.append((i, i + 1))

    # (ii) near-zero local minima (tangential roots)
    if detect_near_zero:
        finite = np.isfinite(F)
        if np.any(finite):
            absF = np.abs(F[finite])
            med = float(np.median(absF))
            scale = med if med > 0.0 else float(np.nanmax(absF)) if absF.size else 1.0
            if not math.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            thresh = near_zero_factor * scale / max(1, n // 10)

            for i in range(1, n - 1):
                f0, fm, fp = float(F[i]), float(F[i - 1]), float(F[i + 1])
                if not (math.isfinite(f0) and math.isfinite(fm) and math.isfinite(fp)):
                    continue
                af0, afm, afp = abs(f0), abs(fm), abs(fp)
                if (af0 <= afm) and (af0 <= afp) and (af0 <= thresh):
                    ranges.append((i - 1, i + 1))

    return ranges


def _merge_index_ranges(
    ranges: List[Tuple[int, int]],
    *,
    n_points: int,
    expand_points: int,
) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    exp = []
    for i0, i1 in ranges:
        a = max(0, i0 - expand_points)
        b = min(n_points - 1, i1 + expand_points)
        if a > b:
            a, b = b, a
        exp.append((a, b))
    exp.sort(key=lambda r: r[0])

    merged = []
    cur_a, cur_b = exp[0]
    for a, b in exp[1:]:
        if a <= cur_b + 1:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    return merged


def _compress_brackets_by_center(
    brackets_scored: List[Tuple[float, float, float]],
    *,
    cluster_tol: float,
) -> List[Tuple[float, float]]:
    if not brackets_scored:
        return []
    items = sorted(brackets_scored, key=lambda b: 0.5 * (b[0] + b[1]))
    out: List[Tuple[float, float]] = []

    cur_cluster = [items[0]]
    cur_center = 0.5 * (items[0][0] + items[0][1])

    def flush(cluster: List[Tuple[float, float, float]]):
        best = min(cluster, key=lambda x: x[2])
        out.append((best[0], best[1]))

    for b in items[1:]:
        c = 0.5 * (b[0] + b[1])
        if abs(c - cur_center) <= cluster_tol:
            cur_cluster.append(b)
            cur_center = float(np.mean([0.5 * (x[0] + x[1]) for x in cur_cluster]))
        else:
            flush(cur_cluster)
            cur_cluster = [b]
            cur_center = c
    flush(cur_cluster)

    return out


def diagnostic_scan(
    r0: float,
    r_max: float,
    p_eqState: Callable[[float], float],
    rho_eqState: Callable[[float], float],
    xi: float,
    m2: float,
    lmbda: float,
    nu_val: float,
    rho0: float,
    frac_pc: float,
    method: str,
    a: float,
    b: float,
    *,
    n_coarse: int = 41,
    n_refine: int = 81,
    target: float | Sequence[float] = 0.0,
    expand_coarse_points: int = 1,
    detect_near_zero: bool = True,
    near_zero_factor: float = 5.0,
    parallel: bool = False,
    max_workers: int | None = None,
    compress_brackets: bool = True,
    compress_tol: float | None = None,
    # soft budget (optional; does NOT hard-stop inside an integrate_star call)
    time_budget_sec: float | None = None,
) -> dict:
    """Multi-resolution scan on the positive domain [a,b], with a>0."""

    t_start = time.monotonic()
    n_coarse = int(n_coarse)
    n_refine = int(n_refine)
    if n_coarse < 3:
        raise ValueError("n_coarse must be >= 3")
    if n_refine < 3:
        raise ValueError("n_refine must be >= 3")

    a = float(a)
    b = float(b)
    if not (0.0 < a < b):
        raise ValueError("Require 0 < a < b for positive-only scanning")

    def _check_budget():
        if time_budget_sec is None:
            return
        if (time.monotonic() - t_start) > float(time_budget_sec):
            raise TimeoutError(
                f"diagnostic_scan exceeded time_budget_sec={float(time_budget_sec):.1f}"
            )

    # --- 1) coarse scan ---
    _check_budget()
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
    _check_budget()

    # --- 2) candidate coarse ranges ---
    cand = _candidate_ranges_from_grid(
        F_coarse,
        detect_near_zero=detect_near_zero,
        near_zero_factor=near_zero_factor,
    )
    if not cand:
        return {
            "brackets": [],
            "S_coarse": S_coarse,
            "F_coarse": F_coarse,
            "S_refined": np.array([], dtype=float),
            "F_refined": np.array([], dtype=float),
            "regions": [],
        }

    merged = _merge_index_ranges(
        cand, n_points=len(S_coarse), expand_points=int(expand_coarse_points)
    )

    # --- 3) refine only merged regions ---
    refined_scored: List[Tuple[float, float, float]] = []
    S_refined_all: List[np.ndarray] = []
    F_refined_all: List[np.ndarray] = []
    regions: List[Tuple[float, float]] = []

    for i0, i1 in merged:
        _check_budget()
        sL = float(S_coarse[i0])
        sR = float(S_coarse[i1])
        if sR <= sL:
            continue
        S_loc = np.linspace(sL, sR, n_refine, endpoint=True)
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
        refined_scored.extend(_detect_brackets_from_grid(S_loc, F_loc))
        S_refined_all.append(S_loc)
        F_refined_all.append(F_loc)
        regions.append((sL, sR))

    if S_refined_all:
        S_refined = np.concatenate(S_refined_all)
        F_refined = np.concatenate(F_refined_all)
        order = np.argsort(S_refined)
        S_refined = S_refined[order]
        F_refined = F_refined[order]
    else:
        S_refined = np.array([], dtype=float)
        F_refined = np.array([], dtype=float)

    # --- 4) bracket compression ---
    if compress_brackets and refined_scored:
        step = (
            float(abs(S_coarse[1] - S_coarse[0])) if len(S_coarse) > 1 else float(b - a)
        )
        tol = float(compress_tol) if compress_tol is not None else 0.35 * step
        brackets = _compress_brackets_by_center(refined_scored, cluster_tol=tol)
    else:
        brackets = [(u, v) for (u, v, _s) in refined_scored]

    brackets = sorted(brackets, key=lambda uv: 0.5 * (uv[0] + uv[1]))

    return {
        "brackets": brackets,
        "S_coarse": S_coarse,
        "F_coarse": F_coarse,
        "S_refined": S_refined,
        "F_refined": F_refined,
        "regions": regions,
    }


def _diagnostic_scan_worker(q, kwargs: dict) -> None:
    """Entry point for the scan subprocess (must be top-level for spawn)."""
    try:
        res = diagnostic_scan(**kwargs)
        q.put(("ok", res))
    except Exception as e:
        tb = traceback.format_exc()
        q.put(("err", (repr(e), tb)))


def diagnostic_scan_with_timeout(timeout_sec: float, /, **kwargs) -> dict:
    """Run :func:`diagnostic_scan` in a separate process with a hard timeout."""
    timeout_sec = float(timeout_sec)
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be > 0")

    ctx = get_context("spawn")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_diagnostic_scan_worker, args=(q, kwargs), daemon=True)
    p.start()

    p.join(timeout=timeout_sec)
    if p.is_alive():
        with contextlib.suppress(Exception):
            p.terminate()
        p.join(timeout=5)
        raise TimeoutError(f"diagnostic_scan timed out after {timeout_sec:.1f} s")

    if q.empty():
        raise RuntimeError(
            "diagnostic_scan subprocess exited without returning a result"
        )
    status, payload = q.get()
    if status == "ok":
        return payload
    err_repr, tb = payload
    raise RuntimeError(f"diagnostic_scan failed in subprocess: {err_repr}\n{tb}")


def find_root_brent(
    u: float,
    v: float,
    r0: float,
    r_max: float,
    p_eqState: Callable[[float], float],
    rho_eqState: Callable[[float], float],
    xi: float,
    m2: float,
    lmbda: float,
    nu: float,
    rho0: float,
    frac_pc: float,
    method: str,
    idx_sigma: int = 2,
    *,
    rtol: float = 1e-12,
    maxiter: int = 200,
    target: float | Sequence[float] = 0.0,
):
    def f(s0: float) -> float:
        return float(
            sigma_residual(
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
                idx_sigma=idx_sigma,
                target=target,
            )
        )

    try:
        fu = f(float(u))
        fv = f(float(v))
        if not (math.isfinite(fu) and math.isfinite(fv)):
            return None, None, None, None
        if fu * fv > 0.0:
            return None, None, None, None

        s_star = float(brentq(f, float(u), float(v), rtol=rtol, maxiter=maxiter))
        if s_star <= 0.0:
            return None, None, None, None

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


def _minimize_residual(
    u: float,
    v: float,
    r0: float,
    r_max: float,
    p_eqState: Callable[[float], float],
    rho_eqState: Callable[[float], float],
    xi: float,
    m2: float,
    lmbda: float,
    nu: float,
    rho0: float,
    frac_pc: float,
    method: str,
    idx_sigma: int,
    target: float | Sequence[float],
    abs_cut: float,
):
    def f(s0: float) -> float:
        return float(
            sigma_residual(
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
                idx_sigma=idx_sigma,
                target=target,
            )
        )

    try:
        res = minimize_scalar(lambda x: abs(f(x)), bounds=(u, v), method="bounded")
    except Exception:
        return None, None, None, None

    if not res.success:
        return None, None, None, None

    s_min = float(res.x)
    if s_min <= 0.0:
        return None, None, None, None

    delta, sigma_end, t_eff = sigma_residual(
        s_min,
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
    if math.isfinite(delta) and abs(delta) <= abs_cut:
        return float(s_min), float(delta), float(sigma_end), float(t_eff)
    return None, None, None, None


def probe_brackets(
    brackets: Sequence[Tuple[float, float]],
    r0: float,
    r_max: float,
    p_eqState: Callable[[float], float],
    rho_eqState: Callable[[float], float],
    xi_val: float,
    rho0: float,
    frac_pc: float,
    method: str,
    m2_val: float,
    lmbda_val: float,
    nu_val: float,
    abs_threshold: float,
    tol_relative: float,
    merge_tol: float,
    target: float | Sequence[float] = 0.0,
    idx_sigma: int = 2,
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    allow_minimize_fallback: bool = True,
) -> Tuple[List[float], List[float]]:
    """Refine brackets, find candidate σ0 roots, filter accepted ones."""

    candidates: List[float] = []
    residuals: List[float] = []
    sigma_ends: List[float] = []
    vac_signs: List[int] = []

    p0 = float(p_eqState(rho0))
    eps0 = float(rho0 * c * c)

    def accept_and_append(
        s_star: float, delta: float, sigma_end: float, t_eff: float
    ) -> None:
        _, psi2 = _sigma2_psi2(s_star, p0, eps0, xi_val, m2_val, lmbda_val, nu_val)
        psi_ok = math.isfinite(psi2) and (psi2 > 0.0)

        m0 = float(np.sqrt(max(0.0, m2_val)))
        m_vac_sq = float(m2_val + 2.0 * lmbda_val * (nu_val**2))
        uses_robin = ((nu_val == 0.0) and (m0 > 0.0)) or (
            (nu_val != 0.0) and (m_vac_sq > 0.0)
        )

        if uses_robin:
            accept = (abs(delta) <= abs_threshold) or (
                abs(s_star) > 0.0 and abs(delta) / abs(s_star) <= tol_relative
            )
            reason = "robin"
        else:
            accept, reason = residual_accept(
                sigma_rmax=sigma_end,
                sigma0=s_star,
                target=t_eff,
                abs_threshold=abs_threshold,
                tol_relative=tol_relative,
            )

        if accept and psi_ok:
            custom_print(
                f"ACCEPT σ0/M={s_star/M:.6e}  |residual|/M={abs(delta)/M:.3e}  Ψ2={psi2:.3e}  via={reason}",
                color="green",
            )
            candidates.append(float(s_star))
            residuals.append(float(abs(delta)))
            sigma_ends.append(float(sigma_end))
            if nu_val != 0.0:
                vac_signs.append(1 if t_eff >= 0.0 else -1)
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

    def process_one(u: float, v: float):
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
        if (s_star is None) and allow_minimize_fallback:
            s_star, delta, sigma_end, t_eff = _minimize_residual(
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
                abs_cut=abs_threshold,
            )
        if s_star is None:
            custom_print(
                f"[root] bracket [{u/M:.3e},{v/M:.3e}] → FAILED", color="yellow"
            )
            return
        accept_and_append(float(s_star), float(delta), float(sigma_end), float(t_eff))

    if not parallel:
        for u, v in brackets:
            process_one(float(u), float(v))
    else:
        with ThreadPoolExecutor(max_workers=max_workers or (os.cpu_count() or 8)) as ex:
            list(ex.map(lambda uv: process_one(float(uv[0]), float(uv[1])), brackets))

    # Merge near duplicates (per vacuum sign for nu!=0)
    if merge_tol is not None and merge_tol > 0.0 and len(candidates) > 1:
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
