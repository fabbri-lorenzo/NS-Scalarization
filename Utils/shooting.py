import numpy as np
from Utils.params import c, M
from Utils.TOV import _sigma2_psi2
from Utils.analysis import integrate_star
from Utils.graphics import custom_print
from scipy.optimize import brentq


def sigma_residual(
    s0,
    r0,
    r_max,
    p_eqState,
    rho_eqState,
    xi,
    lmbda,
    nu,
    rho0,
    frac_pc,
    method,
    idx_sigma=2,
    target=0.0,
    *,
    return_details=False,
):
    """Return residual σ(r_max)−t*, optionally with (sigma_end, t*)."""
    sol, _, _ = integrate_star(
        s0,
        p_eqState,
        rho_eqState,
        r0,
        r_max,
        xi,
        lmbda,
        nu,
        rho0,
        frac_pc,
        method=method,
        stop_at_2r=False,
        record_mu2=False,
    )
    val = float(sol.y[idx_sigma, -1])
    t_eff = _nearest_target_value(val, target)
    delta = val - t_eff
    return (delta, val, t_eff) if return_details else delta


def diagnostic_scan(
    r0,
    r_max,
    p_eqState,
    rho_eqState,
    xi,
    lmbda,
    nu_val,
    rho0,
    frac_pc,
    method,
    a,
    b,
    n=201,
    target=0.0,
):
    S = np.linspace(a, b, int(n))
    F = []
    for s in S:
        try:
            F.append(
                sigma_residual(
                    float(s),
                    r0,
                    r_max,
                    p_eqState,
                    rho_eqState,
                    xi,
                    lmbda,
                    nu_val,
                    rho0,
                    frac_pc,
                    method,
                    idx_sigma=2,
                    target=target,
                )
            )
        except Exception:
            F.append(np.nan)
    F = np.asarray(F, float)
    brackets = []
    for i in range(len(S) - 1):
        f1, f2 = F[i], F[i + 1]
        if (
            np.isfinite(f1)
            and np.isfinite(f2)
            and (f1 == 0 or f2 == 0 or (f1 * f2 < 0))
        ):
            brackets.append((S[i], S[i + 1]))
    return F, brackets


def find_root_brent(
    u,
    v,
    r0,
    r_max,
    p_eqState,
    rho_eqState,
    xi,
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
    lmbda_val,
    nu_val,
    abs_threshold,
    tol_relative,
    merge_tol,
    target=0.0,
    idx_sigma: int = 2,
):
    """
    Probe each sign-change bracket using Brent’s method to recover missing σ0 roots.

    Parameters
    ----------
    brackets : list of tuple
        List of (u, v) pairs where σ(r_max) changes sign.
    existing_s0_list : iterable
        List of σ0 values already found by the shooting method.
    r0, r_max : float
        Integration bounds.
    p_eqState, rho_eqState : callable
        Equation of state functions.
    xi_val, rho0, frac_pc, lmbda_val : float
        Model parameters.
    abs_threshold : float
        Absolute residual tolerance for accepting roots.
    tol_relative : float
        Relative residual tolerance for accepting roots.
    merge_tol : float
        Tolerance for merging near-duplicate solutions.
    idx_sigma : int, optional
        Index of the scalar field component in the solution vector.

    Returns
    -------
    list
        Sorted list of σ0 roots.

    Notes
    -----
    This helper prints diagnostic messages indicating whether a bracketed root
    was accepted or rejected and why. Acceptance requires that the residual at
    r_max passes either the absolute or relative tolerance, and that the Ψ2
    coefficient of the central metric expansion is positive.
    """

    candidates = []
    s_maxes = []

    # Precompute central quantities for the Ψ2 check
    p0 = float(p_eqState(rho0))
    eps0 = float(rho0 * c * c)

    for u, v in brackets:
        s_star, delta, sigma_end, t_eff = find_root_brent(
            u,
            v,
            r0,
            r_max,
            p_eqState,
            rho_eqState,
            xi_val,
            lmbda_val,
            nu_val,
            rho0,
            frac_pc,
            method,
            idx_sigma=idx_sigma,
            target=target,
        )
        if s_star is None or (not np.isfinite(delta)):
            custom_print(
                f"[brent] bracket [{u/M:.3e},{v/M:.3e}] → FAILED",
                color="yellow",
            )
            continue

        accept, reason = residual_accept(
            sigma_rmax=sigma_end,
            sigma0=s_star,
            target=t_eff,
            abs_threshold=abs_threshold,
            tol_relative=tol_relative,
        )

        _, psi2 = _sigma2_psi2(s_star, p0, eps0, xi_val, lmbda_val, nu_val)
        psi_ok = np.isfinite(psi2) and (psi2 > 0.0)

        if accept and psi_ok:
            custom_print(
                f"ACCEPT σ0/M={s_star/M:.6e}  |residual|/M={abs(delta)/M:.3e}  Ψ2={psi2:.3e}  via={reason}",
                color="green",
            )
            candidates.append(float(s_star))
            s_maxes.append(float(abs(delta)))
        else:
            msg = []
            if not accept:
                msg.append("residual too large")
            if not psi_ok:
                msg.append(f"Ψ2≤0 (Ψ2={psi2:.3e})")
            why = "; ".join(msg) if msg else "unknown"
            custom_print(
                f"REJECT σ0/M={s_star/M:.6e}  |residual|/M={abs(delta)/M:.3e} → {why}",
                color="red",
            )

    if merge_tol is not None and merge_tol > 0 and len(candidates) > 1:
        order = np.argsort(np.abs(candidates))
        cand_sorted = [candidates[i] for i in order]
        res_sorted = [s_maxes[i] for i in order]
        merged_c, merged_r = [cand_sorted[0]], [res_sorted[0]]
        for s0, r0 in zip(cand_sorted[1:], res_sorted[1:]):
            if abs(s0 - merged_c[-1]) <= merge_tol:
                # keep the one with smaller residual
                if r0 < merged_r[-1]:
                    merged_c[-1], merged_r[-1] = s0, r0
            else:
                merged_c.append(s0)
                merged_r.append(r0)
        candidates, s_maxes = merged_c, merged_r

    # return sorted unique list of roots by absolute value
    return sorted(candidates, key=lambda z: abs(z)), s_maxes
