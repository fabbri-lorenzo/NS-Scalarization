import numpy as np
from scipy.optimize import brentq
from Utils.params import c, M
from Utils.TOV import _sigma2_psi2
from Utils.solver import integrate_star
from Utils.shooting import _delta_to_target
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
    idx_sigma=2,
    target=0.0,
):
    """Return σ(r_max)−t* where t* is the nearest target to σ(r_max)."""
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
        stop_at_2r=False,
        record_mu2=False,
    )
    val = float(sol.y[idx_sigma, -1])
    return _delta_to_target(val, target)


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
    return S, F, brackets


def print_candidate_info(
    s0_list,
    rho0,
    r0,
    r_max,
    xi_val,
    lmbda_val,
    nu_val,
    p_eqState,
    rho_eqState,
    frac_pc,
    target,
):
    """Print diagnostic information for a list of accepted σ₀ values."""
    custom_print("\n[diagnostics] accepted σ0 candidates:", color="gray")
    s_maxes = []
    for s0 in s0_list:
        fr = sigma_residual(
            s0,
            r0,
            r_max,
            p_eqState,
            rho_eqState,
            xi_val,
            lmbda_val,
            nu_val,
            rho0,
            frac_pc,
            target,
        )
        p0 = float(p_eqState(rho0))
        eps0 = float(rho0 * c * c)
        _, psi2 = _sigma2_psi2(s0, p0, eps0, xi_val, lmbda_val, nu_val)
        custom_print(
            f"σ0/M = {s0/M:.6e} |σ(r_max)|/M = {abs(fr)/M:.3e} | Ψ2 = {psi2:.3e}",
            color="gray",
        )
        s_maxes.append(abs(fr))
    return s_maxes


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
            idx_sigma=idx_sigma,
            target=target,
        )
    try:
        s_star = brentq(f, float(u), float(v), rtol=rtol, maxiter=maxiter)
        return float(s_star), float(f(s_star))
    except Exception:
        return None, None


def probe_brackets_with_brent(
    brackets,
    existing_s0_list,
    r0,
    r_max,
    p_eqState,
    rho_eqState,
    xi_val,
    rho0,
    frac_pc,
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
        Sorted list of σ0 values combining existing and newly accepted roots.

    Notes
    -----
    This helper prints diagnostic messages indicating whether a bracketed root
    was accepted or rejected and why. Acceptance requires that the residual at
    r_max passes either the absolute or relative tolerance, and that the Ψ2
    coefficient of the central metric expansion is positive.
    """

    candidates = list(existing_s0_list) if existing_s0_list else []
    seen = {float(x) for x in existing_s0_list} if existing_s0_list else set()

    # Precompute central quantities for the Ψ2 check
    p0 = float(p_eqState(rho0))
    eps0 = float(rho0 * c * c)

    for u, v in brackets:
        s_star, fr = find_root_brent(
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
            target,
            idx_sigma=idx_sigma,
        )
        # bail out on failure
        if s_star is None or (not np.isfinite(fr)):
            custom_print(
                f"[brent] bracket [{u/M:.3e},{v/M:.3e}] → FAILED",
                color="yellow",
            )
            continue

        # deduplicate: skip if too close to an already-seen root
        duplicate = any(abs(s_star - z) <= merge_tol for z in seen)
        if duplicate:
            custom_print(
                f"[brent] σ0/M≈{s_star/M:.6e} skipped (duplicate within merge_tol).",
                color="yellow",
            )
            continue

        # determine acceptance via absolute then relative residual
        accept = False
        reason = None
        if abs(fr) <= abs_threshold:
            accept, reason = True, "abs"
        elif s_star != 0.0 and abs(fr / s_star) <= tol_relative:
            accept, reason = True, "rel"

        # Ψ2>0 regularity check
        _, psi2 = _sigma2_psi2(s_star, p0, eps0, xi_val, lmbda_val, nu_val)
        psi_ok = np.isfinite(psi2) and (psi2 > 0.0)

        if accept and psi_ok:
            custom_print(
                f"[brent] ACCEPT σ0/M={s_star/M:.6e}  |residual|/M={abs(fr)/M:.3e}  Ψ2={psi2:.3e}  via={reason}",
                color="green",
            )
            candidates.append(s_star)
            seen.add(float(s_star))
        else:
            msg = []
            if not accept:
                msg.append("residual too large")
            if not psi_ok:
                msg.append(f"Ψ2≤0 (Ψ2={psi2:.3e})")
            why = "; ".join(msg) if msg else "unknown"
            custom_print(
                f"[brent] REJECT σ0/M={s_star/M:.6e}  |residual|/M={abs(fr)/M:.3e}  → {why}",
                color="red",
            )

    # return sorted unique list of roots by absolute value
    return sorted({float(x) for x in candidates}, key=lambda z: abs(z))
