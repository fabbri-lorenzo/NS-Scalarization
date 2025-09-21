import numpy as np
from Utils.params import c, M
from Utils.TOV_EMG import _sigma2_psi2
from Utils.solver import integrate_star


def sigma_residual(s0, r0, r_max, p_eqState, rho_eqState, idx_sigma=2):
    """Return σ(r_max) from a single background integration."""
    sol, _, _ = integrate_star(
        s0, p_eqState, rho_eqState, r0, r_max, stop_at_2r=False, record_mu2=False
    )
    return float(sol.y[idx_sigma, -1])


def diagnostic_scan(r0, r_max, p_eqState, rho_eqState, a, b, n=201):
    """Coarse scan of σ(r_max) over [a,b]: report sign changes and minima."""
    S = np.linspace(a, b, int(n))
    F = []
    for s in S:
        try:
            F.append(sigma_residual(float(s), r0, r_max, p_eqState, rho_eqState, idx_sigma=2))
        except Exception:
            F.append(np.nan)
    F = np.asarray(F, float)
    # sign-change brackets
    brackets = []
    for i in range(len(S) - 1):
        f1, f2 = F[i], F[i + 1]
        if (
            np.isfinite(f1)
            and np.isfinite(f2)
            and (f1 == 0 or f2 == 0 or (f1 * f2 < 0))
        ):
            brackets.append((S[i], S[i + 1]))
    # local minima in |F|
    dips = []
    for i in range(1, len(S) - 1):
        if np.isfinite(F[i - 1]) and np.isfinite(F[i]) and np.isfinite(F[i + 1]):
            if abs(F[i]) <= abs(F[i - 1]) and abs(F[i]) <= abs(F[i + 1]):
                dips.append((S[i], F[i]))
    return S, F, brackets, dips


def print_candidate_info(s0_list, rho0, r0, r_max, xi_val, lmbda_val, p_eqState, rho_eqState,):
    print("\n[diagnostics] accepted σ0 candidates:")
    for s0 in s0_list:
        fr = sigma_residual(s0, r0, r_max, p_eqState, rho_eqState, idx_sigma=2)
        p0 = float(p_eqState(rho0))
        eps0 = float(rho0 * c * c)
        _, psi2 = _sigma2_psi2(s0, p0, eps0, xi_val, lmbda_val)
        print(f"  σ0/M = {s0/M:.6e} |σ(r_max)|/M = {abs(fr)/M:.3e} | Ψ2 = {psi2:.3e}")