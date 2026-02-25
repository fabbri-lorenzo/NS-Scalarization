from Utils.params import c, M2
import numpy as np
import warnings

SAFE_CUTOFF = 1e200
DENOM_FLOOR = 1e-200  # avoid near-zero divisions
MAX_EXP_ARG = 700  # exp(700) ~ 1e304, below overflow

warnings.filterwarnings("error", category=RuntimeWarning)
np.seterr(over="raise", divide="raise", invalid="raise")

def _sigma2_psi2(sigma0, p0, eps0, xi, m2, lmbda, nu):
    """
    Compute the second-order expansion coefficients σ₂ and Ψ₂ for the regular
    center expansion.
    """
    denom_sigma = 6 * (M2 + xi * sigma0**2 + 6 * xi**2 * sigma0**2)
    sigma2 = (
        M2 * lmbda * sigma0**3
        - xi * sigma0 * (eps0 - 3 * p0 + lmbda * nu**4 - 2 * lmbda * nu**2 * sigma0**2)
        + m2 * sigma0 * (M2 - xi * sigma0**2)
    ) / denom_sigma
    denom_Psi = 6 * (M2 + xi * sigma0**2)
    Psi2 = (
        12 * xi * sigma0 * sigma2
        + eps0
        + 0.25 * lmbda * sigma0**4
        + 0.25 * lmbda * nu**4
        - 0.5 * lmbda * nu**2 * sigma0**2
        + 0.5 * m2 * sigma0**2
    ) / denom_Psi
    return sigma2, Psi2

def initial_conditions(r0, sigma0, p_eqState, xi, m2, rho0, lmbda, nu):
    # --- central thermodynamics ---
    p0   = float(p_eqState(rho0))
    eps0 = float(rho0 * c * c)

    sigma2, Psi2 = _sigma2_psi2(sigma0, p0, eps0, xi, m2, lmbda, nu)
    if (not np.isfinite(sigma2)) or (not np.isfinite(Psi2)):
        return [np.nan, np.nan, np.nan, np.nan]

    Phi2 = (
        2 * M2 * Psi2
        + 2 * xi * sigma0**2 * Psi2
        - 8 * xi * sigma0 * sigma2
        + p0
        - 0.25 * lmbda * sigma0**4
        - 0.5 * m2 * sigma0**2
        - 0.25 * lmbda * nu**4
        + 0.5 * lmbda * nu**2 * sigma0**2
    ) / (4 * M2 + 4 * xi * sigma0**2)

    p2 = -(p0 + eps0) * Phi2
    p_c = p0 + r0**2 * p2
    Psi_c= r0**2 *Psi2
    sigma_c= sigma0 + r0**2 * sigma2
    dsigma_c = 2*r0*sigma2

    return [p_c, Psi_c, sigma_c, dsigma_c]

def F(sigma, xi):
    return M2+ xi* sigma**2

def dF_dsigma(sigma,xi):
    return 2 * xi * sigma

def V(sigma, m2, lmbda, nu):
    return 0.5 * m2 * sigma**2 + 0.25 * lmbda * (sigma**2 - nu**2) ** 2

def dV_dsigma(sigma, m2, lmbda, nu):
    return m2 * sigma + lmbda * (sigma**3 - nu**2 * sigma)


def make_tov(
    p_c,
    frac_pc,
    rho_eqState,
    xi,
    m2,
    lmbda,
    nu,
    mu2_recorder=None,
):
    """
    Create the TOV system of ODEs with cutoff-based guards to stop the solver
    before values become non-finite or numerically unstable.

    Parameters
    ----------
    SAFE_CUTOFF : float
        If |x| > SAFE_CUTOFF for critical quantities, return NaNs.
        (Use smaller values like 1e100 or 1e50 for earlier stopping.)
    DENOM_FLOOR : float
        Minimum allowed absolute denominator.
    MAX_EXP_ARG : float
        Maximum exponent allowed in exp(2*Psi) to avoid overflow.
    """

    def _unsafe(x, cutoff=SAFE_CUTOFF):
        """True if x is non-finite or too large in magnitude."""
        return (not np.isfinite(x)) or (abs(x) > cutoff)

    def _bad_denom(x):
        """True if denominator is unsafe or too close to zero."""
        return _unsafe(x) or (abs(x) < DENOM_FLOOR)

    def tov_system(r, y):
        try:
            p, Psi, sigma, dsigma = y

            # -------------------------
            # Basic state / radius guards
            # -------------------------
            if _unsafe(r) or (abs(r) < DENOM_FLOOR):
                return [np.nan, np.nan, np.nan, np.nan]

            if any(_unsafe(v) for v in (p, Psi, sigma, dsigma)):
                return [np.nan, np.nan, np.nan, np.nan]

            # -------------------------
            # EOS / thermodynamics
            # -------------------------
            if p < p_c * frac_pc:
                p = 0.0
                eps = 0.0
            else:
                rho = rho_eqState(p)
                eps = rho * c * c
                if any(_unsafe(v) for v in (rho, eps)):
                    return [np.nan, np.nan, np.nan, np.nan]

            # -------------------------
            # Coupling function and metric factor
            # -------------------------
            F_val = F(sigma, xi)
            if _unsafe(F_val) or (F_val <= DENOM_FLOOR):
                return [np.nan, np.nan, np.nan, np.nan]

            dFds = dF_dsigma(sigma, xi)
            if _unsafe(dFds):
                return [np.nan, np.nan, np.nan, np.nan]

            F_prime = dFds * dsigma
            if _unsafe(F_prime):
                return [np.nan, np.nan, np.nan, np.nan]

            # g11 = exp(2Psi), but stop BEFORE overflow
            exponent = 2.0 * Psi
            if _unsafe(exponent) or (exponent > MAX_EXP_ARG):
                return [np.nan, np.nan, np.nan, np.nan]

            # Very negative exponent is okay (g11 -> 0), unless you want a lower guard too
            g11 = np.exp(exponent)
            if _unsafe(g11) or (g11 <= 0.0):
                return [np.nan, np.nan, np.nan, np.nan]

            # -------------------------
            # Potential and derivative
            # -------------------------
            V_val = V(sigma, m2, lmbda, nu)
            dV_val = dV_dsigma(sigma, m2, lmbda, nu)
            if any(_unsafe(v) for v in (V_val, dV_val)):
                return [np.nan, np.nan, np.nan, np.nan]

            # -------------------------
            # dPhi/dr
            # -------------------------
            denom_dPhi = 2.0 * F_val + r * F_prime
            if _bad_denom(denom_dPhi):
                return [np.nan, np.nan, np.nan, np.nan]

            term_dPhi_num = (
                F_val * (g11 - 1.0) / r
                + 0.5 * r * dsigma**2
                + (p - V_val) * r * g11
                - 2.0 * F_prime
            )
            if _unsafe(term_dPhi_num):
                return [np.nan, np.nan, np.nan, np.nan]

            dPhi = term_dPhi_num / denom_dPhi
            if _unsafe(dPhi):
                return [np.nan, np.nan, np.nan, np.nan]

            # -------------------------
            # Psi_bar (auxiliary)
            # -------------------------
            denom_Psi_bar = r * denom_dPhi
            if _bad_denom(denom_Psi_bar):
                return [np.nan, np.nan, np.nan, np.nan]

            # build numerator in pieces for easier guarding/debugging
            num_Psi_bar_1 = (1.0 - g11) * F_val
            num_Psi_bar_2 = r**2 * (
                2.0 * xi * ((dPhi + 2.0 / r) * sigma * dsigma + dsigma**2)
                + g11 * (eps + V_val)
                + 0.5 * dsigma**2
            )
            if any(_unsafe(v) for v in (num_Psi_bar_1, num_Psi_bar_2)):
                return [np.nan, np.nan, np.nan, np.nan]

            Psi_bar = (num_Psi_bar_1 + num_Psi_bar_2) / denom_Psi_bar
            if _unsafe(Psi_bar):
                return [np.nan, np.nan, np.nan, np.nan]

            # -------------------------
            # d²sigma/dr²
            # -------------------------
            denom_ddsigma = 2.0 * F_val * (F_val + 6.0 * xi**2 * sigma**2)
            if _bad_denom(denom_ddsigma):
                return [np.nan, np.nan, np.nan, np.nan]

            prefactor_ddsigma = (2.0 * F_val + r * F_prime) / denom_ddsigma
            if _unsafe(prefactor_ddsigma):
                return [np.nan, np.nan, np.nan, np.nan]

            bracket_1 = F_val * (
                g11 * dV_val - dPhi * dsigma - 2.0 * dsigma / r + Psi_bar * dsigma
            )

            bracket_2 = (
                xi
                * sigma
                * (
                    dsigma**2
                    + g11 * (4.0 * V_val + eps - 3.0 * p)
                    + 6.0
                    * xi
                    * ((dPhi - Psi_bar + 2.0 / r) * sigma * dsigma + dsigma**2)
                )
            )

            if any(_unsafe(v) for v in (bracket_1, bracket_2)):
                return [np.nan, np.nan, np.nan, np.nan]

            dd_sigma = prefactor_ddsigma * (bracket_1 - bracket_2)
            if _unsafe(dd_sigma):
                return [np.nan, np.nan, np.nan, np.nan]

            # -------------------------
            # dPsi/dr
            # -------------------------
            denom_dPsi = denom_dPhi
            if _bad_denom(denom_dPsi):
                return [np.nan, np.nan, np.nan, np.nan]

            correction_dPsi = (2.0 * r * xi) * sigma * dd_sigma / denom_dPsi
            if _unsafe(correction_dPsi):
                return [np.nan, np.nan, np.nan, np.nan]

            dPsi = Psi_bar + correction_dPsi
            if _unsafe(dPsi):
                return [np.nan, np.nan, np.nan, np.nan]

            # -------------------------
            # dp/dr
            # -------------------------
            dp = -(eps + p) * dPhi
            if _unsafe(dp):
                return [np.nan, np.nan, np.nan, np.nan]

            # -------------------------
            # effective mass squared mu²
            # -------------------------
            # inv_g11 = 1.0 / g11
            #
            # mu2_inner = (
            #    inv_g11 * dsigma**2
            #    + 4.0 * V_val
            #    + eps
            #    - 3.0 * p
            #    + 6.0
            #    * xi
            #    * inv_g11
            #    * (
            #        (dPhi - dPsi + 2.0 / r) * sigma * dsigma
            #        + dsigma**2
            #        + sigma * dd_sigma
            #    )
            # )
            # if _unsafe(mu2_inner):
            #    return [np.nan, np.nan, np.nan, np.nan]

            # mu2 = -(xi / F_val) * mu2_inner +2* lmbda * nu**2 + m2
            if nu == 0.0 or (nu != 0.0 and lmbda <= m2 / nu**2):
                mu2 = -(xi / M2) * (eps - 3 * p) + m2 - lmbda * nu**2  # sigma_min is 0

            if nu != 0.0 and lmbda > m2 / nu**2:
                sigma_min = np.sqrt(nu**2 - m2 / lmbda)
                mu2 = (
                    -xi
                    / F(sigma_min, xi)
                    * (eps - 3 * p + 4 * V(sigma_min, m2, lmbda, nu))
                    + 2 * lmbda * nu**2
                    - 2 * m2
                )

            if mu2_recorder is not None:
                # store (r, mu²) without touching solver tolerances/state
                mu2_recorder((r, mu2))

            # Final derivative vector
            if any(_unsafe(v) for v in (dp, dPsi, dsigma, dd_sigma)):
                return [np.nan, np.nan, np.nan, np.nan]

            return [dp, dPsi, dsigma, dd_sigma]

        except (FloatingPointError, RuntimeWarning, OverflowError, ZeroDivisionError):
            return [np.nan, np.nan, np.nan, np.nan]

    return tov_system


class StoppingConditions:
    def __init__(self, p_c, frac_pc):
        self.p_c = float(p_c)
        self.frac_pc = float(frac_pc)
        self.R_star = None

    def pressure_limit(self):
        def _ev(r, y):
            p = y[0]
            val = p - self.frac_pc * self.p_c
            if (val <= 0.0) and (self.R_star is None):
                self.R_star = float(r)
            return val
        _ev.terminal = False       # not stopping at surface
        _ev.direction = -1
        return _ev

    def double_radius(self):
        def _ev(r, y):
            if self.R_star is None:
                return 1.0
            return r - 2.0 * self.R_star
        _ev.terminal = True        # do stop at 2R*
        _ev.direction = +1
        return _ev

    # Terminate when dynamics become singular/unsafe (prevents stalls)
    def blowup_guard(self, xi, cutoff=SAFE_CUTOFF):
        def _ev(r, y):
            if (not np.isfinite(r)) or (abs(r) > cutoff):
                return 0.0
            for v in y:
                if (not np.isfinite(v)) or (abs(v) > cutoff):
                    return 0.0
            return 1.0
        _ev.terminal = True
        _ev.direction = -1
        return _ev
