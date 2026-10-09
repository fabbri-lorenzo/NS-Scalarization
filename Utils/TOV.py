from Utils.params import c, M2
import numpy as np
import warnings
from Utils.graphics_single import custom_print

MAX_EXP_ARG = 700  # exp(700) ~ 1e304, below overflow
MIN_ABS_NUM = 1e-300
MAX_ABS_NUM = 1e300

warnings.filterwarnings("error", category=RuntimeWarning)
np.seterr(over="raise", divide="raise", invalid="raise")

def _sigma2_psi2(sigma0, p0, eps0, xi, m2, lmbda, nu):
    """
    Compute the second-order expansion coefficients σ₂ and Ψ₂ for the regular
    center expansion.
    """
    denom_sigma = 6 * (M2 + xi * sigma0**2 + 6 * xi**2 * sigma0**2)
    # sigma2 = (
    #    M2 * lmbda * sigma0**3
    #    - xi * sigma0 * (eps0 - 3 * p0 + lmbda * nu**4 - 2 * lmbda * nu**2 * sigma0**2)
    #    + m2 * sigma0 * (M2 - xi * sigma0**2)
    # ) / denom_sigma
    sigma2 = (
        M2 * lmbda * sigma0**3
        - M2 * lmbda * nu**2 * sigma0
        + m2 * sigma0 * (M2 - xi * sigma0**2)
        - xi * sigma0 * (eps0 - 3 * p0 + lmbda * nu**4 - lmbda * nu**2 * sigma0**2)
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
    """Second-order central expansion for pressure/metric/scalar at r0."""
    # --- central thermodynamics ---
    p0   = float(p_eqState(rho0))
    eps0 = float(rho0 * c * c)

    sigma2, Psi2 = _sigma2_psi2(sigma0, p0, eps0, xi, m2, lmbda, nu)

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

    if any(
        (abs(x) < MIN_ABS_NUM or abs(x) > MAX_ABS_NUM)
        for x in [p_c, Psi_c, sigma_c, dsigma_c]
    ):
        return [np.nan, np.nan, np.nan, np.nan]

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
    Build the TOV+scalar system dy/dr for solve_ivp, adding guards that end
    integration before numerical blow-ups and optionally recording μ_eff².
    """

    def tov_system(r, y):
        try:
            p, Psi, sigma, dsigma = y

            # -------------------------
            # EOS / thermodynamics
            # -------------------------
            if p < p_c * frac_pc:
                p = 0.0
                eps = 0.0
            else:
                rho = rho_eqState(p)
                eps = rho * c * c

            # -------------------------
            # Coupling function and metric factor
            # -------------------------
            F_val = F(sigma, xi)

            dFds = dF_dsigma(sigma, xi)

            F_prime = dFds * dsigma

            # g11 = exp(2Psi), but stop BEFORE overflow
            exponent = 2.0 * Psi
            if exponent > MAX_EXP_ARG:
                return [np.nan, np.nan, np.nan, np.nan]

            g11 = np.exp(exponent)

            # -------------------------
            # Potential and derivative
            # -------------------------
            V_val = V(sigma, m2, lmbda, nu)
            dV_val = dV_dsigma(sigma, m2, lmbda, nu)

            # -------------------------
            # dPhi/dr
            # -------------------------
            denom_dPhi = 2.0 * F_val + r * F_prime

            term_dPhi_num = (
                F_val * (g11 - 1.0) / r
                + 0.5 * r * dsigma**2
                + (p - V_val) * r * g11
                - 2.0 * F_prime
            )

            dPhi = term_dPhi_num / denom_dPhi

            # -------------------------
            # Psi_bar (auxiliary)
            # -------------------------
            denom_Psi_bar = r * denom_dPhi

            # build numerator in pieces for easier guarding/debugging
            num_Psi_bar = (1.0 - g11) * F_val + r**2 * (
                2.0 * xi * ((2.0 / r) * sigma * dsigma + dsigma**2)
                + g11 * (eps + V_val)
                + 0.5 * dsigma**2
            )

            Psi_bar = num_Psi_bar / denom_Psi_bar

            # -------------------------
            # d²sigma/dr²
            # -------------------------
            prefactor_ddsigma = (2.0 * F_val + r * F_prime) / (2.0 * F_val)

            bracket_1 = (
                g11 * (F_val * dV_val - xi * sigma * (eps - 3 * p + 4 * V_val))
                - xi * sigma * (1 + 6 * xi) * dsigma**2
            ) / (F_val + 6 * (xi * sigma) ** 2)

            bracket_2 = (dPhi + 2.0 / r - Psi_bar) * dsigma

            dd_sigma = prefactor_ddsigma * (bracket_1 - bracket_2)

            # -------------------------
            # dPsi/dr
            # -------------------------
            denom_dPsi = denom_dPhi

            correction_dPsi = (2.0 * r * xi) * sigma * dd_sigma / denom_dPsi

            dPsi = Psi_bar + correction_dPsi

            # -------------------------
            # dp/dr
            # -------------------------
            dp = -(eps + p) * dPhi

            # -------------------------
            # effective mass squared mu²
            # -------------------------
            mu2 = (
                m2
                - lmbda * nu**2
                + 3 * lmbda * sigma**2
                + xi
                * (
                    dsigma**2 / g11
                    + 4 * V_val
                    + eps
                    - 3 * p
                    + 3
                    * 2
                    * xi
                    * (
                        (dPhi - dPsi) * sigma * dsigma
                        + 2 * sigma * dsigma / r
                        + dsigma**2
                        + sigma * dd_sigma
                    )
                    / g11
                )
                / F_val
            )

            if mu2_recorder is not None:
                # store (r, mu²) without touching solver tolerances/state
                mu2_recorder((r, mu2))

            # Final derivative vector

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
                # custom_print(f"surface reached at r={r:.3e} m", color="blue")
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
    def blowup_guard(self):

        def _ev(r, y):
            for v in y:
                if not np.isfinite(v):
                    custom_print(f"blow‑up at r={r:.3e} m, state={y}", color="red")
                    return 0.0
            return 1.0

        _ev.terminal = True
        _ev.direction = -1
        return _ev
