from Utils.params import c, M2
import numpy as np


# === INITIAL CONDITIONS EMG (SI) — regular O(r^2) center expansion ===
def _sigma2_psi2(sigma0, p0, eps0, xi, lmbda):
    # Denominator in the σ₂ expression; can vanish for large |xi| or σ₀
    denom_sigma = 6 * (M2 + xi * sigma0**2 + 6 * xi**2 * sigma0**2)
    # if (not np.isfinite(denom_sigma)) or (abs(denom_sigma) < 1e-30):
    # Return NaNs so the integrator can treat this initial guess as invalid
    #    return float("nan"), float("nan")
    sigma2 = (M2 * lmbda * sigma0**3 - xi * sigma0 * (eps0 - 3 * p0)) / denom_sigma

    denom_Psi = 6 * (M2 + xi * sigma0**2)
    # if (not np.isfinite(denom_Psi)) or (abs(denom_Psi) < 1e-30):
    #    return float("nan"), float("nan")
    Psi2 = (12 * xi * sigma0 * sigma2 + eps0 + 0.25 * lmbda * sigma0**4) / denom_Psi
    return sigma2, Psi2


def initial_conditions(r0, sigma0, p_eqState, xi, rho0, lmbda):
    # --- central thermodynamics ---
    p0   = float(p_eqState(rho0))
    eps0 = float(rho0 * c * c)

    sigma2, Psi2 = _sigma2_psi2(sigma0, p0, eps0, xi, lmbda)

    Phi2= (2*M2*Psi2 + 2*xi*sigma0**2*Psi2 -8*xi*sigma0*sigma2 + p0-0.25*lmbda*sigma0**4)/(4*M2 + 4*xi*sigma0**2)

    p2 = -(p0 + eps0) * Phi2
    p_c = p0 + r0**2 * p2
    Psi_c= r0**2 *Psi2
    sigma_c= sigma0 + r0**2 * sigma2
    dsigma_c = 2*r0*sigma2

    return [p_c, Psi_c, sigma_c, dsigma_c]


# === Scalar-Tensor Functions ===
def F(sigma, xi):
    return M2+ xi* sigma**2

def dF_dsigma(sigma,xi):
    return 2 * xi * sigma


def V(sigma, lmbda):
    return 0.25 * lmbda * sigma**4


def dV_dsigma(sigma, lmbda):
    return lmbda * sigma**3


# === System of First-Order ODEs ===
def make_tov_EMG(p_c, frac_pc, rho_eqState, xi, lmbda, mu2_recorder=None):

    def tov_system(r, y):
        p, Psi, sigma, dsigma = y

        if p < p_c * frac_pc:
            p = 0.0
            eps = 0.0
        else:
            rho = rho_eqState(p)
            eps = rho * c * c

        F_val = F(sigma, xi)
        F_prime = dF_dsigma(sigma, xi) * dsigma
        # safe metric factor: clamp exponent to [-700, 700]
        exponent = np.clip(2.0 * Psi, -700.0, 700.0)
        g11 = np.exp(exponent)

        # dΦ/dr
        denom_dPhi = 2 * F_val + r * F_prime
        # if (not np.isfinite(denom_dPhi)) or (abs(denom_dPhi) < 1e-30):
        #    return [np.inf, np.inf, np.inf, np.inf]

        dPhi = (1 / denom_dPhi) * (
            F_val * (g11 - 1) / r
            + 0.5 * r * dsigma**2
            + (p - V(sigma, lmbda)) * r * g11
            - 2 * F_prime
        )

        # Definition of Ψ̄ for convenience
        denom_Psi_bar = r * denom_dPhi
        Psi_bar = (1 / denom_Psi_bar) * (
            (1 - g11) * F_val
            + r**2
            * (
                2 * xi * ((dPhi + 2 / r) * sigma * dsigma + dsigma**2)
                + g11 * (eps + V(sigma, lmbda))
                + (dsigma**2) / 2
            )
        )

        # d²σ/dr²
        denom_ddsigma = 2 * F_val * (F_val + 6 * xi**2 * sigma**2)
        # if (not np.isfinite(denom_ddsigma)) or (abs(denom_ddsigma) < 1e-30):
        #    return [np.inf, np.inf, np.inf, np.inf]

        dd_sigma = (
            (2 * F_val + r * F_prime)
            / denom_ddsigma
            * (
                F_val
                * (
                    g11 * dV_dsigma(sigma, lmbda)
                    - dPhi * dsigma
                    - 2 * dsigma / r
                    + Psi_bar * dsigma
                )
                - xi
                * sigma
                * (
                    dsigma**2
                    + g11 * (4 * (V(sigma, lmbda)) + eps - 3 * p)
                    + 6 * xi * ((dPhi - Psi_bar + 2 / r) * sigma * dsigma + dsigma**2)
                )
            )
        )

        # dΨ/dr
        denom_dPsi = denom_dPhi
        dPsi = Psi_bar + (2 * r * xi) * sigma * dd_sigma / denom_dPsi

        # dp/dr
        dp = -(eps + p) * dPhi

        # effective mass squared
        mu2 = -(xi / F_val) * (
            1 / g11 * dsigma**2
            + 4 * V(sigma)
            + eps
            - 3 * p
            + 6
            * xi
            / g11
            * ((dPhi - dPsi + 2 / r) * sigma * dsigma + dsigma**2 + sigma * dd_sigma)
        )
        if mu2_recorder is not None:
            # store (r, μ²) without touching solver tolerances/state
            mu2_recorder((r, mu2))

        return [dp, dPsi, dsigma, dd_sigma]
    return tov_system  


# === Stopping Conditions ===
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
    def blowup_guard(self, xi):
        # thresholds: small denominators & very large |Psi|
        DEN_TOL = 1e-30
        PSI_MAX = 350.0  # well within exp() clamp; avoids pathological regimes

        def _ev(r, y):
            _, Psi, sigma, dsigma = y
            # recompute small-denominator conditions cheaply
            F_val = M2 + xi * sigma**2
            F_prime = 2.0 * xi * sigma * dsigma

            denom1 = abs(2.0 * F_val + r * F_prime)
            denom2 = abs(2.0 * F_val * (F_val + 6.0 * xi**2 * sigma**2))

            # event root when min(...) hits zero
            return min(denom1 - DEN_TOL, denom2 - DEN_TOL, PSI_MAX - abs(Psi))

        _ev.terminal = True
        _ev.direction = -1
        return _ev
