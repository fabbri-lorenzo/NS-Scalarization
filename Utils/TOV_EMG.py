from Utils.params import c, M2, lmbda_EMG
import numpy as np

lmbda = lmbda_EMG


# === INITIAL CONDITIONS EMG (SI) — regular O(r^2) center expansion ===
def _sigma2_psi2(sigma0, p0, eps0, xi, lmbda=lmbda):
    sigma2 = (M2 * lmbda * sigma0**3 - xi * sigma0 * (eps0 - 3 * p0)) / (
        6 * (M2 + xi * sigma0**2 - 2 * xi**2 * sigma0**2)
    )

    Psi2 = (12 * xi * sigma0 * sigma2 + eps0 + 0.25 * lmbda * sigma0**4) / (
        6 * (M2 + xi * sigma0**2)
    )
    return sigma2, Psi2


def initial_conditions(r0, sigma0, p_eqState, xi, rho0):
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

def V(sigma):
    return 0.25 * lmbda * sigma**4

def dV_dsigma(sigma):
    return lmbda * sigma**3

# === System of First-Order ODEs ===
def make_tov_EMG(p_c,frac_pc, rho_eqState, xi, mu2_recorder=None):

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
        g11 = np.exp(2 * Psi)

        # dPhi/dr
        dPhi = (1 / (2 * F_val + r * F_prime)) * (
            F_val * (g11 - 1) / r
            + 0.5 * r * dsigma**2
            + (p - V(sigma)) * r * g11
            - 2 * F_prime
        )

        # Definition of \bar{m} for convenience
        Psi_bar = (1 / (r * (2 * F_val + r * F_prime))) * (
            (1 - g11) * F_val
            + r**2
            * (
                2 * xi * ((dPhi + 2 / r) * sigma * dsigma + dsigma**2)
                + g11 * (eps + V(sigma))
                + (dsigma**2) / 2
            )
        )

        # d^2sigma/dr^2)
        dd_sigma = (
            (2 * F_val + r * F_prime)
            / (2 * F_val * (F_val + 6 * xi**2 * sigma**2))
            * (
                F_val
                * (
                    g11 * dV_dsigma(sigma)
                    - dPhi * dsigma
                    - 2 * dsigma / r
                    + Psi_bar * dsigma
                )
                - xi
                * sigma
                * (
                    dsigma**2
                    + g11 * (4 * (V(sigma)) + eps - 3 * p)
                    + 6 * xi * ((dPhi - Psi_bar + 2 / r) * sigma * dsigma + dsigma**2)
                )
            )
        )

        # dm/dr
        dPsi = Psi_bar + (2 * r * xi) * sigma * dd_sigma / (2 * F_val + r * F_prime)

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
