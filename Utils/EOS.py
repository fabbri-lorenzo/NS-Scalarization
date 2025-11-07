# === Equation of State SLy4 (Haensel & Potekhin 2004) ===
# Uses ζ = log10(P[dyn/cm^2]), ξ = log10(ρ[g/cm^3]), with f0(x)=1/(e^x+1).
# Table 1 coefficients for SLy (1-based indexing preserved for readability).

import numpy as np
from scipy.optimize import brentq
from Utils.params import c

_RHO_FLOOR = 1e-12  # kg/m^3, tiny density floor to keep RHS well-defined at the surface

_a = np.array(
    [
        0.0,
        6.22,
        6.121,
        0.005925,
        0.16326,
        6.48,
        11.4971,
        19.105,
        0.8938,
        6.54,
        11.4950,
        -22.775,
        1.5707,
        4.3,
        14.08,
        27.80,
        -1.653,
        1.50,
        14.67,
    ]
)  # Tab.1.

def _f0(x):  # logistic smoother, Eq.(13)
    return 1.0 / (np.exp(x) + 1.0)

def _zeta_of_chi(chi):
    a = _a
    return (
        ((a[1] + a[2]*chi + a[3]*chi**3) / (1.0 + a[4]*chi)) * _f0(a[5]*(chi - a[6]))
        + (a[7]  + a[8]*chi)  * _f0(a[9]*(a[10] - chi))
        + (a[11] + a[12]*chi) * _f0(a[13]*(a[14] - chi))
        + (a[15] + a[16]*chi) * _f0(a[17]*(a[18] - chi))
    )

def p_SLy4(rho_mass_energy_kg_m3):
    """
    Input:  ρ = E/c^2 in kg/m^3  (mass–energy density).
    Output: P in Pa.
    """
    rho_cgs = rho_mass_energy_kg_m3 / 1e3          # kg/m^3 -> g/cm^3
    chi = np.log10(rho_cgs)
    zeta = _zeta_of_chi(chi)
    P_cgs = 10.0**zeta                              # dyn/cm^2
    return 0.1 * P_cgs                              # Pa

def rho_SLy4(p_Pa):
    """
    Invert SLy4 to get rest-mass density ρ [kg/m^3] from pressure p [Pa].
    Works with scalars or numpy arrays.
    """
    P_in = np.asarray(p_Pa, dtype=float)

    # valid chi domain of the fit
    chi_min, chi_max = 0.0, 16.0
    p_min = 0.1 * (10.0 ** _zeta_of_chi(chi_min))  # Pa
    p_max = 0.1 * (10.0 ** _zeta_of_chi(chi_max))  # Pa

    def _invert_scalar(P_target):
        if not np.isfinite(P_target) or P_target <= p_min:
            return _RHO_FLOOR
        if P_target >= p_max:
            return (10.0**chi_max) * 1e3  # kg/m^3

        def f(chi):  # root in chi = log10(rho_cgs)
            return 0.1 * (10.0 ** _zeta_of_chi(chi)) - P_target

        chi_root = brentq(f, chi_min, chi_max)
        return (10.0**chi_root) * 1e3  # kg/m^3

    if P_in.ndim == 0:
        return _invert_scalar(P_in.item())
    out = np.empty_like(P_in, dtype=float)
    it = np.nditer(P_in, flags=["multi_index"])
    for x in it:
        out[it.multi_index] = _invert_scalar(float(x))
    return out
