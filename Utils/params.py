import numpy as np

# === USER-DEFINED PARAMETERS (SI) ===
hbar = 1.054571817e-34  # Reduced Planck constant in J s
c = 299792458.0  # Speed of light
G_N = 6.6743e-11 # Gravitational constant 
SM = 1.9885e30  # Solar mass in Kg
M2 = (c**4)/(8*np.pi*G_N)     # Planck mass squared in Kg m /s^2 
M = np.sqrt(M2)  # Planck mass  (~2 * 10^21 in SI)
Lambda = 0.0  # Cosmological constant in m^-2 * M^2
rho0_lightS = 8.1e17 # Density in Kg/m^3
rho0_heavyS = 3.4e18 # Density in Kg/m^3
# Lambda = 1e-52 *M2  # Cosmological constant in m^-2 * M^2

# --- EMG parameters ---
V0 = 2.44
lmbda_EMG = (
    pow(10, 2 * V0) * 20.869 / ((M2) ** 2)
)  # Scalar self-coupling  in s^2 kg^-1 m^-3
# lmbda_EMG = 5e-46   # Scalar self-coupling  in s^2 kg^-1 m^-3
# print(pow(10, 2 * V0) / (3.516 * 1e109))

# --- Higgs parameters ---
v_higgs = 246e9 * 9 * 1e-7  # 1e-16 M
v_higgs_M = v_higgs / M  # Convert eV to kg
# lmbda_higgs = 0.0   # Scalar self-coupling  in s kg^-1 m^-2


def lmbda_to_SI(lmbda_dimless):
    """Convert dimensionless self-coupling to SI units (s^2 kg^-1 m^-3)."""
    return lmbda_dimless / (hbar * c)


def m_ev_to_SI(mu_ev):
    """Covert scalar mass from eV to SI units (m^-1)"""
    return mu_ev * 5.07 * 1e6


# print(hbar * c / M2)
# print(1 / M2)
# print(1 / M**1.9)
# xi_EMG = 0.5
# xi_H = 1e5
# lmbda_H = xi_H**2 / 49000
# print(
#    np.sqrt(
#        (xi_H * (1e18 * c * c - lmbda_H * v_higgs**4) + lmbda_H * v_higgs**2 * M2)
#        / (M2 * lmbda_H * (M2 + xi_H * v_higgs**2))
#    )
# )
