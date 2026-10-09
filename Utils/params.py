import numpy as np

# === USER-DEFINED PARAMETERS (SI) ===
hbar = 1.054571817e-34  # Reduced Planck constant in J s
c = 299792458.0  # Speed of light
G_N = 6.6743e-11 # Gravitational constant 
SM = 1.988475e30  # Solar mass in Kg
M2 = (c**4) / (8 * np.pi * G_N)  # Planck mass squared in Kg m /s^2
M = np.sqrt(M2)  # Planck mass  (~2 * 10^21 in SI)
Lambda = 0.0  # Cosmological constant in m^-2 * M^2
rho0_lightS = 8.1e17  # Density in Kg/m^3
rho0_heavyS = 3.4e18  # Density in Kg/m^3
# Lambda = 1e-52 *M2  # Cosmological constant in m^-2 * M^2

# --- EMG parameters ---
V0 = 2.44
lmbda_EMG = (
    pow(10, 2 * V0) * 20.869 / ((M2) ** 2)
)  # Scalar self-coupling  in s^2 kg^-1 m^-3
# lmbda_EMG = 5e-46   # Scalar self-coupling  in s^2 kg^-1 m^-3
# print(pow(10, 2 * V0) / (3.516 * 1e109))


# --- Higgs parameters ---
v_higgs = 246e9  # 1e-16 M
v_higgs_M = v_higgs / M  # Convert eV to kg
e = 1.602176634e-19  # J/eV


def lmbda_to_SI(lmbda_dimless):
    """Convert dimensionless self-coupling to SI units (s^2 kg^-1 m^-3)."""
    return lmbda_dimless / (hbar * c)


def m_ev_to_SI(mu_ev):
    """Covert scalar mass from eV to SI units (m^-1)"""
    return mu_ev * 5.07 * 1e6


def nu_ev_to_SI(nu_ev):
    """Convert vacuum expectation value from eV to SI units (kg)."""
    return nu_ev * e / np.sqrt(hbar * c)


########
lmbda_test = lmbda_to_SI(1e-60)
mu2_test = m_ev_to_SI(1e-5) ** 2
lmbda_Higgs = lmbda_to_SI(0.5)
xi_Higgs = 49000 * np.sqrt(lmbda_Higgs)
rho_3P = 4.5e34
xi_EMG = 0.2


def sigma_min(xi, lmbda, nu2, mu2, rho_3P):
    sigma_min = np.sqrt((xi * rho_3P - M2 * mu2) / (M2 * lmbda) + nu2)
    return sigma_min


sigma_EMG = sigma_min(xi=100, lmbda=lmbda_test, nu2=0.0, mu2=0.0, rho_3P=rho_3P) / M
# print("\nEMG, sigma min/M = ", f"{sigma_EMG:.2e}")

sigma_Higgs = (
    sigma_min(
        xi=xi_Higgs,
        lmbda=lmbda_Higgs,
        nu2=v_higgs**2,
        mu2=0.0,
        rho_3P=rho_3P,
    )
    / M
)
# print("\nHiggs, sigma min/M = ", f"{sigma_Higgs:.2e}")
# sigma_ULA = sigma_min(xi=5, lmbda=0.0, nu2=0.0, mu2=mu2_test, rho_3P=rho_3P) / M
# print("\nULA, sigma min/M = ", f"{sigma_ULA:.2e}")
# print(lmbda_to_SI(1e-60))
#print("\n", np.sqrt(100 * rho_3P / (M2 * lmbda_to_SI(1e-60)))/M)
