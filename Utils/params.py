import numpy as np

# === USER-DEFINED PARAMETERS (SI) ===
c = 299792458.0  # Speed of light
G_N = 6.6743e-11 # Gravitational constant 
SM = 1.9885e30  # Solar mass in Kg
M2 = (c**4)/(8*np.pi*G_N)     # Planck mass squared in Kg m /s^2 
M=np.sqrt(M2)  # Planck mass 
Lambda = 0.0  # Cosmological constant in m^-2 * M^2
rho0_lightS = 8.1e17 # Density in Kg/m^3
rho0_heavyS = 3.4e18 # Density in Kg/m^3
# Lambda = 1e-52 *M2  # Cosmological constant in m^-2 * M^2

# --- EMG parameters ---
# V0=2.44
# lmbda_EMG = pow(10,2*V0)*20.869/((M2)**2)   # Scalar self-coupling  in s^2 kg^-1 m^-3
# lmbda_EMG = 5e-46   # Scalar self-coupling  in s^2 kg^-1 m^-3

# --- Higgs parameters ---
# v_higgs = 246e9 *9*1e-7
# lmbda_higgs = 0.0   # Scalar self-coupling  in s kg^-1 m^-2
