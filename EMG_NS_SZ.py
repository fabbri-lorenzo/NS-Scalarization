# ---------------------------------------------
# EMG Scalarization in Jordan Frame - Solver
# Author: Lorenzo Fabbri (2025)
# Description: Modular Python code to integrate
# the modified TOV + scalar field equations in
# Early Modified Gravity (Jordan frame)
# ---------------------------------------------

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === USER-DEFINED PARAMETERS ===
xi = 2.5       # Non-minimal coupling
lmbda = 10**(2*2.22)    # Scalar self-coupling
K = 100.0        # Polytropic constant
gamma = 2.0      # Polytropic exponent
Lambda = 1e-6     # Cosmological constant 
M = 1     # Planck mass 

# === Equation of State (Polytropic) ===
def rho_from_p(p):
    return (p / K) ** (1 / gamma)

# === Scalar-Tensor Functions ===
def F(sigma):
    return M*M + xi * sigma**2

def dF_dsigma(sigma):
    return 2 * xi * sigma

def V(sigma):
    return 0.25 * lmbda * sigma**4

def dV_dsigma(sigma):
    return lmbda * sigma**3

# === Box F ===
def box_F(r, Psi, sigma, dsigma, dPhi, dPsi, ddsigma):
    F_prime = 2 * xi * sigma * dsigma
    F_double = 2 * xi * (dsigma**2 + sigma * ddsigma)
    return np.exp(-2 * Psi) * (F_double + (2 / r + dPhi - dPsi) * F_prime)

# === System of First-Order ODEs ===
def tov_system(r, y):
    m, sigma, dsigma, p = y

    if p <= 0:
        return [0, 0, 0, 0, 0]  # Stop past surface

    rho = rho_from_p(p)
    F_val = F(sigma)
    F_prime = dF_dsigma(sigma) * dsigma
    Psi = -0.5 * np.log(1 - 2 * m / r)
    e2Psi = np.exp(2 * Psi)

    # Temporary dm for estimating dPsi
    dm_temp = (r**2 / (2 * F_val)) * (
        0 + Lambda + V(sigma) + rho + 0.5 * (1 - 2 * m / r) * dsigma**2
    )

    denom = 1 - 2 * m / r
    dPsi = (m / r**2 - dm_temp / r) / denom if denom > 0 else 0

    # Preliminary Ricci scalar for Klein-Gordon
    R = (1 / F_val) * ((1 - 2 * m / r) * dsigma**2 + 4 * (Lambda + V(sigma)) + rho - 3 * p)

    # d^2sigma/dr^2 (Klein-Gordon)
    ddsigma = e2Psi * (dV_dsigma(sigma) - xi * sigma * R) - (2 / r - dPsi) * dsigma

    # boxF with sigma'' and dPsi included
    dPhi_temp = 0  # initialize to 0 for boxF, will correct later
    boxF_val = box_F(r, Psi, sigma, dsigma, dPhi_temp, dPsi, ddsigma)

    # Recalculate dPhi with updated boxF
    F_second = 2 * xi * (dsigma**2 + sigma * ddsigma)
    dPhi = (r / (2 * F_val)) * (
        F_val * (e2Psi - 1) / r**2 + F_second - dPsi * F_prime
        - e2Psi * boxF_val + 0.5 * dsigma**2 + p * e2Psi - (Lambda + V(sigma)) * e2Psi
    )

    # Final recompute of boxF with correct dPhi
    boxF_val = box_F(r, Psi, sigma, dsigma, dPhi, dPsi, ddsigma)

    # dm/dr
    dm = (r**2 / (2 * F_val)) * (
        boxF_val + Lambda + V(sigma) + rho + 0.5 * (1 - 2 * m / r) * dsigma**2
    )

    # dp/dr
    dp = -(rho + p) * dPhi

    return [dm, dPhi, dsigma, ddsigma, dp]

# === Initial Conditions ===
def initial_conditions(p_c, sigma_c):
    m0 = 0.0
    Phi0 = 0.0
    dsigma0 = 0.0
    return [m0, Phi0, sigma_c, dsigma0, p_c]

# === Integrator Wrapper ===
def integrate_star(p_c, sigma_c, r_max=30.0, r_points=1000):
    r_span = (1e-6, r_max)
    r_eval = np.linspace(*r_span, r_points)
    y0 = initial_conditions(p_c, sigma_c)

    sol = solve_ivp(tov_system, r_span, y0, t_eval=r_eval,
                    method='RK45', rtol=1e-6, atol=1e-9)
    return sol

# === Scalarization Scanner ===
def scan_scalarization(p_c, sigma_c, xi_range, threshold=1e-3):
    global xi
    results = []
    for xi_trial in xi_range:
        xi = xi_trial
        sol = integrate_star(p_c, sigma_c)
        sigma_surface = sol.y[2][-1]
        scalarized = abs(sigma_surface) > threshold
        results.append((xi_trial, sigma_surface, scalarized))
    return results

# === Example Call ===
if __name__ == "__main__":
    p_c = 1e-3
    sigma_c = 1e-6  # Very small to probe spontaneous scalarization

    # Scan over xi to detect scalarization onset
    xi_values = np.linspace(0, 100, 50)
    scan_results = scan_scalarization(p_c, sigma_c, xi_values)

    for xi_val, sigma_surf, active in scan_results:
        status = "SCALARIZED" if active else "no scalarization"
        print(f"xi = {xi_val:.2f} | sigma(R) = {sigma_surf:.2e} | {status}")

    # Plotting scalar field at surface vs xi
    xi_vals_plot = [r[0] for r in scan_results]
    sigma_surf_plot = [r[1] for r in scan_results]

    plt.figure(figsize=(8, 5))
    plt.plot(xi_vals_plot, sigma_surf_plot, marker='o', linestyle='-', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel(r'$\xi$ (non-minimal coupling)')
    plt.ylabel(r'$\sigma(R_*)$ (scalar field at surface)')
    plt.title("Scalar Field at Star Surface vs Coupling Constant ξ")
    plt.grid(True)
    plt.tight_layout()
    plt.show()