import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Utils.params import c, G_N, M2, SM, rho0_lightS, rho0_heavyS
from Utils.EOS import p_SLy4, rho_SLy4

p_eqState = p_SLy4       # Pressure as function of density
rho_eqState = rho_SLy4   # Density as function of pressure

# === Initial Conditions ===
def initial_conditions(r0):
    rho0 = rho0_lightS # Density in Kg/m^3 - will give 1.12 solar masses in GR
    p0 = p_eqState(rho0)
    m0 = (4/3) * np.pi * rho0 * r0**3 *G_N/(c*c) #reduced mass
    Phi2 = (p0+2*rho0/3)/(4*M2) * r0**2  # from TOV, O(r^2) expansion
    Phi_c = 0.0
    p2 = -(p0+rho0*c*c)*Phi2 
    p_c = p0 + p2* r0**2
    m_c = m0 
    return [p_c, m_c, Phi_c]


# === System of First-Order ODEs ===
def tov_system(r, y):
    p, m, Phi= y
    
    if p <= 1e-3:
        return [0.0, 0.0, 0.0]   # stop evolving once surface reached

    rho = rho_eqState(p)
    eps = rho*c*c 
    
    den = r-2*m
    
    # Check for near Schwarzschild condition  
    if den <= 0:
     raise RuntimeError(f"r ≤ 2m (would hit a horizon) at r={r:.3g}")
    
    # dPhi/dr
    dPhi = (2*m + r**3*(p/M2)) / (2*r*den)

    # dm/dr
    dm = r*r/2 * (eps/M2)
    
    # dp/dr
    dp = -(eps + p) * dPhi
     
    #print(" P=",p, "\n")
    return [dp, dm, dPhi]

# === Integrator Wrapper ===
def integrate_star(r0, r_max):
    r_span = (r0, r_max)
    y0 = initial_conditions(r0)
    p_c = y0[0]

    def surface(r, y):
        return y[0] - 1e-15*p_c  # pressure
    surface.terminal = True
    surface.direction = -1

    sol = solve_ivp(
        tov_system,
        r_span,
        y0,
        method="RK45",
        rtol=1e-6,
        atol=np.array([1e-8, 1e-10, 1e-10]),
        events=surface,
    )

    print("» solver msg:", sol.message)
    return sol


# === Testing GR reduction ===
if __name__ == "__main__":

 r0= 1e-2  # Start radius in m 
 r_max=3e5  # Max radius in m 
 r_points=int(1e8)
 r_span = (r0, r_max)
 r_eval = np.linspace(*r_span, r_points)
 sol = integrate_star(r0,r_max)
 
 Phi_surf = sol.y[2, -1]
 Phi_norm = sol.y[2] - Phi_surf
 
 print("Integration successful:", sol.success)
 print("Final radius:", sol.t[-1]/1e3, "Km")
 print("Final pressure:", sol.y[0][-1], "Pa")
 print("Final mass:", sol.y[1][-1]*(c*c/G_N)/SM, "Solar Masses")  # in Solar Masses
 
 #--- Pressure profile ---
 plt.figure()
 plt.plot(sol.t/1e3, sol.y[0], color='blue')
 plt.xlabel('r (Km)')
 plt.ylabel('P (Pa)')
 plt.grid(True)
 plt.title('Pressure GR')
 plt.savefig("Tests/Results_rhoe18/pressureGR.png", dpi=300)
 plt.show()
 
 # --- Mass and metric ---
 plt.plot(sol.t/1e3, sol.y[1]*(c*c/G_N)/(SM),color='indigo', label='Mass m(r) in Solar Masses')
 plt.plot(sol.t/1e3, np.exp(Phi_norm), color='darkcyan', label='Metric Function e^2Φ(r)')
 plt.xlabel('r (Km)')
 plt.legend()
 plt.grid(True)
 plt.title('Mass Metric GR')
 plt.savefig("Tests/Results_rhoe18/mass_metricGR.png", dpi=300)
 plt.show()
