from Utils.params import c, G_N, M2, Lambda, lmbda_higgs, v_higgs, rho0_lightS,rho0_heavyS
import numpy as np

# === INITIAL CONDITIONS EMG (SI) ===
def initial_conditions(r0, sigma0, p_eqState):
    rho0 = rho0_lightS # Density in Kg/m^3 
    p0 = p_eqState(rho0)
    m0 = (4/3) * np.pi * rho0 * r0**3 *G_N/(c*c) #reduced mass
    Phi0 = 0.0  # Metric function at center
    dsigma0 = 0.0  # Initial scalar field derivative value in M_Pl units
    return [p0, m0, Phi0, sigma0, dsigma0]

# === Scalar-Tensor Functions ===
def F(sigma, xi):
    return M2+ xi* sigma**2

def dF_dsigma(sigma,xi):
    return 2 * xi * sigma

def V(sigma):
    return 0.25 * lmbda_higgs * (sigma*sigma-v_higgs*v_higgs)**2

def dV_dsigma(sigma):
    return lmbda_higgs * sigma* (sigma*sigma-v_higgs*v_higgs)

# === System of First-Order ODEs ===
def make_tov_system(p_c,frac_pc, rho_eqState, xi, mu2_recorder=None):
    
    def tov_system(r, y):
      p, m, Phi, sigma, sigma1 = y

      if p < p_c*frac_pc:
              p = 0.0
              eps = 0.0
      else:
              rho = rho_eqState(p)
              eps = rho * c * c
          
      den = r-2*m
      
      F_val = F(sigma, xi)
      #print(F_val)
      #print(V(sigma))
      F_prime = dF_dsigma(sigma, xi) * sigma1 
      
      # Check for near Schwarzschild condition  
      if den <= 0.0:
       raise RuntimeError(f"r ≤ 2m (would hit a horizon) at r={r:.3g}")
      
      # Positive mass sanity check 
      if m <= 0.0:
       raise RuntimeError("Mass became non-positive")
      
      # dPhi/dr
      dPhi = (2/(2*F_val + r*F_prime)) *( m*F_val/(r*den) + 0.25*r * sigma1**2 + (p-Lambda -V(sigma))*r*r/(2*den)- F_prime)
      
      # dsigma/dr
      dsigma = sigma1
      
      # Definition of \bar{m} for convenience
      m_bar = m/r + pow(2*F_val + r*F_prime,-1) *(-2*m*F_val/r + 2*r*xi*den*((dPhi +2/r)*sigma*sigma1 +sigma1**2 )+r**2 *( eps + Lambda + V(sigma)+den*(sigma1**2) /(2*r)))
      
      #d^2sigma/dr^2)
      dsigma1 = ((2*F_val + r*F_prime)/((F_val+6*xi**2*sigma**2)*(2*F_val+r*F_prime)-(3*xi*sigma*F_prime+F_val*sigma1)*2*r*xi*sigma)) * (F_val*(r/den *dV_dsigma(sigma) -dPhi*sigma1-2*sigma1/r+ m_bar*sigma1/den- m*sigma1/(r*den)) - xi*sigma*(sigma1**2 + r/den *(4*(Lambda+V(sigma))+eps - 3*p)+6*xi*((dPhi-(r*m_bar+m)/(r*den)+2/r) *sigma*sigma1 +sigma1**2))) 
      
      # dm/dr
      dm = m_bar + (2*r*xi*den)*sigma*dsigma1/(2*F_val+r*F_prime) 
      
      # dp/dr
      dp = -(eps + p) * dPhi
      
      #effective mass squared
      mu2= -(xi/F_val)*(den/r* sigma1**2 +4*(Lambda + V(sigma)) + eps - 3*p + 6*xi*den/r*(dPhi-(r*dm-m)/(r*den) +2/r)*sigma*sigma1+sigma1**2 +sigma*dsigma1)
      if mu2_recorder is not None:
            # store (r, μ²) without touching solver tolerances/state
            mu2_recorder((r, mu2))
                  
      return [dp, dm, dPhi, dsigma, dsigma1]
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
