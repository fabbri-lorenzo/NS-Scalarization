# scan_xi.py
import os, csv, time
import numpy as np
from tqdm import tqdm  

from Utils.params import M, SM, rho0_lightS, rho0_heavyS, lmbda_to_SI
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.graphics import custom_print
from Utils.shooting import diagnostic_scan, probe_brackets
from Utils.analysis import integrate_star, node_count_to_2R, adm_mass_from_sol, scal_charge_from_sol

# -------- fixed inputs --------
p_eqState   = p_SLy4
rho_eqState = rho_SLy4
frac_pc     = 1e-10
rho0        = rho0_lightS
r0, r_max   = 1e-2, 3e5

#xi_min, xi_max, xi_step = -1, 1, 0.05
xi_min, xi_max, xi_step = -100, 100, 1
lmbda = 1/M**2.2
nu   = 1e-5*M
#nu   = 246.0 * 9.0 * 1e2

N_efolds = 55

a, b = 1e-8 * M, 0.5*M
abs_cut   = a * 1e-2
rel_cut   = 1e-2
merge_tol = a* 1e-2
# --------------------------------

def lambda_from_xi(xi):
    lmbda_NU = (xi / N_efolds) ** 2 * 3.0 / (0.02656 ** 4) 
    lmbda_SI = lmbda_to_SI(lmbda_NU)
    return lmbda_SI

def main():
    t0 = time.perf_counter()
    star_weight = "L" if rho0 is rho0_lightS else "H"
    sw = "Light" if rho0 is rho0_lightS else "Heavy"

    out_dir = f"Results/scan/{star_weight}/nu={nu:.0e}"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir,f"lmbda={lmbda:.0e}.csv")
    
    custom_print(
        f"\nξ in ({xi_min},{xi_max}) | λ = {lmbda:.2e} | ν = {nu:.2e} | {sw} star",
        style="bold",
    )

    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
                "xi", "lambda", "nu",
                "rho0_tag",
                "scalarized",
                "mode_n",
                "vacuum_sign",   # +1 / -1 / 0 (for nu=0)
                "sigma0_over_M",
                "ADM_over_Msun",
                "Q_over_Msun",
                "Q_over_ADM",
                "R_star_km"
            ])

        for xi_val in tqdm(np.arange(xi_min, xi_max +xi_step, xi_step),
                           desc="Scanning ξ values", unit="ξ"):
            lmbda_val = lmbda if lmbda is not None else lambda_from_xi(xi_val)
            target = [0.0] if nu == 0.0 else [abs(nu), -abs(nu)]
            method = "BDF" if (xi_val < 0.0 or lmbda > 1e-40) else "RK45" 

            try:
                F, brackets = diagnostic_scan(
                    r0, r_max,
                    p_eqState, rho_eqState,
                    xi_val, lmbda_val, nu,
                    rho0, frac_pc, method,
                    a, b,
                    n=201,
                    target=target,
                )
                #custom_print(
                # f"[diagnostics] scan over [{a/M:.1e},{b/M:.1e}]*M: "
                # f"{np.sum(np.isfinite(F))}/{len(F)} finite samples, sign-change brackets={len(brackets)}",
                # color="gray",
                #  )

                if len(brackets) == 0:
                    w.writerow([xi_val, f"{lmbda_val:.1e}", f"{nu:.1e}", star_weight, 0, "", "", "", "", "", "", ""])
                    continue

                s0_list, s_maxes = probe_brackets(
                    brackets=brackets,
                    r0=r0, r_max=r_max,
                    p_eqState=p_eqState, rho_eqState=rho_eqState,
                    xi_val=xi_val, rho0=rho0, frac_pc=frac_pc, method=method,
                    lmbda_val=lmbda_val, nu_val=nu,
                    abs_threshold=abs_cut, tol_relative=rel_cut, merge_tol=merge_tol,
                    target=target, idx_sigma=2,
                )

                if not s0_list:
                    w.writerow([xi_val, f"{lmbda_val:.1e}", f"{nu:.1e}", star_weight, 0, "", "", "", "", "", "", ""])
                    continue

                smax_by_s0 = {s: sm for s, sm in zip(s0_list, s_maxes)}
                
                nu_abs = abs(nu)
                
                per_mode = {}
                for s0 in s0_list:
                    sol, _, R_star_m = integrate_star(
                        s0, p_eqState, rho_eqState, r0, r_max,
                        xi_val, lmbda_val, nu, rho0, frac_pc, method,
                        stop_at_2r=True, record_mu2=False,
                    )
                    if R_star_m is None:
                        continue
                    if nu != 0.0:
                      diff_plus = abs(smax_by_s0[s0] - nu_abs)
                      diff_minus = abs(smax_by_s0[s0] + nu_abs)
                      target = nu_abs if (diff_plus <= diff_minus) else -nu_abs
                    else:
                      target = 0.0 
                       
                    n_nodes = node_count_to_2R(sol, R_star_m, target, idx_sigma=2)
                    per_mode.setdefault(n_nodes, []).append(dict(s0=s0, sol=sol, R_star_m=R_star_m, smax=smax_by_s0[s0]))

                selected = []
                for n, cands in sorted(per_mode.items()):
                    if not cands:
                        continue
                    if nu != 0.0:
                        for sign in (+1, -1):
                            group = []
                            for d in cands:
                                val = float(d["sol"].y[2, -1])
                                diff_plus = abs(val - nu_abs)
                                diff_minus = abs(val + nu_abs)
                                sign_val = +1 if (diff_plus <= diff_minus) else -1
                                if sign_val == sign:
                                    diff_to_vac = diff_plus if sign == +1 else diff_minus
                                    group.append((diff_to_vac, d["s0"]))
                            if group:
                                s0_pick = min(group, key=lambda t: t[0])[1]
                                selected.append((n, s0_pick, sign))
                    else:
                        s0_pick = min(cands, key=lambda d: abs(d["smax"]))["s0"]
                        selected.append((n, s0_pick, 0))

                if not selected:
                    w.writerow([xi_val, f"{lmbda_val:.1e}", f"{nu:.1e}", star_weight, 0, "", "", "", "", "", "", ""])
                    continue

                for n, s0, vac_sign in selected:
                    R_star_m = next(d["R_star_m"] for d in per_mode[n] if d["s0"] == s0)
                    r_star_km = (R_star_m / 1e3) if (R_star_m is not None) else np.nan

                    sol_bnd, _, _ = integrate_star(
                        s0, p_eqState, rho_eqState, r0, r_max,
                        xi_val, lmbda_val, nu, rho0, frac_pc, method,
                        stop_at_2r=False, record_mu2=False,
                    )
                    ADM_mass      = adm_mass_from_sol(sol_bnd, k_tail=15)
                    scalar_charge = scal_charge_from_sol(sol_bnd, lmbda_val, nu, r_max)

                    w.writerow([
                        xi_val, f"{lmbda_val:.1e}", f"{nu:.1e}",
                        star_weight,
                        1,
                        n,
                        vac_sign,
                        s0 / M,
                        ADM_mass / SM,
                        scalar_charge / SM,
                        (scalar_charge / ADM_mass) if ADM_mass != 0 else np.nan,
                        r_star_km
                    ])

            except Exception as e:
                custom_print(f"[ξ={xi_val}] error: {e}", color="red")
                w.writerow([xi_val, f"{lmbda_val:.1e}", f"{nu:.1e}", star_weight, 0, "", "", "", "", "", "", ""])
                continue

    custom_print(f"\nDone in {time.perf_counter() - t0:.1f}s → {out_csv}", style="dim")

if __name__ == "__main__":
    main()
