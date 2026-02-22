import numpy as np
import time, os, csv
from concurrent.futures import ProcessPoolExecutor
import io
import contextlib
import traceback

from Utils.params import M, SM, rho0_lightS, rho0_heavyS, lmbda_to_SI, m_ev_to_SI
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.graphics import plotResults_multi, custom_print
from Utils.shooting import (
    diagnostic_scan,
    probe_brackets,
)
from Utils.analysis import (
    integrate_star,
    node_count_to_2R,
    adm_mass_from_sol,
    scal_charge_from_sol,
)

# ---------- USER DEFINED PARAMETERS ----------
p_eqState = p_SLy4  
rho_eqState = rho_SLy4  
frac_pc = 1e-10  # Fraction of p_c to stop integration

xi = 2

# m_val = 0.0
m = 1e-20

lmbda = 0.0
# lmbda = 1e-40

nu = 0.0
# nu = M * 1e-6

method = "BDF" if (xi < 0.0 or lmbda > 1e-40 or nu > M * 1e-1) else "RK45"

# Shooting method parameters
a, b = 1e-10 * M, 10 * M  #  Bracket for σ0
target_shooting = [0.0] if nu == 0.0 else [abs(nu), -abs(nu)]

abs_cut = a * 1e-2
rel_cut = 1e-2
merge_tol = a * 1e-2

# ---------------------------------------------


def run_solve_model_captured(rho0):
    """
    Worker wrapper: capture all prints + errors from solve_model into a string,
    return (label, log_text, ok_bool).
    """
    label = (
        "Light"
        if rho0 == rho0_lightS
        else ("Heavy" if rho0 == rho0_heavyS else str(rho0))
    )

    buf = io.StringIO()
    ok = True

    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            solve_model(rho0)
        except Exception:
            ok = False
            # include traceback inside the captured output
            traceback.print_exc()

    return label, buf.getvalue(), ok


def solve_model(rho0):
    t0 = time.perf_counter()

    star_weight, sw = "", ""

    if rho0 == rho0_lightS:
        star_weight = "L"  # Light star
        sw = "Light"
    elif rho0 == rho0_heavyS:
        star_weight = "H"  # Heavy star
        sw = "Heavy"

    xi_val, m_val, lmbda_val, nu_val = xi, m, lmbda, nu
    custom_print(
        f"\nξ = {xi_val} | µ = {m_val:.2e} | λ = {lmbda_val:.2e} | ν = {nu_val:.2e} | ν/M = {nu_val/M:.2e} | {sw} star",
        style="bold",
    )

    m_val = m_ev_to_SI(m_val)  # Convert from eV to SI units
    lmbda_val = lmbda_to_SI(lmbda_val)  # Convert from eV to SI units
    m2_val = m_val**2
    nu_abs = abs(nu_val)

    r0 = 1e-2  # m
    r_max = 3e5  # m

    # --- adaptive pre-scan (coarse + local refinement) ---
    scan = diagnostic_scan(
        r0=r0,
        r_max=r_max,
        p_eqState=p_eqState,
        rho_eqState=rho_eqState,
        xi=xi_val,
        m2=m2_val,
        lmbda=lmbda_val,
        nu_val=nu_val,
        rho0=rho0,
        frac_pc=frac_pc,
        method=method,
        a=a,
        b=b,
        n_coarse=41,  # cheap first pass
        n_refine=81,  # refine only promising regions
        target=target_shooting,  # <-- FIXED
        expand_coarse_points=1,
        detect_near_zero=True,
        compress_brackets=True,
    )

    brackets = scan["brackets"]

    # diagnostics from adaptive scan
    F_coarse = scan["F_coarse"]
    custom_print(
        f"[diagnostics] adaptive scan over [{a/M:.1e},{b/M:.1e}]*M: "
        f"{np.sum(np.isfinite(F_coarse))}/{len(F_coarse)} finite coarse samples, "
        f"refined regions={len(scan['regions'])}, Brent brackets={len(brackets)}",
        color="gray",
    )

    if len(brackets) == 0:
        custom_print(
            "No promising brackets found in the given range. Scalarization does not occur here.",
            color="magenta",
        )
        t_tot = time.perf_counter() - t0
        custom_print(f"\nExecution time: {t_tot:.0f} s", style="dim")
        return None

    # --- 1) Find ALL σ0 roots (absolute-first, then relative) ---

    s0_list, s_maxes = probe_brackets(
        brackets=brackets,
        r0=r0,
        r_max=r_max,
        p_eqState=p_eqState,
        rho_eqState=rho_eqState,
        xi_val=xi_val,
        rho0=rho0,
        frac_pc=frac_pc,
        method=method,
        m2_val=m2_val,
        lmbda_val=lmbda_val,
        nu_val=nu_val,
        abs_threshold=abs_cut,
        tol_relative=rel_cut,
        merge_tol=merge_tol,
        target=target_shooting,
        idx_sigma=2,
    )

    if not s0_list:
        raise RuntimeError("No σ0 roots found; see diagnostics above.")

    # align σ(r_max) with σ0 list (order-preserving)
    smax_by_s0 = {s: sm for s, sm in zip(s0_list, s_maxes)}

    # --- 2) For each σ0: integrate to 2R and count nodes ---
    per_mode = {}     # n -> list of dicts with entries

    for s0 in s0_list:
        sol, mu2_log, R_star_m = integrate_star(
            s0,
            p_eqState,
            rho_eqState,
            r0,
            r_max,
            xi_val,
            m2_val,
            lmbda_val,
            nu_val,
            rho0,
            frac_pc,
            method=method,
            stop_at_2r=True,
            record_mu2=True,
        )

        if R_star_m is None:
            custom_print(f"[WARN] R_* not found for σ₀={s0/M:.4e} M_Pl; skipping 2R node check.", color="yellow")
            continue
        if nu_val != 0.0:
            diff_plus = abs(smax_by_s0[s0] - nu_abs)
            diff_minus = abs(smax_by_s0[s0] + nu_abs)
            target = nu_abs if (diff_plus <= diff_minus) else -nu_abs
        else:
            target = 0.0

        n_nodes = node_count_to_2R(sol, R_star_m, target, idx_sigma=2)
        per_mode.setdefault(n_nodes, []).append(
            dict(
                s0=s0, sol=sol, mu2_log=mu2_log, R_star_m=R_star_m, smax=smax_by_s0[s0]
            )
        )

    # --- 3) Select modes until we have as many solutions as sign-change brackets ---
    sigma0_by_mode = {}  # maps node count to list of chosen σ₀ values

    # print('\n')
    for n in sorted(per_mode.keys()):
        cands = per_mode.get(n, [])
        if not cands:
            custom_print(f"[MISS] No candidate with n={n} nodes.", color="red")
            continue

        if nu_val != 0.0:
            # handle both +ν and −ν vacua
            for sign in (+1, -1):
                group = []
                for d in cands:
                    # value of σ at r_max from the stored solution
                    val = float(d["smax"])
                    diff_plus = abs(val - nu_abs)
                    diff_minus = abs(val + nu_abs)
                    sign_val = +1 if (diff_plus <= diff_minus) else -1
                    if sign_val == sign:
                        # distance to the appropriate vacuum
                        diff_to_vac = diff_plus if sign == +1 else diff_minus
                        d_candidate = d.copy()
                        d_candidate["vacuum_sign"] = sign
                        d_candidate["sigma_rmax"] = val
                        d_candidate["diff_to_vac"] = diff_to_vac
                        group.append(d_candidate)
                if group:
                    # choose the candidate closest to its vacuum
                    pick = min(group, key=lambda d: d["diff_to_vac"])
                    sigma0_by_mode.setdefault(n, []).append(pick["s0"])
                    vac_label = "+" if sign == +1 else "-"
                    custom_print(
                        f"n={n}, vacuum={vac_label}ν: σ₀ = {pick['s0']/M:.4e} M_Pl",
                        color="cyan",
                    )
        else:
            # ν=0: single vacuum at zero; choose the smallest residual
            pick = min(cands, key=lambda d: abs(d["smax"]))
            sigma0_by_mode.setdefault(n, []).append(pick["s0"])
            custom_print(f"[OK] n={n}: σ₀ = {pick['s0']/M:.4e} M_Pl", color="cyan")

        # flatten the chosen (n, σ₀) pairs for further integration
    selected_pairs = [(n, s0) for n, s_list in sigma0_by_mode.items() for s0 in s_list]
    if not selected_pairs:
        custom_print(
            "[WARN] No σ₀ solutions selected. Consider widening the bracket or relaxing thresholds.",
            color="yellow",
        )

    # --- 4) Final integrate (again) for selected modes for plotting + diagnostics ---
    plot_entries = []
    vacuum_sols ={"+": 0, "-": 0, "0": 0} 

    path = f"Results/single/{star_weight}/lmbda={lmbda:.0e}_nu={nu:.0e}_xi={xi:.2f}/"
    if m_val != 0.0:
        path = f"Results/single/{star_weight}/lmbda={lmbda:.0e}_m={m:.0e}_xi={xi:.2f}/"
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
                "mode_n",
                "vacuum_sign",   # +1 / -1 / 0 (for nu=0)
                "sigma0_over_M",
                "ADM_over_Msun",
                "Q_over_Msun",
                "Q_over_ADM",
                "R_star_km"
            ])

        for n, s0 in selected_pairs:
            sol, mu2_log, R_star_m = integrate_star(
                s0,
                p_eqState,
                rho_eqState,
                r0,
                r_max,
                xi_val,
                m2_val,
                lmbda_val,
                nu_val,
                rho0,
                frac_pc,
                method=method,
                stop_at_2r=True,
                record_mu2=True,
            )
            r_star_km = (R_star_m/1e3) if (R_star_m is not None) else np.nan

            sol_bnd, _, _ = integrate_star(
                s0,
                p_eqState,
                rho_eqState,
                r0,
                r_max,
                xi_val,
                m2_val,
                lmbda_val,
                nu_val,
                rho0,
                frac_pc,
                method=method,
                stop_at_2r=False,
                record_mu2=False,
            )

            ADM_mass = adm_mass_from_sol(sol_bnd, k_tail=15)
            scalar_charge = scal_charge_from_sol(
                sol_bnd, m2_val, lmbda_val, nu_val, r_max
            )

            if nu_val != 0.0:
                nu_abs = abs(nu_val)
                val = float(sol_bnd.y[2, -1])
                diff_plus = abs(val - nu_abs)
                diff_minus = abs(val + nu_abs)
                vac_sign = +1 if (diff_plus <= diff_minus) else -1
            else:
                vac_sign=0

            custom_print(f"\nResults for n={n} mode", style="bold")
            print("ADM mass / M_sun = ", f"{ADM_mass/SM:.2e}")
            print("Scalar charge / M_sun = ", f"{scalar_charge/SM:.2e}")
            print("Q/M ratio = ", f"{scalar_charge/ADM_mass:.2e}")

            vac_label = "0" if vac_sign == 0 else ("+" if vac_sign == +1 else "-")
            vacuum_sols[vac_label] += 1

            writer.writerow([n, vac_sign, f"{s0/M:.3e}", f"{ADM_mass / SM:.3e}", f"{scalar_charge / SM:.3e}",
                             f"{scalar_charge/ADM_mass:.3e}", f"{r_star_km:.3e}"])

            r_mu, mu2 = (np.array(mu2_log, dtype=float).T
                         if len(mu2_log) else (np.array([]), np.array([])))
            plot_entries.append(
                {
                    "label": f"n={n}",
                    "sol": sol,
                    "r_star": r_star_km,
                    "r_mu": r_mu,
                    "mu2": mu2,
                    "vacuum_sign": vac_sign,  # +1 for +ν, –1 for –ν, 0 for ν=0
                }
            )

    t_tot = time.perf_counter() - t0
    custom_print(f"\nExecution time: {t_tot:.0f} s", style="dim")

    # --- 5) Plots
    plotResults_multi(plot_entries, nu_val,vacuum_sols, path)


def main():
    jobs = [rho0_lightS, rho0_heavyS]

    with ProcessPoolExecutor(max_workers=2) as ex:
        for label, out, ok in ex.map(run_solve_model_captured, jobs):
            # parent prints in-order, no interleaving
            custom_print(f"\n===== {label} star log (ok={ok}) =====", style="bold")
            print(out, end="")


if __name__ == "__main__":
    main()
