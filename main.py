import numpy as np

from Utils.params import M, SM, rho0_lightS, rho0_heavyS
from Utils.shooting import shoot_sigma0 
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.graphics import printResults_multi, custom_print
from Utils.diagnostic import (
    diagnostic_scan,
    print_candidate_info,
    probe_brackets_with_brent,
)
from Utils.solver import (
    integrate_star,
    node_count_to_2R,
    adm_mass_from_sol,
    scal_charge_from_sol,
)

import time
import csv

# ---------- USER DEFINED PARAMETERS ----------
p_eqState = p_SLy4  # Choose the EOS
rho_eqState = rho_SLy4  # Choose the EOS
frac_pc = 1e-10  # Fraction of p_c to stop integration

rho0 = rho0_lightS
xi_val = 10
lmbda_val = 0.0
# lmbda_val = 1 / M**2
# lmbda_val = xi_val**2 * 1e-10 / 4.165
nu_val = 0.0
# nu_val = 246e9 * 9e-7


# Shooting method parameters
a, b = 1e-8 * M, 0.5 * M  #  Bracket for σ0
# Increase the number of seeds to sample the bracket finely
n_seeds = 25
target_shooting = [0.0] if nu_val == 0.0 else [abs(nu_val), -abs(nu_val)]
# ---------------------------------------------


# ---------- main ----------
if __name__ == "__main__":
    t0 = time.perf_counter()

    star_weight, sw = "", ""

    if rho0 == rho0_lightS:
        star_weight = "L"  # Light star
        sw = "Light"
    elif rho0 == rho0_heavyS:
        star_weight = "H"  # Heavy star
        sw = "Heavy"

    custom_print(
        f"\nxi = {xi_val} | λ = {lmbda_val:.2e} | ν = {nu_val:.2e} | {sw} star",
        style="bold",
    )

    r0 = 1e-2  # m
    r_max = 3e5  # m
    r_eval = np.linspace(r0, r_max, int(1e6))  # (not strictly needed here)

    # --- pre-scan (coarse) for visibility
    S, F, brackets = diagnostic_scan(
        r0,
        r_max,
        p_eqState,
        rho_eqState,
        xi_val,
        lmbda_val,
        nu_val,
        rho0,
        frac_pc,
        a,
        b,
        n=201,
        target=target_shooting,
    )
    custom_print(
        f"[diagnostics] scan over [{a/M:.1e},{b/M:.1e}]*M: "
        f"{np.sum(np.isfinite(F))}/{len(F)} finite samples, sign-change brackets={len(brackets)}",
        color="gray",
    )

    # Set acceptance thresholds to use consistently across root-finding and bracketing
    abs_cut = a * 1e-2
    rel_cut = 1e-2
    merge_tol = 1e-10 * M

    # --- 1) Find ALL σ0 roots via fsolve (absolute-first, then relative) ---
    s0_list = shoot_sigma0(
        integrate_fn=integrate_star,
        p_eqState=p_eqState,
        rho_eqState=rho_eqState,
        r0=r0,
        r_max=r_max,
        xi=xi_val,
        bracket=(a, b),  # σ≥0 (σ→-σ symmetry)
        rho0=rho0,
        frac_pc=frac_pc,
        target=target_shooting,
        lmbda=lmbda_val,
        nu=nu_val,
        abs_threshold=abs_cut,
        tol_relative=rel_cut,
        sigma0_min_abs=0.0,  # allow tiny amplitudes near threshold
        n_seeds=n_seeds,
        xtol=1e-12,
        idx_sigma=2,
        merge_tol=merge_tol,  # tighter de-dup to keep close roots distinct
    )

    # --- fallback: if empty or missing roots, use Brent per bracket via diagnostic helper ---
    if (not s0_list) or (len(brackets) > len(s0_list)):
        custom_print(
            "[fallback] probing each sign-change sub-bracket with Brent’s method...",
            color="yellow",
        )
        s0_list = probe_brackets_with_brent(
            brackets=brackets,
            existing_s0_list=s0_list,
            r0=r0,
            r_max=r_max,
            p_eqState=p_eqState,
            rho_eqState=rho_eqState,
            xi_val=xi_val,
            rho0=rho0,
            frac_pc=frac_pc,
            lmbda_val=lmbda_val,
            nu_val=nu_val,
            abs_threshold=abs_cut,
            tol_relative=rel_cut,
            merge_tol=merge_tol,
            target=target_shooting,
            idx_sigma=2,
        )

    if not s0_list:
        # as a last hint to the user: report best dips
        custom_print(
            "[fallback] still empty; reporting |σ(r_max)| minima as hints:",
            color="yellow",
        )
        # sort the coarse scan by absolute residual and inspect a few points
        for s, f in sorted(zip(S, F), key=lambda t: abs(t[1]))[:6]:
            try:
                # compute sigma(r_max) directly for this σ₀
                sol, _, _ = integrate_star(
                    s,
                    p_eqState,
                    rho_eqState,
                    r0,
                    r_max,
                    xi_val,
                    lmbda_val,
                    nu_val,
                    rho0,
                    frac_pc,
                    stop_at_2r=False,
                    record_mu2=False,
                )
                sigma_val = float(sol.y[2, -1])
                sigma_abs = abs(sigma_val)
            except Exception:
                # fall back to absolute residual if integration fails
                sigma_abs = abs(f)
            custom_print(
                f"   near σ0/M≈{s/M:.3e}: |σ(r_max)|/M≈{sigma_abs/M:.3e}",
                color="yellow",
            )
        raise RuntimeError("No σ0 roots found; see diagnostics above.")

    # --- print diagnostics for accepted candidates
    s_maxes = print_candidate_info(
        s0_list,
        rho0,
        r0,
        r_max,
        xi_val,
        lmbda_val,
        nu_val,
        p_eqState,
        rho_eqState,
        frac_pc,
        target_shooting,
    )
    # align σ(r_max) with σ0 list (order-preserving)
    smax_by_s0 = {s: sm for s, sm in zip(s0_list, s_maxes)}

    # --- decide how many solutions to keep based on sign-change brackets
    target_solutions = len(brackets)

    if not s0_list:
        raise RuntimeError("No σ0 roots found in the given bracket.")

    # --- 2) For each σ0: integrate to 2R and count nodes ---
    per_mode = {}     # n -> list of dicts with entries
    all_entries = []  # for plotting

    for s0 in s0_list:
        sol, mu2_log, R_star_m = integrate_star(
            s0,
            p_eqState,
            rho_eqState,
            r0,
            r_max,
            xi_val,
            lmbda_val,
            nu_val,
            rho0,
            frac_pc,
            stop_at_2r=True,
            record_mu2=True,
        )

        if R_star_m is None:
            custom_print(f"[WARN] R_* not found for σ₀={s0/M:.4e} M_Pl; skipping 2R node check.", color="yellow")
            continue

        n_nodes = node_count_to_2R(sol, R_star_m, idx_sigma=2)
        per_mode.setdefault(n_nodes, []).append(
            dict(
                s0=s0, sol=sol, mu2_log=mu2_log, R_star_m=R_star_m, smax=smax_by_s0[s0]
            )
        )

    # --- 3) Select modes until we have as many solutions as sign-change brackets ---
    sigma0_by_mode = {}  # maps node count to list of chosen σ₀ values

    for n in sorted(per_mode.keys()):
        cands = per_mode.get(n, [])
        if not cands:
            custom_print(f"[MISS] No candidate with n={n} nodes.", color="red")
            continue

        if nu_val != 0.0:
            nu_abs = abs(nu_val)
            # handle both +ν and −ν vacua
            for sign in (+1, -1):
                group = []
                for d in cands:
                    # value of σ at r_max from the stored solution
                    val = float(d["sol"].y[2, -1])
                    # compare distances to +ν and −ν
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
                        f"[OK] n={n}, vacuum={vac_label}ν: σ₀ = {pick['s0']/M:.4e} M_Pl",
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

    n = 0
    while len(sigma0_by_mode) < target_solutions:
        cands = per_mode.get(n, [])
        if not cands:
            custom_print(f"[MISS] No candidate with n={n} nodes.", color="red")
        else:
            # choose the smallest |σ0| among candidates with this node count
            pick = min(cands, key=lambda d: abs(d["smax"]))
            if n not in sigma0_by_mode:
                sigma0_by_mode[n] = pick["s0"]
                custom_print(f"[OK] n={n}: σ₀ = {pick['s0']/M:.4e} M_Pl ", color="cyan")
        n += 1

    if len(sigma0_by_mode) < target_solutions:
        custom_print(
            f"[WARN] Only {len(sigma0_by_mode)} solution(s) selected "
            f"out of target {target_solutions}. Consider widening the bracket or relaxing thresholds.",
            color="yellow",
        )

    # --- 4) Final integrate (again) for selected modes for plotting + diagnostics ---
    plot_entries = []
    for n, s0 in selected_pairs:
        sol, mu2_log, R_star_m = integrate_star(
            s0,
            p_eqState,
            rho_eqState,
            r0,
            r_max,
            xi_val,
            lmbda_val,
            nu_val,
            rho0,
            frac_pc,
            stop_at_2r=True,
            record_mu2=True,
        )

        r_star_km = (R_star_m/1e3) if (R_star_m is not None) else np.nan

        # Background leg for ADM mass & scalar charge
        sol_bnd, _, _ = integrate_star(
            s0,
            p_eqState,
            rho_eqState,
            r0,
            r_max,
            xi_val,
            lmbda_val,
            nu_val,
            rho0,
            frac_pc,
            stop_at_2r=False,
            record_mu2=False,
        )

        ADM_mass = adm_mass_from_sol(sol_bnd, k_tail=15)
        scalar_charge = scal_charge_from_sol(sol_bnd, lmbda_val, nu_val, r_max)

        custom_print(f"\nResults for n={n} mode", style="bold")
        print("ADM mass / M_sun = ", f"{ADM_mass/SM:.2e}")
        print("Scalar charge / M_sun = ", f"{scalar_charge/SM:.2e}")
        print("Q/M ratio = ", f"{scalar_charge/ADM_mass:.2e}")

        with open(
            f"Results_lmbda={lmbda_val:.0e}/Results_{star_weight}_xi={xi_val:.0f}/data.csv",
            "a",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    n,
                    ADM_mass / SM,
                    scalar_charge / SM,
                    scalar_charge / ADM_mass,
                    r_star_km,
                ]
            )

        r_mu, mu2 = (np.array(mu2_log, dtype=float).T if len(mu2_log) else (np.array([]), np.array([])))
        plot_entries.append({
            "label": f"n={n}",
            "sol": sol,
            "r_star": r_star_km,
            "r_mu": r_mu,
            "mu2": mu2
        })

    t_tot = time.perf_counter() - t0
    custom_print(f"\nExecution time: {t_tot:.0f} s", style="dim")

    # --- 5) Plots
    printResults_multi(plot_entries, xi_val, lmbda_val, nu_val, star_weight)
