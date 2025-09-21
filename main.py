import numpy as np

from Utils.params import c, G_N, M, SM, lmbda_EMG, rho0_lightS, rho0_heavyS
from Utils.shooting import shoot_sigma0 
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.graphics import printResults_multi, custom_print
from Utils.diagnostic import (
    diagnostic_scan,
    print_candidate_info,
    probe_brackets_with_brent,
)
from Utils.solver import integrate_star, node_count_to_2R, adm_mass_from_solution

import time

# ---------- USER DEFINED PARAMETERS ----------
p_eqState = p_SLy4  # Choose the EOS
rho_eqState = rho_SLy4  # Choose the EOS

# Choose between light or heavy star
rho0 = rho0_heavyS

xi_val = -15
lmbda_val = lmbda_EMG
frac_pc = 1e-10


# Shooting method parameters
# Use a wider bracket and a tighter absolute cutoff
a, b = 1e-8 * M, 0.5 * M  #  Bracket for σ0
# Increase the number of seeds to sample the bracket finely
n_seeds = 25
modes = (0, 1, 2)  # Modes to search for (0,1,2,...)
# ---------------------------------------------


# ---------- main ----------
if __name__ == "__main__":
    t0 = time.perf_counter()

    star_weight = ""
    if rho0 == rho0_lightS:
        star_weight = "L"  # Light star
    elif rho0 == rho0_heavyS:
        star_weight = "H"  # Heavy star

    custom_print(
        f"\nxi = {xi_val} | λ = {lmbda_val:.2e} | ρ0 = {rho0:.1e}\n", style="bold"
    )

    r0     = 1e-2         # m
    # Increase r_max so that σ(r_max) is smaller on the n=1 branch
    r_max = 5e5  # m
    r_eval = np.linspace(r0, r_max, int(1e6))  # (not strictly needed here)

    # --- pre-scan (coarse) for visibility
    S, F, brackets, dips = diagnostic_scan(
        r0, r_max, p_eqState, rho_eqState, xi_val, rho0, frac_pc, a, b, n=201
    )
    custom_print(
        f"[diagnostics] scan over [{a/M:.1e},{b/M:.1e}]*M: "
        f"{np.sum(np.isfinite(F))}/{len(F)} finite samples, sign-change brackets={len(brackets)}, dips={len(dips)}",
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
        lmbda=lmbda_val,
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
            abs_threshold=abs_cut,
            tol_relative=rel_cut,
            merge_tol=merge_tol,
            idx_sigma=2,
        )

    if not s0_list:
        # as a last hint to the user: report best dips
        custom_print(
            "[fallback] still empty; reporting |σ(r_max)| minima as hints:",
            color="yellow",
        )
        for s, f in sorted(dips, key=lambda t: abs(t[1]))[:6]:
            custom_print(
                f"   near σ0/M≈{s/M:.3e}: |σ(r_max)|/M≈{abs(f)/M:.3e}",
                color="yellow",
            )
        raise RuntimeError("No σ0 roots found; see diagnostics above.")

    # --- print diagnostics for accepted candidates
    print_candidate_info(
        s0_list, rho0, r0, r_max, xi_val, lmbda_val, p_eqState, rho_eqState, frac_pc
    )

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
            rho0,
            frac_pc,
            stop_at_2r=True,
            record_mu2=True,
        )

        if R_star_m is None:
            custom_print(f"[WARN] R_* not found for σ₀={s0/M:.4e} M_Pl; skipping 2R node check.", color="yellow")
            continue

        n_nodes = node_count_to_2R(sol, R_star_m, idx_sigma=2)
        per_mode.setdefault(n_nodes, []).append(dict(
            s0=s0, sol=sol, mu2_log=mu2_log, R_star_m=R_star_m
        ))

    # --- 3) Select desired modes (e.g., n=0,1): choose smallest |σ0| if multiple ---
    sigma0_by_mode = {}
    for n in modes:
        cands = per_mode.get(n, [])
        if not cands:
            custom_print(f"[MISS] No candidate with n={n} nodes.", color="red")
            continue
        pick = min(cands, key=lambda d: abs(d["s0"]))
        sigma0_by_mode[n] = pick["s0"]
        custom_print(f"[OK] n={n}: σ₀ = {pick['s0']/M:.4e} M_Pl", color="cyan")

    # --- 4) Final integrate (again) for selected modes for plotting + diagnostics ---
    plot_entries = []
    for n, s0 in list(sigma0_by_mode.items()):
        sol, mu2_log, R_star_m = integrate_star(
            s0,
            p_eqState,
            rho_eqState,
            r0,
            r_max,
            xi_val,
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
            rho0,
            frac_pc,
            stop_at_2r=False,
            record_mu2=False,
        )

        ADM_mass = adm_mass_from_solution(sol_bnd, k_tail=15)
        R_log = R_star_m  # FIXME:
        if lmbda_val != 0.0:
            scalar_charge = -r_max**2 * np.sqrt(np.log(r_max/max(R_log, 1.0))) * sol_bnd.y[3][-1]/M * (c*c) / G_N
        else:
            scalar_charge = -r_max**2 * sol_bnd.y[3][-1]/M * (c*c) / G_N
        custom_print(f"\nResults for n={n} mode", style="bold")
        print("ADM mass / M_sun = ", f"{ADM_mass/SM:.2e}")
        print("Scalar charge / M_sun = ", f"{scalar_charge/SM:.2e}")
        print("Q/M ratio = ", f"{scalar_charge/ADM_mass:.2e}")

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
    printResults_multi(plot_entries, xi_val, star_weight)
