"""
Core solver routines for neutron–star scalarisation.

This module factors out the heavy lifting previously performed in
``sz_single.py`` into standalone, importable functions.  These
functions accept an explicit parameter dictionary rather than relying
on module‑level globals, making them easier to reuse from other
scripts such as scans or notebooks.  The high–level logic mirrors
the old ``solve_model`` implementation: perform an adaptive scan to
identify promising shooting brackets, refine them to find σ₀ roots,
integrate each candidate solution, compute physical quantities and
prepare the data needed for plotting.  In addition it records the
results in a CSV file and returns both the plotting entries and the
raw rows for further processing.
"""

from __future__ import annotations

import os
import csv
import io
import contextlib
import traceback
from typing import Dict, Any, Tuple, List

import numpy as np

from Utils.params import M, SM, rho0_lightS, rho0_heavyS, lmbda_to_SI, m_ev_to_SI
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.graphics_single import custom_print
from Utils.shooting import diagnostic_scan, probe_brackets
from Utils.analysis import (
    integrate_star,
    node_count_to_2R,
    adm_mass_from_sol,
    scal_charge_from_sol,
)


def solve_model(params: Dict[str, Any], rho0: float) -> Dict[str, Any]:
    """Solve the stellar structure and scalar field for a given central density.

    Parameters
    ----------
    params : dict
        Dictionary of input parameters.  Expected keys are:

        ``p_eqState`` : equation of state p(ρ) callable

        ``rho_eqState`` : density as a function of pressure callable

        ``frac_pc`` : fractional pressure cutoff for the integrator

        ``xi`` : non‑minimal coupling constant ξ

        ``m`` : scalar mass in eV

        ``lmbda`` : self–interaction coupling λ in natural units

        ``nu`` : vacuum expectation value in Planck units

        ``method`` : integrator method ("BDF" or "RK45")

        ``a``, ``b`` : bracketing interval for the shooting parameter σ₀ (in SI metres)

        ``target_shooting`` : list of target values for σ(r_max) used by
            ``probe_brackets`` to locate roots (e.g. [|ν|, −|ν|] or [0.0]).

        ``abs_cut``, ``rel_cut``, ``merge_tol`` : numerical tolerances for
            root finding.

    rho0 : float
        Central density of the neutron star (in kg/m³).  Use
        ``rho0_lightS`` or ``rho0_heavyS`` from :mod:`Utils.params` for the
        canonical “light” and “heavy” stars.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``plot_entries`` : list of dicts, one per plotted solution, ready for
            consumption by :func:`Utils.graphics_single.plotResults_multi`.

        ``vacuum_sols`` : dict mapping vacuum sign labels ('+', '-', '0') to
            the number of solutions found in that vacuum.

        ``nu_val`` : float, the value of ν used for this run.

        ``path`` : str, directory where results (plots and CSV) have been
            written.  Created if necessary.

        ``csv_path`` : str, absolute path to the CSV file created.

        ``data_rows`` : list of lists, each inner list corresponding to a
            row written to the CSV file.  The first row is the header.
    """
    # Unpack required parameters; provide sensible defaults where possible
    p_eqState = params.get("p_eqState", p_SLy4)
    rho_eqState = params.get("rho_eqState", rho_SLy4)
    frac_pc = params.get("frac_pc", 1e-10)
    xi_val = float(params.get("xi", 0.0))
    m_val = float(params.get("m", 0.0))
    lmbda_val = float(params.get("lmbda", 0.0))
    nu_val = float(params.get("nu", 0.0))
    method = params.get("method")
    a = float(params.get("a", 1e-10 * M))
    b = float(params.get("b", M))
    target_shooting = params.get("target_shooting", [0.0])
    abs_cut = float(params.get("abs_cut", a * 1e-2))
    rel_cut = float(params.get("rel_cut", 1e-2))
    merge_tol = float(params.get("merge_tol", a * 1e-2))

    # Determine star label
    star_weight = ""
    sw = ""
    if rho0 == rho0_lightS:
        star_weight = "L"
        sw = "Light"
    elif rho0 == rho0_heavyS:
        star_weight = "H"
        sw = "Heavy"
    else:
        # Custom densities get an explicit tag
        star_weight = f"rho0={rho0:g}"
        sw = star_weight

    # Print run header
    custom_print(
        f"\nξ = {xi_val} | µ = {m_val:.2e} | λ = {lmbda_val:.2e} | ν = {nu_val:.2e} | ν/M = {nu_val/M:.2e} | {sw} star",
        style="bold",
    )

    # Convert mass and lambda to SI units
    m2_val = 0.0
    if m_val != 0.0:
        m_val_SI = m_ev_to_SI(m_val)
        m2_val = m_val_SI**2
    lmbda_val_SI = lmbda_to_SI(lmbda_val)

    # Integration radial domain
    r0 = 1e-2  # m
    r_max = 3e5  # m

    # ---------- 1) adaptive diagnostic scan ----------
    scan = diagnostic_scan(
        r0=r0,
        r_max=r_max,
        p_eqState=p_eqState,
        rho_eqState=rho_eqState,
        xi=xi_val,
        m2=m2_val,
        lmbda=lmbda_val_SI,
        nu_val=nu_val,
        rho0=rho0,
        frac_pc=frac_pc,
        method=method,
        a=a,
        b=b,
        n_coarse=41,
        n_refine=81,
        target=target_shooting,
        expand_coarse_points=1,
        detect_near_zero=True,
        compress_brackets=True,
    )
    brackets = scan["brackets"]
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
        return {
            "plot_entries": [],
            "vacuum_sols": {"+": 0, "-": 0, "0": 0},
            "nu_val": nu_val,
            "path": "",  # no directory created
            "csv_path": "",
            "data_rows": [],
        }

    # ---------- 2) refine brackets to find σ₀ roots ----------
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
        lmbda_val=lmbda_val_SI,
        nu_val=nu_val,
        abs_threshold=abs_cut,
        tol_relative=rel_cut,
        merge_tol=merge_tol,
        target=target_shooting,
        idx_sigma=2,
    )
    if not s0_list:
        raise RuntimeError("No σ₀ roots found; see diagnostics above.")

    smax_by_s0 = {s: sm for s, sm in zip(s0_list, s_maxes)}

    # ---------- 3) integrate each σ₀ candidate to 2R and classify modes ----------
    per_mode: Dict[int, List[Dict[str, Any]]] = {}
    nu_abs = abs(nu_val)
    for s0 in s0_list:
        sol, mu2_log, R_star_m = integrate_star(
            s0,
            p_eqState,
            rho_eqState,
            r0,
            r_max,
            xi_val,
            m2_val,
            lmbda_val_SI,
            nu_val,
            rho0,
            frac_pc,
            method=method,
            stop_at_2r=True,
            record_mu2=True,
        )
        if R_star_m is None:
            custom_print(
                f"[WARN] R_* not found for σ₀={s0/M:.4e} M_Pl; skipping 2R node check.",
                color="yellow",
            )
            continue
        # choose the target sign for node counting based on the vacuum reached
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

    # ---------- 4) select one candidate per node count and vacuum sign ----------
    sigma0_by_mode: Dict[int, List[float]] = {}
    for n in sorted(per_mode.keys()):
        cands = per_mode.get(n, [])
        if not cands:
            custom_print(f"[MISS] No candidate with n={n} nodes.", color="red")
            continue
        if nu_val != 0.0:
            for sign in (+1, -1):
                group: List[Dict[str, Any]] = []
                for d in cands:
                    val = float(d["smax"])
                    diff_plus = abs(val - nu_abs)
                    diff_minus = abs(val + nu_abs)
                    sign_val = +1 if (diff_plus <= diff_minus) else -1
                    if sign_val == sign:
                        diff_to_vac = diff_plus if sign == +1 else diff_minus
                        d_candidate = d.copy()
                        d_candidate["vacuum_sign"] = sign
                        d_candidate["sigma_rmax"] = val
                        d_candidate["diff_to_vac"] = diff_to_vac
                        group.append(d_candidate)
                if group:
                    pick = min(group, key=lambda d: d["diff_to_vac"])
                    sigma0_by_mode.setdefault(n, []).append(pick["s0"])
                    vac_label = "+" if sign == +1 else "-"
                    custom_print(
                        f"n={n}, vacuum={vac_label}ν: σ₀ = {pick['s0']/M:.4e} M_Pl",
                        color="cyan",
                    )
        else:
            # ν=0: single vacuum at zero; choose the candidate with the smallest residual at r_max
            pick = min(cands, key=lambda d: abs(d["smax"]))
            sigma0_by_mode.setdefault(n, []).append(pick["s0"])
            custom_print(f"[OK] n={n}: σ₀ = {pick['s0']/M:.4e} M_Pl", color="cyan")

    selected_pairs = [(n, s0) for n, s_list in sigma0_by_mode.items() for s0 in s_list]
    if not selected_pairs:
        custom_print(
            "[WARN] No σ₀ solutions selected. Consider widening the bracket or relaxing thresholds.",
            color="yellow",
        )

    # ---------- 5) final integration for selected modes and write results ----------
    plot_entries: List[Dict[str, Any]] = []
    vacuum_sols = {"+": 0, "-": 0, "0": 0}
    # build output directory structure similar to the old script
    base_dir = os.path.join("Results", "single", star_weight)
    if m_val == 0.0:
        subdir = f"lmbda={lmbda_val:.0e}_nu={nu_val:.0e}_xi={xi_val:.2f}"
    else:
        subdir = f"lmbda={lmbda_val:.0e}_m={m_val:.0e}_xi={xi_val:.2f}"
    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, "data.csv")
    # prepare rows list; include header row
    rows: List[List[Any]] = [
        [
            "mode_n",
            "vacuum_sign",
            "sigma0_over_M",
            "ADM_over_Msun",
            "Q_over_Msun",
            "Q_over_ADM",
            "R_star_km",
        ]
    ]
    # open CSV for writing
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rows[0])
        for n, s0 in selected_pairs:
            # integrate twice: once to 2R for r_star and mu2, once fully for ADM and Q
            sol, mu2_log, R_star_m = integrate_star(
                s0,
                p_eqState,
                rho_eqState,
                r0,
                r_max,
                xi_val,
                m2_val,
                lmbda_val_SI,
                nu_val,
                rho0,
                frac_pc,
                method=method,
                stop_at_2r=True,
                record_mu2=True,
            )
            r_star_km = (R_star_m / 1e3) if (R_star_m is not None) else float("nan")
            sol_bnd, _, _ = integrate_star(
                s0,
                p_eqState,
                rho_eqState,
                r0,
                r_max,
                xi_val,
                m2_val,
                lmbda_val_SI,
                nu_val,
                rho0,
                frac_pc,
                method=method,
                stop_at_2r=False,
                record_mu2=False,
            )
            ADM_mass = adm_mass_from_sol(sol_bnd, k_tail=15)
            scalar_charge = scal_charge_from_sol(
                sol_bnd, m2_val, lmbda_val_SI, nu_val, r_max
            )
            # determine vacuum sign of this solution
            if nu_val != 0.0:
                val = float(sol_bnd.y[2, -1])
                diff_plus = abs(val - nu_abs)
                diff_minus = abs(val + nu_abs)
                vac_sign = +1 if (diff_plus <= diff_minus) else -1
            else:
                vac_sign = 0
            vac_label = "0" if vac_sign == 0 else ("+" if vac_sign == +1 else "-")
            vacuum_sols[vac_label] += 1
            # print small summary to captured log
            custom_print(f"\nResults for n={n} mode", style="bold")
            print("ADM mass / M_sun = ", f"{ADM_mass/SM:.2e}")
            print("Scalar charge / M_sun = ", f"{scalar_charge/SM:.2e}")
            print(
                "Q/M ratio = ",
                f"{(scalar_charge/ADM_mass) if ADM_mass != 0 else float('nan'):.2e}",
            )
            # build plotting entry
            r_mu, mu2 = (
                np.array(mu2_log, dtype=float).T
                if mu2_log
                else (np.array([]), np.array([]))
            )
            plot_entries.append(
                {
                    "label": f"n={n}",
                    "sol": sol,
                    "r_star": r_star_km,
                    "r_mu": r_mu,
                    "mu2": mu2,
                    "vacuum_sign": vac_sign,
                }
            )
            # write CSV row
            row = [
                n,
                vac_sign,
                f"{s0/M:.3e}",
                f"{ADM_mass/SM:.3e}",
                f"{scalar_charge/SM:.3e}",
                f"{(scalar_charge/ADM_mass) if ADM_mass != 0 else float('nan'):.3e}",
                f"{r_star_km:.3e}",
            ]
            writer.writerow(row)
            rows.append(row)

    return {
        "plot_entries": plot_entries,
        "vacuum_sols": vacuum_sols,
        "nu_val": nu_val,
        "path": path,
        "csv_path": csv_path,
        "data_rows": rows,
    }


def run_solve_model_captured(
    params: Dict[str, Any], rho0: float
) -> Tuple[str, str, bool, Dict[str, Any]]:
    """Run :func:`solve_model` capturing its printed output.

    Returns a tuple ``(label, log_text, ok, result)``.  The
    ``label`` is "Light" or "Heavy" depending on the central density,
    ``log_text`` is all text printed during the solve, ``ok``
    indicates whether the solve completed without exceptions, and
    ``result`` is the dictionary returned by :func:`solve_model` (or
    ``None`` if an exception occurred).
    """
    label = (
        "Light"
        if rho0 == rho0_lightS
        else ("Heavy" if rho0 == rho0_heavyS else str(rho0))
    )
    buf = io.StringIO()
    ok = True
    result: Dict[str, Any] | None = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            result = solve_model(params, rho0)
        except Exception:
            ok = False
            traceback.print_exc()
    return label, buf.getvalue(), ok, result
