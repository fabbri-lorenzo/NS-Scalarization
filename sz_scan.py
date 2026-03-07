"""
Parallelised scan over coupling parameters for scalarised neutron stars.

This script performs two complementary scans:

* **ξ–scan at fixed λ** – varying ξ over a user‑specified range while
  keeping λ and ν fixed.  For each central density (light and heavy
  stars) the solver is invoked in parallel and the resulting
  Q/ℳ ratios are recorded in a CSV file.

* **λ–scan at fixed ξ** – varying λ over a user‑specified range while
  keeping ξ and ν fixed.  Results are stored analogously.

The output CSVs can be consumed by the plotting routines in
``Q_M.py`` to produce Q/M versus ξ or λ graphs.  Each row of the CSV
contains the scanned parameter value, the star tag (``L`` or ``H``),
a flag indicating whether scalarisation occurred (``scalarized``),
the mode number, vacuum sign, σ₀/M, ADM mass and scalar charge in
solar masses, their ratio, and the stellar radius.  If no
scalarised solution is found at a given scan point, a row with
``scalarized=0`` and empty fields is written.
"""

from __future__ import annotations

import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Iterable, Tuple, List

from glob import glob
import numpy as np

from Utils.params import (
    M,
    SM,
    rho0_lightS,
    rho0_heavyS,
)
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.graphics_single import custom_print
from Utils.graphics_scan import plot_scan_results
from Utils.core import solve_model

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _run_single_point(
    params_template: Dict[str, Any],
    rho0: float,
    xi_val: float,
    lmbda_val: float,
    nu_val: float,
    m_val: float,
) -> Tuple[float, float, Dict[str, Any]]:
    """Worker helper to evaluate one (ξ, λ) point.

    It constructs a fresh parameter dict from ``params_template`` and
    overrides the values of ξ, λ, ν, m and the integrator method.
    Returns the (ξ, λ, result) triple.  Any exception inside
    :func:`solve_model` will propagate to the caller.
    """
    # Copy base params to avoid cross‑talk between tasks
    p = dict(params_template)
    p["xi"] = xi_val
    p["lmbda"] = lmbda_val
    p["nu"] = nu_val
    p["m"] = m_val
    # Update shooting targets and method accordingly
    p["target_shooting"] = [0.0]
    if (lmbda_val != 0.0) and (nu_val**2 > m_val**2 / lmbda_val):
        sigma_min = np.sqrt(nu_val**2 - m_val**2 / lmbda_val)
        p["target_shooting"] = [abs(sigma_min), -abs(sigma_min)]

    # Determine method heuristically: replicate sz_single logic
    p["method"] = "BDF"
    result = solve_model(p, rho0)
    return xi_val, lmbda_val, result


def _write_scan_row(
    writer: csv.writer,
    xi_val: float,
    mu_val: float,
    lmbda_val: float,
    nu_val: float,
    star_tag: str,
    result: Dict[str, Any] | None,
):
    """Write one scan row to CSV given the solve result.

    If ``result`` is ``None`` or contains no data rows, write a row
    indicating that scalarisation did not occur.
    """
    if not result or not result.get("data_rows") or len(result["data_rows"]) <= 1:
        # no solutions found
        writer.writerow(
            [
                xi_val,
                f"{mu_val:.1e}",
                f"{lmbda_val:.1e}",
                f"{nu_val:.1e}",
                star_tag,
                0,  # scalarized flag
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        return
    # At least one solution: iterate over all modes
    for row in result["data_rows"][1:]:
        (
            mode_n,
            vac_sign,
            sigma0_over_M,
            ADM_over_Msun,
            Q_over_Msun,
            Q_over_ADM,
            R_star_km,
        ) = row
        writer.writerow(
            [
                xi_val,
                f"{mu_val:.1e}",
                f"{lmbda_val:.1e}",
                f"{nu_val:.1e}",
                star_tag,
                1,
                mode_n,
                vac_sign,
                sigma0_over_M,
                ADM_over_Msun,
                Q_over_Msun,
                Q_over_ADM,
                R_star_km,
            ]
        )


def _resolve_stars(selection: str | Iterable[str]) -> List[Tuple[float, str]]:
    """Resolve star selection into (rho0, tag) pairs."""
    if isinstance(selection, str):
        key = selection.strip().lower()
        if key in {
            "both",
            "all",
            "lh",
            "hl",
            "l+h",
            "h+l",
            "light+heavy",
            "heavy+light",
        }:
            tags = {"L", "H"}
        else:
            tags = set()
            if key in {"l", "light"}:
                tags.add("L")
            elif key in {"h", "heavy"}:
                tags.add("H")
            else:
                raise ValueError(
                    "stars must be 'light', 'heavy', 'both' or an iterable of those"
                )
    else:
        tags = set()
        for item in selection:
            if not isinstance(item, str):
                raise TypeError("stars iterable must contain strings")
            key = item.strip().lower()
            if key in {"l", "light"}:
                tags.add("L")
            elif key in {"h", "heavy"}:
                tags.add("H")
            elif key in {"both", "all"}:
                tags.update({"L", "H"})
            else:
                raise ValueError(
                    "stars iterable must contain 'light', 'heavy', or 'both'"
                )
    stars: List[Tuple[float, str]] = []
    if "L" in tags:
        stars.append((rho0_lightS, "L"))
    if "H" in tags:
        stars.append((rho0_heavyS, "H"))
    if not stars:
        raise ValueError("stars selection produced no targets")
    return stars


def _resolve_scans(selection: str | Iterable[str]) -> Tuple[bool, bool, bool]:
    """Resolve scan selection into (do_xi, do_lambda, do_mu)."""

    def _norm_key(item: str) -> str:
        return item.strip().lower().replace(" ", "")

    # accepted aliases per scan
    XI_KEYS = {"xi"}
    LAMBDA_KEYS = {"lambda", "lmbda"}
    MU_KEYS = {"mu", "mass", "baremass", "scalarmass"}

    # accepted "all/both" combos
    ALL_KEYS = {
        "all",
        "everything",
        "xi+lambda+mu",
        "xi+mu+lambda",
        "lambda+xi+mu",
        "lambda+mu+xi",
        "mu+xi+lambda",
        "mu+lambda+xi",
    }

    # 2-way combos (kept for backward compatibility)
    XI_LAMBDA_KEYS = {
        "both",
        "xi+lambda",
        "lambda+xi",
        "xi,lambda",
        "lambda,xi",
        "xi,lmbda",
        "lmbda,xi",
    }
    XI_MU_KEYS = {"xi+mu", "mu+xi", "xi,mu", "mu,xi"}
    LAMBDA_MU_KEYS = {
        "lambda+mu",
        "mu+lambda",
        "lambda,mu",
        "mu,lambda",
        "lmbda+mu",
        "mu+lmbda",
        "lmbda,mu",
        "mu,lmbda",
    }

    if isinstance(selection, str):
        key = _norm_key(selection)

        # full set
        if key in ALL_KEYS:
            return True, True, True

        # two-way combos
        if key in XI_LAMBDA_KEYS:
            return True, True, False
        if key in XI_MU_KEYS:
            return True, False, True
        if key in LAMBDA_MU_KEYS:
            return False, True, True

        # single
        if key in XI_KEYS:
            return True, False, False
        if key in LAMBDA_KEYS:
            return False, True, False
        if key in MU_KEYS:
            return False, False, True

        raise ValueError(
            "scans must be 'xi', 'lambda', 'mu', 'all' (or combos like 'xi+lambda', 'xi+mu', 'lambda+mu') "
            "or an iterable of those"
        )

    do_xi = False
    do_lambda = False
    do_mu = False

    for item in selection:
        if not isinstance(item, str):
            raise TypeError("scans iterable must contain strings")
        key = _norm_key(item)

        if key in XI_KEYS:
            do_xi = True
        elif key in LAMBDA_KEYS:
            do_lambda = True
        elif key in MU_KEYS:
            do_mu = True
        elif key in {"both"}:
            # keep old meaning: xi + lambda
            do_xi = True
            do_lambda = True
        elif key in {"all", "everything"}:
            do_xi = True
            do_lambda = True
            do_mu = True
        else:
            raise ValueError(
                "scans iterable must contain 'xi', 'lambda', 'mu', 'both', or 'all'"
            )

    if not (do_xi or do_lambda or do_mu):
        raise ValueError("scans selection produced no targets")

    return do_xi, do_lambda, do_mu


def run_scan(
    stars: str | Iterable[str] = "heavy",
    scans: str | Iterable[str] = "both",
) -> None:
    """Perform ξ and/or λ scans for selected stars.

    Args:
        stars: "light", "heavy", "both" or an iterable of those.
        scans: "xi", "lambda", "mu", "both" or an iterable of those.
    """
    # ----- define the scan ranges -----
    # ξ scan: set the range and step here
    xi_min, xi_max, xi_step = -10.0, 10.0, 0.25
    # λ scan: list of values to sample; choose logarithmic spacing
    lmbda_scan = [
        0.0,
        1e-80,
        1e-78,
        1e-76,
        1e-74,
        1e-72,
        1e-70,
        1e-68,
        1e-66,
        1e-64,
        1e-62,
        1e-60,
        1e-58,
        1e-56,
        1e-54,
        1e-52,
        1e-50,
    ]
    mu_scan = [
        0.0,
        1e-20,
        1e-19,
        1e-18,
        1e-17,
        1e-16,
        1e-15,
        1e-14,
        1e-13,
        1e-12,
        1e-11,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
    ]
    # fixed values for the other parameter during each scan
    xi_fixed = 10
    lmbda_fixed_for_xi_scan = 0.0
    mu_fixed_for_xi_scan = 1e-10
    # physical constants
    nu_val = 0.0

    # base parameter template (common to all scan points)
    params_template: Dict[str, Any] = {
        "p_eqState": p_SLy4,
        "rho_eqState": rho_SLy4,
        "frac_pc": 1e-10,
        # The following will be overridden per point
        "xi": xi_fixed,
        "mu": mu_fixed_for_xi_scan,
        "lmbda": lmbda_fixed_for_xi_scan,
        "nu": nu_val,
        "method": "RK45",
        "a": 1e-10 * M,
        "b": M,
        "abs_cut": 1e-12 * M,
        "rel_cut": 1e-2,
        "merge_tol": 1e-12 * M,
        "target_shooting": [0.0],
    }

    # List of stars to scan
    stars_to_scan = _resolve_stars(stars)
    do_xi_scan, do_lambda_scan, do_mu_scan = _resolve_scans(scans)

    # ---- ξ scan at fixed λ, mu ----
    if do_xi_scan:
        xi_values = list(np.arange(xi_min, xi_max + xi_step * 0.5, xi_step))
        lmbda_val = lmbda_fixed_for_xi_scan
        m_val = mu_fixed_for_xi_scan
        for rho0, tag in stars_to_scan:
            custom_print(
                f"\nPerforming ξ–scan for star {tag}: µ={m_val:.2e}, λ={lmbda_val:.2e}, ν={nu_val:.2e}",
                style="bold",
            )
            out_dir = os.path.join("Results", "scan", tag)
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(
                out_dir,
                f"xi_scan_mu={m_val:.0e}_lmbda={lmbda_val:.0e}_nu={nu_val:.0e}.csv",
            )
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "xi",
                        "mu",
                        "lambda",
                        "nu",
                        "rho0_tag",
                        "scalarized",
                        "mode_n",
                        "vacuum_sign",
                        "sigma0_over_M",
                        "ADM_over_Msun",
                        "Q_over_Msun",
                        "Q_over_ADM",
                        "R_star_km",
                    ]
                )
                # Parallelise over ξ values
                with ProcessPoolExecutor(max_workers=4) as ex:
                    futures = {}
                    for xi_val in xi_values:
                        fut = ex.submit(
                            _run_single_point,
                            params_template,
                            rho0,
                            xi_val,
                            lmbda_val,
                            nu_val,
                            m_val,
                        )
                        futures[fut] = (xi_val, lmbda_val)
                    iterator = as_completed(futures)
                    if tqdm is not None:
                        iterator = tqdm(
                            iterator,
                            total=len(futures),
                            desc=f"ξ-scan {tag}",
                            unit="pt",
                        )
                    for fut in iterator:
                        xi_hint, l_hint = futures[fut]
                        try:
                            xi_ret, l_ret, result = fut.result()
                            _write_scan_row(
                                writer, xi_ret, m_val, l_ret, nu_val, tag, result
                            )
                        except Exception as e:
                            # Log error and write a non–scalarised row
                            if tqdm is not None:
                                tqdm.write(f"[ξ={xi_hint}] error: {e}")
                            else:
                                custom_print(f"[ξ={xi_hint}] error: {e}", color="red")
                            _write_scan_row(
                                writer, xi_hint, m_val, l_hint, nu_val, tag, None
                            )

            custom_print(
                f"Completed ξ–scan for star {tag}. CSV saved to {csv_path}",
                style="dim",
            )

    # ---- λ scan at fixed ξ ----
    if do_lambda_scan:
        xi_val = xi_fixed
        for rho0, tag in stars_to_scan:
            custom_print(
                f"\nPerforming λ–scan for star {tag}: ξ={xi_val:.2e}, ν={nu_val:.2e}",
                style="bold",
            )
            out_dir = os.path.join("Results", "scan", tag)
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(
                out_dir, f"lmbda_scan_xi={xi_val:.0e}_nu={nu_val:.0e}.csv"
            )
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "xi",
                        "mu",
                        "lambda",
                        "nu",
                        "rho0_tag",
                        "scalarized",
                        "mode_n",
                        "vacuum_sign",
                        "sigma0_over_M",
                        "ADM_over_Msun",
                        "Q_over_Msun",
                        "Q_over_ADM",
                        "R_star_km",
                    ]
                )
                with ProcessPoolExecutor(max_workers=4) as ex:
                    futures = {}
                    for lam_val in lmbda_scan:
                        fut = ex.submit(
                            _run_single_point,
                            params_template,
                            rho0,
                            xi_val,
                            lam_val,
                            nu_val,
                            m_val,
                        )
                        futures[fut] = (xi_val, lam_val)
                    iterator = as_completed(futures)
                    if tqdm is not None:
                        iterator = tqdm(
                            iterator,
                            total=len(futures),
                            desc=f"λ-scan {tag}",
                            unit="pt",
                        )
                    for fut in iterator:
                        xi_hint, l_hint = futures[fut]
                        try:
                            xi_ret, l_ret, result = fut.result()
                            _write_scan_row(
                                writer, xi_ret, m_val, l_ret, nu_val, tag, result
                            )
                        except Exception as e:
                            if tqdm is not None:
                                tqdm.write(f"[λ={l_hint}] error: {e}")
                            else:
                                custom_print(f"[λ={l_hint}] error: {e}", color="red")
                            _write_scan_row(
                                writer, xi_hint, m_val, l_hint, nu_val, tag, None
                            )
            custom_print(
                f"Completed λ–scan for star {tag}. CSV saved to {csv_path}",
                style="dim",
            )
    # ---- μ scan at fixed ξ and λ ----
    if do_mu_scan:
        xi_val = xi_fixed
        lam_val = (
            lmbda_fixed_for_xi_scan  # or define lmbda_fixed_for_mu_scan explicitly
        )
        for rho0, tag in stars_to_scan:
            custom_print(
                f"\nPerforming μ–scan for star {tag}: ξ={xi_val:.2e}, λ={lam_val:.2e}, ν={nu_val:.2e}",
                style="bold",
            )
            out_dir = os.path.join("Results", "scan", tag)
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(
                out_dir,
                f"mu_scan_xi={xi_val:.0e}_lmbda={lam_val:.0e}_nu={nu_val:.0e}.csv",
            )

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "xi",
                        "mu",
                        "lambda",
                        "nu",
                        "rho0_tag",
                        "scalarized",
                        "mode_n",
                        "vacuum_sign",
                        "sigma0_over_M",
                        "ADM_over_Msun",
                        "Q_over_Msun",
                        "Q_over_ADM",
                        "R_star_km",
                    ]
                )

                with ProcessPoolExecutor(max_workers=4) as ex:
                    futures = {}
                    for mu_val in mu_scan:
                        fut = ex.submit(
                            _run_single_point,
                            params_template,
                            rho0,
                            xi_val,
                            lam_val,
                            nu_val,
                            mu_val,
                        )
                        futures[fut] = mu_val

                    iterator = as_completed(futures)
                    if tqdm is not None:
                        iterator = tqdm(
                            iterator,
                            total=len(futures),
                            desc=f"μ-scan {tag}",
                            unit="pt",
                        )

                    for fut in iterator:
                        mu_hint = futures[fut]
                        try:
                            xi_ret, l_ret, result = fut.result()
                            _write_scan_row(
                                writer, xi_ret, mu_hint, l_ret, nu_val, tag, result
                            )
                        except Exception as e:
                            if tqdm is not None:
                                tqdm.write(f"[μ={mu_hint}] error: {e}")
                            else:
                                custom_print(f"[μ={mu_hint}] error: {e}", color="red")
                            _write_scan_row(
                                writer, xi_val, mu_hint, lam_val, nu_val, tag, None
                            )

            custom_print(
                f"Completed μ–scan for star {tag}. CSV saved to {csv_path}", style="dim"
            )


if __name__ == "__main__":
    # run_scan(stars="both", scans="xi")
    plot_scan_results(
        scan="lambda_scan",
        xi=100,
        mu=0.0,
        nu=0.0,
        base_dir="Results_final/EMG/scan",
        include_modes=(0, 1, 2),
        markers=False,
        dpi=600,
        out_path="Results/QM_vs_lambda_xi100.pdf",
    )
    # plot_scan_results(
    #    scan="xi_scan_mu",
    #    lmbda=0.0,
    #    mu=(1e-20, 1e-15),
    #    nu=0.0,
    #    base_dir="Results_final/ULA/scan",
    #    include_modes=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    #    markers=False,
    #    dpi=600,
    #    out_path="Results/QM_vs_xi.pdf",
    # )
