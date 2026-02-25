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
from typing import Dict, Any, Iterable, Tuple

import numpy as np

from Utils.params import (
    M,
    SM,
    rho0_lightS,
    rho0_heavyS,
)
from Utils.EOS import p_SLy4, rho_SLy4
from Utils.graphics_single import custom_print
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
    p["target_shooting"] = [0.0] if nu_val == 0.0 else [abs(nu_val), -abs(nu_val)]
    # Determine method heuristically: replicate sz_single logic
    p["method"] = (
        "BDF"
        if (xi_val <= 0.0 or lmbda_val > 1e-65 or nu_val > M * 0.1 or m_val >= 1e-11)
        else "RK45"
    )
    result = solve_model(p, rho0)
    return xi_val, lmbda_val, result


def _write_scan_row(
    writer: csv.writer,
    xi_val: float,
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


def run_scan() -> None:
    """Perform ξ and λ scans for both light and heavy stars."""
    # ----- define the scan ranges -----
    # ξ scan: set the range and step here
    # xi_min, xi_max, xi_step = -10.0, 10.0, 0.25
    xi_min, xi_max, xi_step = -10.0, 5, 2.5
    # λ scan: list of values to sample; choose logarithmic spacing
    lmbda_scan = [
        0.0,
        # 1e-76,
        1e-74,
        # 1e-72,
        # 1e-70,
        # 1e-68,
        # 1e-66,
        # 1e-64,
        # 1e-62,
        # 1e-60,
        # 1e-58,
        # 1e-56,
        # 1e-54,
        1e-52,
        1e-50,
    ]
    # fixed values for the other parameter during each scan
    xi_fixed_for_lambda_scan = 100
    lmbda_fixed_for_xi_scan = 0.0
    # physical constants
    nu_val = 0.0
    m_val = 0.0

    # base parameter template (common to all scan points)
    params_template: Dict[str, Any] = {
        "p_eqState": p_SLy4,
        "rho_eqState": rho_SLy4,
        "frac_pc": 1e-10,
        # The following will be overridden per point
        "xi": xi_fixed_for_lambda_scan,
        "m": m_val,
        "lmbda": lmbda_fixed_for_xi_scan,
        "nu": nu_val,
        "method": "RK45",
        "a": 1e-10 * M,
        "b": M,
        "abs_cut": (1e-10 * M) * 1e-2,
        "rel_cut": 1e-2,
        "merge_tol": (1e-10 * M) * 1e-2,
        "target_shooting": [0.0],
    }

    # List of stars to scan
    stars = [
        (rho0_lightS, "L"),
        (rho0_heavyS, "H"),
    ]

    # ---- ξ scan at fixed λ ----
    xi_values = list(np.arange(xi_min, xi_max + xi_step * 0.5, xi_step))
    lmbda_val = lmbda_fixed_for_xi_scan
    for rho0, tag in stars:
        custom_print(
            f"\nPerforming ξ–scan for star {tag}: λ={lmbda_val:.2e}, ν={nu_val:.2e}",
            style="bold",
        )
        out_dir = os.path.join("Results", "scan", tag)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(
            out_dir, f"xi_scan_lmbda={lmbda_val:.0e}_nu={nu_val:.0e}.csv"
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "xi",
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
                        _write_scan_row(writer, xi_ret, l_ret, nu_val, tag, result)
                    except Exception as e:
                        # Log error and write a non–scalarised row
                        if tqdm is not None:
                            tqdm.write(f"[ξ={xi_hint}] error: {e}")
                        else:
                            custom_print(f"[ξ={xi_hint}] error: {e}", color="red")
                        _write_scan_row(writer, xi_hint, l_hint, nu_val, tag, None)

        custom_print(
            f"Completed ξ–scan for star {tag}. CSV saved to {csv_path}",
            style="dim",
        )

    # ---- λ scan at fixed ξ ----
    xi_val = xi_fixed_for_lambda_scan
    for rho0, tag in stars:
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
                        _write_scan_row(writer, xi_ret, l_ret, nu_val, tag, result)
                    except Exception as e:
                        if tqdm is not None:
                            tqdm.write(f"[λ={l_hint}] error: {e}")
                        else:
                            custom_print(f"[λ={l_hint}] error: {e}", color="red")
                        _write_scan_row(writer, xi_hint, l_hint, nu_val, tag, None)
        custom_print(
            f"Completed λ–scan for star {tag}. CSV saved to {csv_path}",
            style="dim",
        )


if __name__ == "__main__":
    run_scan()
