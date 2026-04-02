# Scalarization of Neutron Stars

Python code for studying spontaneous scalarization of neutron stars in Early Modified Gravity (EMG) and related scalar–tensor setups. It solves the TOV + scalar field system, scans coupling parameters, and produces publication‑ready plots.

This repository accompanies my MSc thesis in Theoretical Physics (University of Bologna).

## Contents
- [Project layout](#project-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Scripts](#scripts)
- [Outputs](#outputs)
- [Notes & references](#notes--references)

## Project layout
- `sz_single.py` — Solve for a single star (light/heavy) at one parameter point and plot σ, μ_eff², and pressure profiles.
- `sz_compare.py` — Sweep either λ or μ values and overlay the resulting profiles.
- `sz_scan.py` — Parallel scans over ξ, λ, or μ; writes CSVs and can generate Q/M vs parameter figures.
- `Utils/` — Numerical core (shooting, integration, analysis) and plotting utilities.
- `Tests/GR_SLY4.py` — GR baseline without scalar field (sanity check).
- `Results/`, `Results_final/` — Example outputs (plots and CSVs).
- `docs/` — Sphinx sources (API reference & workflow diagrams).

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # if present; otherwise install the libs below
```
Core dependencies (PyPI names):
`numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm`, `colorama`.

Tested with Python ≥ 3.11.

## Quick start
Run one configuration (light & heavy stars):
```bash
python sz_single.py
```

Compare several μ (or λ) values at fixed other couplings:
```bash
# choose 'compare_param' and values inside sz_compare.py
python sz_compare.py
```

Scan over ξ / λ / μ and write CSVs:
```bash
# configure ranges in sz_scan.py, then:
python sz_scan.py
```
Plots are saved under `Results/...` with filenames like `pressure.png`, `mu2.png`, `sigma.png`, or `*_compare_param.pdf`.

## Scripts
- **sz_single.py**: sets physical couplings and shooting brackets, dispatches to `Utils.core.run_solve_model_captured`, then calls `Utils.graphics_single.plotResults_multi`.
- **sz_compare.py**: builds parameter dictionaries via `make_params`, runs each point in parallel, and overlays curves with `Utils.graphics_compare.plotResults_compare`.
- **sz_scan.py**: uses `ProcessPoolExecutor` to scan grids, writes CSVs per star, and offers `plot_scan_results` to regenerate figures from saved data.

Adjust couplings (ξ, μ, λ, ν), integrator (`BDF`/`RK45`), and shooting bounds (`a`, `b`, tolerances) at the top of each script.

## Outputs
- **Plots**: PNG/PDF profiles and comparison figures under `Results/` or `Results_final/`.
- **CSVs**: Scan results contain ξ, μ, λ, ν, star tag (L/H), scalarized flag, mode number, vacuum sign, σ₀/M, ADM mass, scalar charge, Q/M, and radius.

## Notes & references
- EOS: SLy4 from Haensel & Potekhin (2004).
- EMG model: Braglia *et al.* (2020) [arXiv:2011.12934].
- Scalarization overview: Doneva *et al.* (2022) [arXiv:2211.01766].


