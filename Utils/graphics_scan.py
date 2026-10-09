"""
Plotting utilities for scalarised neutron–star scans.

This module provides routines to visualise the dimensionless charge to
mass ratio Q/ℳ as a function of the coupling parameters λ and ξ.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import (
    FuncFormatter,
    MaxNLocator,
    FixedLocator,
    NullLocator,
    NullFormatter,
    SymmetricalLogLocator,
)
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from Utils.params import M as M_CONST, hbar, c


# ---------- tick helpers ----------

SIZE_MAJOR_TICKS = 4.5
SIZE_MINOR_TICKS = 3


def decade_label(val: float, *, symbol=r"\lambda", unit=None):
    """Return latex label like μ=10^{exp} with optional unit."""
    val = float(val)

    if val == 0.0:
        label = rf"${symbol}=0$"
    elif not np.isfinite(val) or val <= 0:
        label = rf"${symbol}=?$"
    else:
        exp = int(np.round(np.log10(val)))
        label = rf"${symbol}=10^{{{exp}}}$"

    if unit is not None:
        label = label[:-1] + rf"\,\mathrm{{{unit}}}$"

    return label


def signed_sci10_even_only(y, _pos=None):
    """Label only ±10^{even} and 0."""
    if y == 0:
        return r"$0$"

    sign = "-" if y < 0 else ""
    v = abs(float(y))
    if v <= 0 or not np.isfinite(v):
        return ""

    exp = np.log10(v)
    e = int(np.round(exp))

    # label only exact decades
    if not np.isclose(v, 10**e, rtol=0.0, atol=v * 1e-12):
        return ""

    # label only even exponents
    if e % 2 != 0:
        return ""

    return rf"${sign}10^{{{e}}}$"


def signed_sci10_all(y, _pos=None):
    """Label any exact decade and 0."""
    if y == 0:
        return r"$0$"

    sign = "-" if y < 0 else ""
    v = abs(float(y))
    if v <= 0 or not np.isfinite(v):
        return ""

    exp = np.log10(v)
    e = int(np.round(exp))

    if not np.isclose(v, 10**e, rtol=0.0, atol=v * 1e-12):
        return ""

    return rf"${sign}10^{{{e}}}$"


def sci10_ticks_symlog(x, _pos):
    """
    Label-only formatter: show lambda values as 10^{exp}.

    NOTE: Here we treat the input x as already in the "natural units" you want
    to label. If you later want x*(ħc), reintroduce that scaling here.
    """
    if x == 0:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(float(x)))))
    return rf"$10^{{{exp}}}$"


def sci_a10_ticks(y, _pos):
    if y == 0:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(y))))
    mant = y / (10**exp)
    return rf"${mant:.3g}\times 10^{{{exp}}}$"

# ---------- IO ----------
_NUMERIC_COLS = (
    "xi",
    "mu",
    "lambda",
    "nu",
    "scalarized",
    "mode_n",
    "ADM_over_Msun",
    "Q_over_Msun",
    "Q_over_ADM",
)


def _load_csvs(path_or_list):
    files = (
        glob(os.path.join(path_or_list, "**", "*.csv"), recursive=True)
        if isinstance(path_or_list, (str, os.PathLike))
        else list(path_or_list)
    )
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path_or_list!r}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        for c in _NUMERIC_COLS:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # compute Q/M (dimensionless)
    if "Q_over_ADM" in df and df["Q_over_ADM"].notna().any():
        df["Q_over_M"] = df["Q_over_ADM"].astype(float)
    elif {"Q_over_Msun", "ADM_over_Msun"}.issubset(df.columns):
        df["Q_over_M"] = df["Q_over_Msun"] / df["ADM_over_Msun"]
    else:
        raise ValueError(
            "Need Q_over_ADM or (Q_over_Msun & ADM_over_Msun) to compute Q/M."
        )

    return df

# ---------- plotting ----------
def _plot_with_gaps(ax, x, y, *, max_dx, **plot_kw):
    """
    Plot (x,y) but break the curve whenever consecutive x points are farther
    apart than max_dx. This prevents connecting across missing scan regions.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return

    order = np.argsort(x)
    x, y = x[order], y[order]

    dx = np.diff(x)
    breaks = np.where(dx > max_dx)[0]

    start = 0
    for b in breaks:
        end = b + 1
        if end - start >= 2:
            ax.plot(x[start:end], y[start:end], **plot_kw)
        start = end

    if x.size - start >= 2:
        ax.plot(x[start:], y[start:], **plot_kw)


def _find_contiguous_true_regions(mask):
    """
    Return list of (start, end) index pairs for contiguous True regions.
    end is exclusive.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    regions = []
    in_region = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_region:
            start = i
            in_region = True
        elif not val and in_region:
            regions.append((start, i))
            in_region = False

    if in_region:
        regions.append((start, mask.size))

    return regions


def _detect_close_regions(
    curves,
    *,
    rel_tol=0.15,
    abs_tol=None,
    min_points=4,
    require_same_star_mode=True,
):
    """
    Detect x-regions where at least one PAIR of curves is close together.

    Curves are compared pairwise. By default, comparisons are only made
    between curves with the same (star, mode).

    Returns a list of dicts with keys:
        x_min, x_max, y_min, y_max, star, mode, n_points
    """
    if not curves:
        return []

    regions_out = []

    # ---- group curves ----
    groups = {}
    if require_same_star_mode:
        for c in curves:
            key = (c["star"], c["mode"])
            groups.setdefault(key, []).append(c)
    else:
        groups[("all", "all")] = list(curves)

    # ---- pairwise comparison inside each group ----
    for (star, mode), group in groups.items():
        if len(group) < 2:
            continue

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                c1 = group[i]
                c2 = group[j]

                x1 = np.asarray(c1["x"], float)
                y1 = np.asarray(c1["y"], float)
                x2 = np.asarray(c2["x"], float)
                y2 = np.asarray(c2["y"], float)

                m1 = np.isfinite(x1) & np.isfinite(y1)
                m2 = np.isfinite(x2) & np.isfinite(y2)

                x1, y1 = x1[m1], y1[m1]
                x2, y2 = x2[m2], y2[m2]

                if x1.size < 2 or x2.size < 2:
                    continue

                # sort
                o1 = np.argsort(x1)
                o2 = np.argsort(x2)
                x1, y1 = x1[o1], y1[o1]
                x2, y2 = x2[o2], y2[o2]

                # remove duplicate x values
                x1u, idx1 = np.unique(x1, return_index=True)
                y1u = y1[idx1]
                x2u, idx2 = np.unique(x2, return_index=True)
                y2u = y2[idx2]

                if x1u.size < 2 or x2u.size < 2:
                    continue

                # common overlap in x
                x_min = max(np.min(x1u), np.min(x2u))
                x_max = min(np.max(x1u), np.max(x2u))

                if not np.isfinite(x_min) or not np.isfinite(x_max):
                    continue
                if x_max <= x_min:
                    continue

                # common interpolation grid
                n_grid = min(400, max(80, max(len(x1u), len(x2u))))
                x_ref = np.linspace(x_min, x_max, n_grid)

                y1i = np.interp(x_ref, x1u, y1u)
                y2i = np.interp(x_ref, x2u, y2u)

                spread = np.abs(y1i - y2i)
                scale = np.maximum(np.maximum(np.abs(y1i), np.abs(y2i)), 1e-30)

                close_mask = (spread / scale) < rel_tol
                if abs_tol is not None:
                    close_mask |= spread < abs_tol

                close_mask &= np.isfinite(y1i) & np.isfinite(y2i)

                regions = _find_contiguous_true_regions(close_mask)

                for i0, i1 in regions:
                    if i1 - i0 < min_points:
                        continue

                    xr = x_ref[i0:i1]
                    y_pair = np.vstack([y1i[i0:i1], y2i[i0:i1]])

                    regions_out.append(
                        {
                            "star": star,
                            "mode": mode,
                            "x_min": float(np.min(xr)),
                            "x_max": float(np.max(xr)),
                            "y_min": float(np.nanmin(y_pair)),
                            "y_max": float(np.nanmax(y_pair)),
                            "n_points": int(i1 - i0),
                            "pair": (
                                c1.get("slice_val", None),
                                c2.get("slice_val", None),
                            ),
                        }
                    )

    return regions_out


def _merge_zoom_regions(regions, *, x_overlap_frac=0.2):
    if not regions:
        return []

    regions = sorted(regions, key=lambda r: (r["x_min"], r["x_max"]))
    merged = [regions[0].copy()]

    for r in regions[1:]:
        last = merged[-1]

        width_last = max(last["x_max"] - last["x_min"], 1e-30)
        width_r = max(r["x_max"] - r["x_min"], 1e-30)

        overlap = min(last["x_max"], r["x_max"]) - max(last["x_min"], r["x_min"])
        touches = overlap >= -x_overlap_frac * min(width_last, width_r)

        if touches:
            last["x_min"] = min(last["x_min"], r["x_min"])
            last["x_max"] = max(last["x_max"], r["x_max"])
            last["y_min"] = min(last["y_min"], r["y_min"])
            last["y_max"] = max(last["y_max"], r["y_max"])
            last["n_points"] += r["n_points"]
        else:
            merged.append(r.copy())

    return merged


def _add_zoom_insets(
    ax,
    curves,
    zoom_regions,
    *,
    max_windows=3,
    inset_size=("30%", "30%"),
    pad_x_frac=0.08,
    pad_y_frac=0.15,
    locs=None,
):
    """
    Draw zoom inset windows for the selected regions.
    """
    if not zoom_regions:
        return []

    # Keep the most substantial regions
    zoom_regions = sorted(zoom_regions, key=lambda r: r["n_points"], reverse=True)[
        :max_windows
    ]

    # Default positions of inset windows inside main axes
    #default_locs = [
    #    dict(bbox_to_anchor=(0.275, 0.25, 0.35, 0.35)),
    #    dict(bbox_to_anchor=(0.75, 0.25, 0.35, 0.35)),
    #]
    default_locs = [
        dict(bbox_to_anchor=(0.28, 0.2315, 0.25, 0.25)),
        dict(bbox_to_anchor=(0.22, 0.75, 0.25, 0.25)),
    ]

    if locs is None:
        locs = default_locs

    insets = []

    for k, region in enumerate(zoom_regions):
        if k >= len(locs):
            break

        loc_kw = locs[k]

        axins = inset_axes(
            ax,
            width=inset_size[0],
            height=inset_size[1],
            loc="upper left",
            bbox_transform=ax.transAxes,
            borderpad=0.8,
            **loc_kw,
        )

        x0, x1 = region["x_min"], region["x_max"]
        y0, y1 = region["y_min"], region["y_max"]

        # Start from the detected region
        dx = max(x1 - x0, 1e-30)
        x0_sel = x0 - pad_x_frac * dx
        x1_sel = x1 + pad_x_frac * dx

        # Find actual plotted data inside that x-window
        xmins_local = []
        xmaxs_local = []
        ymins_local = []
        ymaxs_local = []

        for c in curves:
            m = (
                (c["x"] >= x0_sel)
                & (c["x"] <= x1_sel)
                & np.isfinite(c["x"])
                & np.isfinite(c["y"])
            )
            if np.count_nonzero(m) >= 2:
                xmins_local.append(np.min(c["x"][m]))
                xmaxs_local.append(np.max(c["x"][m]))
                ymins_local.append(np.min(c["y"][m]))
                ymaxs_local.append(np.max(c["y"][m]))

        if xmins_local:
            x0p = min(xmins_local)
            x1p = max(xmaxs_local)
        else:
            x0p, x1p = x0_sel, x1_sel

        if ymins_local:
            y0 = min(ymins_local)
            y1 = max(ymaxs_local)

        dy = max(y1 - y0, 1e-30)
        y0p = y0 - pad_y_frac * dy
        y1p = y1 + pad_y_frac * dy

        for c in curves:
            m = (
                (c["x"] >= x0p)
                & (c["x"] <= x1p)
                & np.isfinite(c["x"])
                & np.isfinite(c["y"])
            )
            if np.count_nonzero(m) >= 2:
                axins.plot(
                    c["x"][m],
                    c["y"][m],
                    color=c["color"],
                    linestyle=c["linestyle"],
                    linewidth=c["linewidth"],
                )

        axins.set_xlim(x0p, x1p)
        axins.set_ylim(y0p, y1p)
        from matplotlib.ticker import FormatStrFormatter

        axins.tick_params(
            direction="in", which="both", top=True, right=True, labelsize=8
        )
        axins.yaxis.set_major_formatter(
            FormatStrFormatter("%.5f")
        )
        axins.grid(False)

        # rectangle on main plot + connectors
        pp, p1, p2 = mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")

        p1.set_visible(True)
        p2.set_visible(False)

        insets.append(axins)

    return insets


def plot_scan_results(
    *,
    base_dir="Results/scan",
    scan=None,
    xi=None,
    lmbda=None,  # float or list of floats for xi_scan
    mu=None,
    nu=0.0,
    stars=("L", "H"),
    include_modes=None,  # e.g. (0,1,2)
    scalarized_only=True,
    xi_tol=1e-12,
    figsize=(7.4, 4.4),
    dpi=600,
    out_path=None,
    EMG_region=False,
    connect_points=True,
    plot_zooms=False,
    manual_zoom_regions=None,
    markers=False,
    lw=1.6,
    ms=4.5,
    gap_factor=1.5,  # break if dx > gap_factor * (estimated xi step)
    headroom_decades_top=0,
    headroom_decades_bottom=0,
):
    """
    High-level dispatcher for paper figures from sz_scan.py CSVs.
    """
    df = _load_csvs(base_dir)

    # Zoom inset parameters
    zoom_max_windows = 1
    zoom_rel_tol = 0.001
    zoom_abs_tol = 1e-10
    zoom_min_points = 4
    zoom_pad_x_frac = 0.1
    zoom_pad_y_frac = 0.01
    zoom_inset_size = ("70%", "70%")
    zoom_locs = None

    curves = []

    # --- filter by scan type via filenames (avoid mixing xi_scan and lambda_scan) ---
    scan_norm = str(scan).strip().lower()

    if scan_norm in ("lambda_scan", "lmbda_scan", "lambda"):
        df = df[
            df["source_file"]
            .astype(str)
            .str.contains(r"lmbda_scan|lambda_scan", regex=True, na=False)
        ]
        scan_kind = "lambda_scan"

    elif scan_norm in ("mu_scan", "mu"):
        df = df[
            df["source_file"].astype(str).str.contains(r"mu_scan", regex=True, na=False)
        ]
        scan_kind = "mu_scan"

    elif scan_norm in ("xi_scan_lambda", "xi_lambda"):
        df = df[
            df["source_file"].astype(str).str.contains(r"lmbda=", regex=True, na=False)
        ]
        scan_kind = "xi_scan_lambda"

    elif scan_norm in ("xi_scan_mu", "xi_mu", "xi@mu", "xi_by_mu"):
        df = df[
            df["source_file"]
            .astype(str)
            .str.contains(r"xi_scan_mu=", regex=True, na=False)
        ]
        scan_kind = "xi_scan_mu"

    else:
        raise ValueError(
            "scan must be 'lambda_scan', 'xi_scan_lambda', 'xi_scan_mu', or 'mu_scan'."
        )

    required = {
        "xi",
        "mu",
        "lambda",
        "nu",
        "rho0_tag",
        "scalarized",
        "mode_n",
        "Q_over_M",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV(s): {missing}")

    # ---- base mask (common filters) ----
    m = (
        np.isfinite(df["xi"])
        & np.isfinite(df["mu"])
        & np.isfinite(df["lambda"])
        & np.isfinite(df["nu"])
    )

    if scalarized_only and "scalarized" in df.columns:
        m &= pd.to_numeric(df["scalarized"], errors="coerce") == 1

    # nu filter
    m &= np.isclose(
        pd.to_numeric(df["nu"], errors="coerce"), float(nu), atol=1e-15, rtol=0.0
    )
    # mu filter
    if mu is not None and scan_kind not in ("xi_scan_mu",):
        m &= np.isclose(
            pd.to_numeric(df["mu"], errors="coerce"), float(mu), atol=1e-15, rtol=0.0
        )

    # star filter
    stars = tuple(str(s) for s in stars)
    m &= df["rho0_tag"].astype(str).isin(stars)

    # ---- scan selection + x axis choice ----
    if scan_kind == "lambda_scan":
        if xi is None:
            raise ValueError("scan='lambda_scan' requires xi=...")
        if mu is None:
            raise ValueError(
                "scan='lambda_scan' requires mu=... (fixed mu for this scan)"
            )
        m &= np.isclose(
            pd.to_numeric(df["xi"], errors="coerce"), float(xi), atol=xi_tol, rtol=0.0
        )
        m &= np.isclose(
            pd.to_numeric(df["mu"], errors="coerce"), float(mu), atol=1e-15, rtol=0.0
        )

        xcol = "lambda"
        xlabel = r"$\lambda$"
        title = rf"$\xi={float(xi):g}$, $\mu={float(mu):g}\,$eV, $v={float(nu):g}\,$eV"
        slice_col = None
        slice_list = [None]

    elif scan_kind == "mu_scan":
        # Q/M vs mu at fixed xi and lambda
        if xi is None:
            raise ValueError("scan='mu_scan' requires xi=...")
        if lmbda is None:
            raise ValueError("scan='mu_scan' requires lmbda=... (fixed lambda)")
        m &= np.isclose(
            pd.to_numeric(df["xi"], errors="coerce"), float(xi), atol=xi_tol, rtol=0.0
        )

        lam_col = pd.to_numeric(df["lambda"], errors="coerce").to_numpy(dtype=float)
        lam_sel = float(lmbda)
        if lam_sel == 0.0:
            m &= lam_col == 0.0
        else:
            m &= np.isclose(lam_col, lam_sel, rtol=1e-6, atol=0.0)

        xcol = "mu"
        xlabel = r"$\mu$"
        title = rf"$\xi={float(xi):g}$, $\lambda={float(lmbda):g}$, $v={float(nu):g}$"
        slice_col = None
        slice_list = [None]

    elif scan_kind == "xi_scan_lambda":
        # Q/M vs xi at fixed mu and (possibly multiple) lambda values
        if mu is None:
            raise ValueError("scan='xi_scan_lambda' requires mu=... (fixed mu)")
        if lmbda is None:
            raise ValueError("scan='xi_scan_lambda' requires lmbda=... (float or list)")

        m &= np.isclose(
            pd.to_numeric(df["mu"], errors="coerce"), float(mu), atol=1e-15, rtol=0.0
        )

        slice_col = "lambda"
        slice_list = (
            list(lmbda)
            if isinstance(lmbda, (list, tuple, np.ndarray))
            else [float(lmbda)]
        )

        xcol = "xi"
        xlabel = r"$\xi$"
        title = (
            rf"$\mu={float(mu):g}\,"
            + r"\mathrm{eV}$, "
            + rf"$v={float(nu):g}\,"
            + r"\mathrm{eV}$"
        )

    elif scan_kind == "xi_scan_mu":
        # Q/M vs xi at fixed lambda and (possibly multiple) mu values
        if lmbda is None:
            raise ValueError("scan='xi_scan_mu' requires lmbda=... (fixed lambda)")
        if mu is None:
            raise ValueError(
                "scan='xi_scan_mu' requires mu=... (float or list of mu values)"
            )

        # enforce fixed lambda
        lam_col = pd.to_numeric(df["lambda"], errors="coerce").to_numpy(dtype=float)
        lam_sel = float(lmbda)
        if lam_sel == 0.0:
            m &= lam_col == 0.0
        else:
            m &= np.isclose(lam_col, lam_sel, rtol=1e-6, atol=0.0)

        slice_col = "mu"
        slice_list = (
            list(mu) if isinstance(mu, (list, tuple, np.ndarray)) else [float(mu)]
        )

        xcol = "xi"
        xlabel = r"$\xi$"
        title = rf"$\lambda={float(lmbda):g}$, $v={float(nu):g}\,$eV"

    # ---- figure ----
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    # styling: use markers for mode, not color
    star_colors = {"L": "#1f77b4", "H": "#d62728"}
    mode_markers = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v"}

    # For xi scans: color by star (blue/red) and line style by slice (mu or lambda).
    node_distinction = not (xcol == "xi" and slice_col in ("lambda", "mu"))
    slice_linestyles = ["-", "--", ":", "-."]
    slice_style_map = {}
    if not node_distinction:
        for i, _v in enumerate(slice_list):
            slice_style_map[i] = slice_linestyles[i % len(slice_linestyles)]

    # estimate xi step for gap breaking (only needed for xi_scan)
    xi_unique = np.sort(pd.to_numeric(df["xi"], errors="coerce").dropna().unique())
    xi_step = float(np.median(np.diff(xi_unique))) if xi_unique.size >= 2 else 0.25
    max_dx_gap = float(gap_factor) * xi_step

    # Track what actually got plotted (for ticks/legends/limits)
    ys_plotted = []
    xs_plotted = []
    modes_plotted = set()

    # ---- plot per slice (lambda OR mu) ----
    any_plotted = False

    for slice_num, slice_val in enumerate(slice_list):
        if slice_col is None:
            d = df[m].copy()
        else:
            col = pd.to_numeric(df[slice_col], errors="coerce").to_numpy(dtype=float)
            sel = float(slice_val)
            if sel == 0.0:
                m_local = m & (col == 0.0)
            else:
                m_local = m & np.isclose(col, sel, rtol=1e-6, atol=0.0)
            d = df[m_local].copy()

        if d.empty:
            continue

        d = d[np.isfinite(d["Q_over_M"])]
        d["mode_n"] = pd.to_numeric(d["mode_n"], errors="coerce")
        d = d.dropna(subset=["mode_n", xcol, "Q_over_M"])
        d["mode_n"] = d["mode_n"].astype(int)

        if include_modes is not None:
            include_modes_set = set(int(n) for n in include_modes)
            d = d[d["mode_n"].isin(include_modes_set)]
        if d.empty:
            continue

        d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
        d = (
            d.groupby(["rho0_tag", "mode_n", xcol], as_index=False)["Q_over_M"]
            .median()
            .sort_values(["rho0_tag", "mode_n", xcol])
        )

        for star in stars:
            for n in sorted(d["mode_n"].unique()):
                dn = d[(d["rho0_tag"] == star) & (d["mode_n"] == n)].sort_values(xcol)
                if dn.empty:
                    continue

                x = dn[xcol].to_numpy(dtype=float)
                y = dn["Q_over_M"].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if not np.any(mask):
                    continue

                if node_distinction:
                    line_color = star_colors.get(star, "black")
                    line_style = slice_linestyles[int(n) % len(slice_linestyles)]
                    marker_style = "o"
                else:
                    line_color = star_colors.get(star, "black")
                    line_style = slice_style_map.get(slice_num, "-")
                    marker_style = "o"

                kw = dict(
                    color=line_color,
                    linestyle=line_style,
                    linewidth=lw if connect_points else 0.0,
                    marker=(marker_style if markers else None),
                    markersize=(ms if markers else 0.0),
                    markerfacecolor=line_color,
                    markeredgecolor=line_color,
                    markeredgewidth=0.6,
                )

                xx = x[mask]
                yy = y[mask]

                curves.append(
                    {
                        "x": xx.copy(),
                        "y": yy.copy(),
                        "star": star,
                        "mode": int(n),
                        "slice_num": slice_num,
                        "slice_val": slice_val,
                        "color": line_color,
                        "linestyle": line_style,
                        "linewidth": lw,
                    }
                )

                if xcol == "xi":
                    _plot_with_gaps(ax, xx, yy, max_dx=max_dx_gap, **kw)
                else:
                    ax.plot(xx, yy, **kw)

                xs_plotted.append(xx)
                ys_plotted.append(yy)
                modes_plotted.add(int(n))
                any_plotted = True

        # slice_num increments via enumerate

    if not any_plotted:
        raise ValueError("Nothing was plotted (no matching data after filtering).")

    # bounds
    ax.axhspan(-6e-4, 6e-4, color="gray", alpha=0.12, zorder=0)
    ax.axhline(6e-4, color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(-6e-4, color="gray", linestyle="--", linewidth=1.0)
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transAxes, ax.transData)

    ax.text(
        0.25,  # x = 3/4 of axis width
        6e-4,  # y = absolute data value
        "PSR J1738+0333",
        transform=trans,
        fontsize=10,
        color="gray",
        ha="left",
        va="bottom",
    )
    # ---------------- X headroom ONLY on the right (set before tick locators) ----------------
    # NOTE: we do this before configuring tick locators so ticks can extend into headroom.
    x_pad_frac_effective = 10 if scan == "lambda_scan" else 0.0
    if xs_plotted and xcol in ("lambda", "mu"):
        xs = np.concatenate([xx[np.isfinite(xx)] for xx in xs_plotted if xx.size])
        if xs.size:
            xmin, xmax = float(np.min(xs)), float(np.max(xs))
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                dx = (xmax - xmin) * float(x_pad_frac_effective)
                ax.set_xlim(xmin, xmax + dx)

    # ---------------- X axis formatting: ticks at every point, labels every 2 ----------------
    if xcol in ("lambda", "mu"):
        # Use real lambda values on a symlog x-axis (to allow lambda=0)
        lam_all = np.concatenate([xx[np.isfinite(xx)] for xx in xs_plotted if xx.size])
        lam_pos = lam_all[lam_all > 0.0]
        if lam_pos.size == 0:
            raise ValueError("No positive lambda values to set a log scale.")

        linthresh_x = (
            float(np.min(lam_pos)) * 0.5
        )  # transition near smallest positive λ
        ax.set_xscale("symlog", linthresh=linthresh_x, linscale=1.0, base=10)

        # Nice decade ticks (minor every decade; major every 2 decades)
        # Extend tick range to the padded xlim (headroom).
        xlim_left, xlim_right = ax.get_xlim()
        lam_pos_max = float(np.max(lam_pos))
        if xlim_right > 0.0:
            lam_pos_max = max(lam_pos_max, float(xlim_right))

        emax = int(np.floor(np.log10(lam_pos_max)))
        emin = int(np.floor(np.log10(np.min(lam_pos))))

        # minor ticks every decade
        minor = [0.0] + [10.0**e for e in range(emin, emax + 1)]
        ax.xaxis.set_minor_locator(FixedLocator(minor))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=SIZE_MINOR_TICKS)

        # major ticks every 2 decades
        major = [0.0] + [10.0**e for e in range(emin, emax + 1, 2)]
        ax.xaxis.set_major_locator(FixedLocator(major))

        def _fmt_lam(x, _pos):
            if x == 0.0:
                return r"$0$"
            exp = int(np.round(np.log10(abs(float(x)))))
            return rf"$10^{{{exp}}}$"

        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lam))
        ax.tick_params(axis="x", which="major", length=SIZE_MAJOR_TICKS)

    else:
        # xi scan: clean symmetric ticks with ONE minor tick between majors
        xi_all = np.sort(pd.to_numeric(df[m]["xi"], errors="coerce").dropna().unique())
        if xi_all.size == 0:
            raise ValueError("No xi points available after filtering.")

        xmin, xmax = float(xi_all.min()), float(xi_all.max())
        ax.set_xlim(xmin, xmax)

        # --- Major ticks (labeled) ---
        # If symmetric like [-10, 10], force the clean 5-point grid
        if np.isclose(xmin, -xmax, atol=1e-12):
            A = max(abs(xmin), abs(xmax))
            major_ticks = [-A, -A / 2, 0.0, A / 2, A]
        else:
            # fallback: 5 evenly spaced nice ticks
            major_ticks = np.linspace(xmin, xmax, 5)

        ax.xaxis.set_major_locator(FixedLocator(major_ticks))
        ax.tick_params(axis="x", which="major", length=SIZE_MAJOR_TICKS)

        def _fmt_xi(x, _pos):
            if np.isclose(x, round(x), atol=1e-10):
                return rf"${int(round(x))}$"
            return rf"${x:.1f}$"

        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_xi))

        # --- Minor ticks: subdivide each major interval ---
        minor_ticks = []

        n_sub = 5  # number of subdivisions per interval

        for i in range(len(major_ticks) - 1):
            left = major_ticks[i]
            right = major_ticks[i + 1]
            step = (right - left) / n_sub

            for k in range(1, n_sub):
                minor_ticks.append(left + k * step)

        ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=SIZE_MINOR_TICKS)

        if EMG_region:
            # --- EMG shaded region ---
            xi_emg_min = 0.0
            xi_emg_max = 0.63

            ax.axvspan(
                xi_emg_min,
                xi_emg_max,
                color="#3274e6",
                alpha=0.5,
                zorder=0,  # soft blue
            )

            # EMG label centered in region
            # x_center = 0.5 * (xi_emg_min + xi_emg_max)
            # ymin, ymax = ax.get_ylim()
            # y_center = 0.5 * ymax

            ax.text(
                -1.1,
                1e-1,
                r"$\mathrm{EMG}$",
                fontsize=10,
                color="#3274e6",
                zorder=5,
            )

    # ---------------- Y axis formatting: symlog + symmetric + minor every decade ----------------
    if not ys_plotted:
        raise ValueError("No finite y-data was plotted.")
    y_all = np.concatenate([yy[np.isfinite(yy)] for yy in ys_plotted if yy.size])
    yabs = np.abs(y_all[y_all != 0.0])
    if yabs.size == 0:
        yabs = np.array([1.0])

    y_abs_min = float(np.min(yabs))
    linthresh_y = max(2 * y_abs_min if np.any(y_all < 0) else 1e-20, 1e-9)

    ax.set_yscale("symlog", linthresh=linthresh_y, linscale=1.0)

    y_all = np.concatenate([yy[np.isfinite(yy)] for yy in ys_plotted if yy.size])

    def _decade_above(v, headroom_decades):
        e = int(np.floor(np.log10(v)))
        return 10.0 ** (e + headroom_decades)

    def _decade_below(v, headroom_decades):
        e = int(np.floor(np.log10(v)))
        return 10.0 ** (e - headroom_decades)

    pos_vals = y_all[y_all > 0.0]
    neg_vals = y_all[y_all < 0.0]

    if pos_vals.size:
        y_high = _decade_above(float(np.max(pos_vals)), headroom_decades_top)
    elif neg_vals.size:
        # all negative: move upper limit one decade closer to zero
        y_high = -_decade_below(
            float(np.max(np.abs(neg_vals))), headroom_decades_bottom
        )
    else:
        y_high = 1.0

    if neg_vals.size:
        y_low = -_decade_above(float(np.max(np.abs(neg_vals))), headroom_decades_top)
    elif pos_vals.size:
        y_low = _decade_below(float(np.min(pos_vals)), headroom_decades_bottom)
    else:
        y_low = -1.0

    if y_low > 6e-4:
        y_low = 1e-4
    ax.set_ylim(y_low, y_high)

    ax.yaxis.set_major_locator(
        SymmetricalLogLocator(base=10, linthresh=linthresh_y, subs=(1.0,))
    )
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(
            base=10, linthresh=linthresh_y, subs=np.arange(2, 10) * 0.1
        )
    )

    y_abs_max = float(max(abs(y_low), abs(y_high)))
    e_max = int(np.ceil(np.log10(y_abs_max))) if y_abs_max > 0 else 0
    if neg_vals.size:
        e_min = int(np.ceil(np.log10(linthresh_y)))
    else:
        e_min = int(np.floor(np.log10(max(y_low, np.min(pos_vals)))))

    num_decades = e_max - e_min + 1
    # minor ticks: every decade
    exps_minor = list(range(e_max, e_min - 1, -1))
    minor_pos = [10.0**e for e in exps_minor]
    minor_ticks = sorted([-t for t in minor_pos] + [0.0] + minor_pos)

    # major ticks: every 2 decades (labeled)
    if num_decades <= 6:
        exps_major = list(range(e_max, e_min - 1, -1))
        major_formatter = signed_sci10_all
    else:
        exps_major = list(range(e_max, e_min - 1, -2))
        major_formatter = signed_sci10_even_only
    major_pos = [10.0**e for e in exps_major]
    major_ticks = sorted([-t for t in major_pos] + [0.0] + major_pos)

    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(major_formatter))

    ax.tick_params(axis="y", which="minor", length=SIZE_MINOR_TICKS)
    ax.tick_params(axis="y", which="major", length=SIZE_MAJOR_TICKS)

    # Zoom insets
    if plot_zooms and xcol == "xi" and len(curves) >= 2:
        if manual_zoom_regions is not None:
            zoom_regions = manual_zoom_regions
        elif plot_zooms:
            
            zoom_regions = _detect_close_regions(
                curves,
                rel_tol=zoom_rel_tol,
                abs_tol=zoom_abs_tol,
                min_points=zoom_min_points,
            )
            zoom_regions = _merge_zoom_regions(zoom_regions)
       
        _add_zoom_insets(
            ax,
            curves,
            zoom_regions,
            max_windows=zoom_max_windows,
            inset_size=zoom_inset_size,
            pad_x_frac=zoom_pad_x_frac,
            pad_y_frac=zoom_pad_y_frac,
            locs=zoom_locs,
        )

    # cosmetics
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$Q/\mathcal{M}$")
    ax.grid(False)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_linewidth(1.0)
    ax.set_title(title, pad=6)

    # ---------------- Legends ----------------

    # ---- Star handles ----
    star_handles = [
        mlines.Line2D(
            [0],
            [0],
            color=star_colors.get("L", "black"),
            linestyle="-",
            linewidth=1.8,
            label="Light star",
        ),
        mlines.Line2D(
            [0],
            [0],
            color=star_colors.get("H", "black"),
            linestyle="-",
            linewidth=1.8,
            label="Heavy star",
        ),
    ]

    # ---- Mode handles ----
    mode_handles = []
    for n in sorted(modes_plotted):
        if node_distinction:
            mode_handles.append(
                mlines.Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=slice_linestyles[int(n) % len(slice_linestyles)],
                    linewidth=1.8,
                    label=rf"$n={n}$",
                )
            )
        else:
            # text-only row
            mode_handles.append(mlines.Line2D([], [], color="none", label=rf"$n={n}$"))

    # ---- Slice handles ----
    slice_handles = []
    if slice_col in ("lambda", "mu") and len(slice_list) > 1:
        sym = r"\lambda" if slice_col == "lambda" else r"\mu"

        if slice_col == "mu":
            slice_labels = [decade_label(v, symbol=sym, unit="eV") for v in slice_list]
        else:
            slice_labels = [decade_label(v, symbol=sym) for v in slice_list]

        if xcol == "xi":
            slice_handles = [
                mlines.Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=slice_style_map.get(i, "-"),
                    linewidth=1.8,
                    label=lab,
                )
                for i, lab in enumerate(slice_labels)
            ]
        else:
            slice_handles = [
                mlines.Line2D([], [], color="none", label=lab) for lab in slice_labels
            ]

    # ---- Build one combined legend ----
    legend_handles = []

    # Star section
    # legend_handles.append(mlines.Line2D([], [], color="none", label=r"$\bf{Star}$"))
    legend_handles.extend(star_handles)

    # Slice section
    if slice_handles:
        section_title = r"$\bf{\lambda}$" if slice_col == "lambda" else r"$\bf{\mu}$"
        # legend_handles.append(mlines.Line2D([], [], color="none", label=section_title))
        legend_handles.extend(slice_handles)

    # Mode section
    # legend_handles.append(mlines.Line2D([], [], color="none", label=r"$\bf{Mode}$"))
    legend_handles.extend(mode_handles)

    legend_labels = [h.get_label() for h in legend_handles]

    ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        #bbox_to_anchor=(0, 0.4),  # for xi EMG graph
        frameon=True,
        handlelength=2.0,
        handletextpad=0.6,
        borderpad=0.6,
        labelspacing=0.6,
    )

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")

    return fig, ax
