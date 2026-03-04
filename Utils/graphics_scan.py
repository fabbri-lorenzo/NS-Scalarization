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


from Utils.params import M as M_CONST, hbar, c


# ---------- tick helpers ----------
def _p_from_lambda(lmbda, M=M_CONST):
    if not np.isfinite(lmbda) or lmbda <= 0.0:
        return None
    return float(-np.log(lmbda) / np.log(M))


def _lambda_tick_formatter(M=M_CONST, decimals=1):
    def _fmt(x, _pos):
        if x == 0:
            return "0"
        p = _p_from_lambda(x, M=M)
        if p is None:
            return ""
        p = round(p, decimals)
        if abs(p) < 10 ** (-decimals):
            p = 0.0
        return r"$1/M_{Pl}^{%.*f}$" % (decimals, p)

    return FuncFormatter(_fmt)


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
    if e % 4 != 0:
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


def _drop_ticks_too_close_to_zero(ax, ticks, *, min_sep_px=14):
    """
    Remove symmetric tick pairs that are too close (in screen pixels) to the 0 tick,
    to avoid label/mark superposition near the symlog linear region.

    Keeps 0 if present. Drops both +t and -t together.
    """
    # Need a draw so transforms know the final layout (important when saving to PDF)
    ax.figure.canvas.draw()

    trans = ax.transData
    y0_px = trans.transform((0.0, 0.0))[1]

    ticks_set = set(float(t) for t in ticks)
    keep = set([0.0]) if 0.0 in ticks_set else set()

    # consider positive ticks only, then keep them symmetrically
    pos = sorted([t for t in ticks_set if t > 0], reverse=True)

    for t in pos:
        y_px = trans.transform((0.0, t))[1]
        if abs(y_px - y0_px) < min_sep_px:
            # too close -> stop adding smaller decades (they'll be even closer)
            break
        keep.add(t)
        if -t in ticks_set:
            keep.add(-t)

    # preserve original ordering
    return [t for t in ticks if float(t) in keep]


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


def plot_scan_results(
    *,
    base_dir="Results/scan",
    scan="lambda_scan",  # "lambda_scan" or "xi_scan"
    xi=None,
    lmbda=None,  # float or list of floats for xi_scan
    mu=0.0,
    nu=0.0,
    stars=("L", "H"),
    include_modes=None,  # e.g. (0,1,2)
    scalarized_only=True,
    xi_tol=1e-12,
    figsize=(7.4, 4.4),
    dpi=300,
    out_path=None,
    natural_units=True,
    connect_points=True,
    markers=False,
    lw=1.6,
    ms=4.5,
    gap_factor=1.5,  # break if dx > gap_factor * (estimated xi step)
    # headroom
    x_pad_frac=0.03,  # add padding ONLY on the right
    y_pad_decades=0.25,  # symmetric y headroom (in decades)
):
    """
    High-level dispatcher for paper figures from sz_scan.py CSVs.

    - scan="lambda_scan": Q/M vs lambda at fixed xi
    - scan="xi_scan":     Q/M vs xi at fixed lambda (supports multiple lambdas)

    Overlays Light (L) and Heavy (H) on the same axes.
    Uses a signed "log-like" y-axis (symlog) with:
      - y minor ticks every decade (tick marks only)
      - y major ticks every 2 decades (labeled)
    For x-axis:
      - ticks at every scanned point
      - labels every 2 points
    """
    df = _load_csvs(base_dir)

    # --- filter by scan type via filenames (avoid mixing xi_scan and lambda_scan) ---
    scan_norm = str(scan).strip().lower()
    if scan_norm in ("lambda_scan", "lmbda_scan", "lambda"):
        df = df[
            df["source_file"]
            .astype(str)
            .str.contains("lmbda_scan|lambda_scan", regex=True, na=False)
        ]
    elif scan_norm in ("xi_scan", "xi"):
        df = df[
            df["source_file"].astype(str).str.contains("xi_scan", regex=True, na=False)
        ]
    else:
        raise ValueError("scan must be 'lambda_scan' or 'xi_scan'.")

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
    m &= np.isclose(
        pd.to_numeric(df["mu"], errors="coerce"), float(mu), atol=1e-15, rtol=0.0
    )

    # star filter
    stars = tuple(str(s) for s in stars)
    m &= df["rho0_tag"].astype(str).isin(stars)

    # ---- scan selection + x axis choice ----
    if scan_norm in ("lambda_scan", "lmbda_scan", "lambda"):
        if xi is None:
            raise ValueError("scan='lambda_scan' requires xi=...")
        m &= np.isclose(
            pd.to_numeric(df["xi"], errors="coerce"), float(xi), atol=xi_tol, rtol=0.0
        )

        xcol = "lambda"
        xlabel = r"$\lambda$"
        title = rf"$\xi={float(xi):g}$, $\mu={float(mu):g}$, $v={float(nu):g}$"
        lmbda_list = [None]  # not used here

    else:
        # xi_scan
        if lmbda is None:
            raise ValueError("scan='xi_scan' requires lmbda=... (float or list)")

        lmbda_list = (
            list(lmbda)
            if isinstance(lmbda, (list, tuple, np.ndarray))
            else [float(lmbda)]
        )

        xcol = "xi"
        xlabel = r"$\xi$"
        title = rf"$\mu={float(mu):g}$, $v={float(nu):g}$"

    # ---- figure ----
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    # styling: color=mode, linestyle=star
    mode_colors = {0: "#1f77b4", 1: "#8658C2", 2: "#f4828f", 3: "#1e988a"}
    star_ls = {"L": "-", "H": "--"}
    mode_markers = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v"}

    # estimate xi step for gap breaking (only needed for xi_scan)
    xi_unique = np.sort(pd.to_numeric(df["xi"], errors="coerce").dropna().unique())
    xi_step = float(np.median(np.diff(xi_unique))) if xi_unique.size >= 2 else 0.25
    max_dx_gap = float(gap_factor) * xi_step

    # Will be set for lambda_scan
    lam_order = None

    # Track what actually got plotted (for ticks/legends/limits)
    ys_plotted = []
    xs_plotted = []
    modes_plotted = set()

    # ----- Lambda legend handles (xi_scan only) -----
    lambda_handles = []
    if xcol == "xi":

        def _lam_label(lam, i):
            lam = float(lam)
            if lam == 0.0:
                return rf"$\lambda = 0$"
            exp = int(np.floor(np.log10(abs(lam))))
            return rf"$\lambda = 10^{{{exp}}}$"

        for i, L in enumerate(lmbda_list):
            lambda_handles.append(
                mlines.Line2D([], [], color="none", label=_lam_label(L, i))
            )
            i += 1

    any_plotted = False

    # ---- plot per lambda slice (xi_scan can have multiple) ----
    for lmbda_sel in lmbda_list:
        if xcol == "xi":
            lam_col = pd.to_numeric(df["lambda"], errors="coerce").to_numpy(dtype=float)
            lam_sel = float(lmbda_sel)

            if lam_sel == 0.0:
                m_local = m & (lam_col == 0.0)
            else:
                # relative match for tiny numbers; absolute tol must be 0 here
                m_local = m & np.isclose(lam_col, lam_sel, rtol=1e-6, atol=0.0)

            d = df[m_local].copy()
        else:
            d = df[m].copy()

        if d.empty:
            continue

        # keep only rows with valid Q/M
        d = d[np.isfinite(d["Q_over_M"])]

        d["mode_n"] = pd.to_numeric(d["mode_n"], errors="coerce")
        d = d.dropna(subset=["mode_n", xcol, "Q_over_M"])
        d["mode_n"] = d["mode_n"].astype(int)

        if include_modes is not None:
            include_modes_set = set(int(n) for n in include_modes)
            d = d[d["mode_n"].isin(include_modes_set)]

        if d.empty:
            continue

        # deduplicate: median per (star, mode, x)
        d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
        d = (
            d.groupby(["rho0_tag", "mode_n", xcol], as_index=False)["Q_over_M"]
            .median()
            .sort_values(["rho0_tag", "mode_n", xcol])
        )

        # draw curves
        for star in stars:
            for n in sorted(d["mode_n"].unique()):
                dn = d[(d["rho0_tag"] == star) & (d["mode_n"] == n)].sort_values(xcol)
                if dn.empty:
                    continue

                if xcol == "lambda":
                    x = dn["lambda"].to_numpy(dtype=float)
                else:
                    x = dn["xi"].to_numpy(dtype=float)

                y = dn["Q_over_M"].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if not np.any(mask):
                    continue

                kw = dict(
                    color=mode_colors.get(n, "black"),
                    linestyle=star_ls.get(star, "-."),
                    linewidth=lw if connect_points else 0.0,
                    marker=(mode_markers.get(n, "o") if markers else None),
                    markersize=(ms if markers else 0.0),
                    markerfacecolor=mode_colors.get(n, "black"),
                    markeredgecolor=mode_colors.get(n, "black"),
                    markeredgewidth=0.6,
                )

                xx = x[mask]
                yy = y[mask]

                if xcol == "xi":
                    _plot_with_gaps(ax, xx, yy, max_dx=max_dx_gap, **kw)
                else:
                    ax.plot(xx, yy, **kw)

                xs_plotted.append(xx)
                ys_plotted.append(yy)
                modes_plotted.add(int(n))
                any_plotted = True

    if not any_plotted:
        raise ValueError("Nothing was plotted (no matching data after filtering).")

    # bounds
    ax.axhline(6e-4, color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(-6e-4, color="gray", linestyle="--", linewidth=1.0)

    # ---------------- X axis formatting: ticks at every point, labels every 2 ----------------
    if xcol == "lambda":
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
        emax = int(np.floor(np.log10(np.max(lam_pos))))
        emin = int(np.floor(np.log10(np.min(lam_pos))))

        # minor ticks every decade
        minor = [0.0] + [10.0**e for e in range(emin, emax + 1)]
        ax.xaxis.set_minor_locator(FixedLocator(minor))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=3)

        # major ticks every 2 decades
        major = [0.0] + [10.0**e for e in range(emin, emax + 1, 2)]
        ax.xaxis.set_major_locator(FixedLocator(major))

        def _fmt_lam(x, _pos):
            if x == 0.0:
                return r"$0$"
            exp = int(np.round(np.log10(abs(float(x)))))
            return rf"$10^{{{exp}}}$"

        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lam))
        ax.tick_params(axis="x", which="major", length=4)

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
        ax.tick_params(axis="x", which="major", length=4)

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
        ax.tick_params(axis="x", which="minor", length=3)

        # --- EMG shaded region ---
        xi_emg_min = 0.0
        xi_emg_max = 0.63

        ax.axvspan(
            xi_emg_min, xi_emg_max, color="#3274e6", alpha=0.5, zorder=0  # soft blue
        )

        # EMG label centered in region
        x_center = 0.5 * (xi_emg_min + xi_emg_max)
        ymin, ymax = ax.get_ylim()
        y_center = 0.5 * ymax

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
    linthresh_y = max(1e-12, 0.5 * y_abs_min)

    ax.set_yscale("symlog", linthresh=linthresh_y, linscale=1.0)

    ax.yaxis.set_major_locator(
        SymmetricalLogLocator(base=10, linthresh=linthresh_y, subs=(1.0,))
    )
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(
            base=10, linthresh=linthresh_y, subs=np.arange(2, 10) * 0.1
        )
    )

    y_abs_max = float(np.max(yabs))
    pad_factor = 1e3 ** float(y_pad_decades)
    ylim = y_abs_max * pad_factor
    ax.set_ylim(-ylim, +ylim)  # symmetric: avoids “disappearing branch”

    # Build decade ticks
    e_max = int(np.ceil(np.log10(ylim))) if ylim > 0 else 0
    e_min = int(np.ceil(np.log10(linthresh_y)))  # stop before too close to 0

    # minor ticks: every decade
    exps_minor = list(range(e_max, e_min - 1, -1))
    minor_pos = [10.0**e for e in exps_minor]
    minor_ticks = sorted([-t for t in minor_pos] + [0.0] + minor_pos)

    # major ticks: every 2 decades (labeled)
    exps_major = list(range(e_max, e_min - 1, -2))
    major_pos = [10.0**e for e in exps_major]
    major_ticks = sorted([-t for t in major_pos] + [0.0] + major_pos)

    # Drop ticks too close to 0 (pixel-aware)
    major_ticks = _drop_ticks_too_close_to_zero(ax, major_ticks, min_sep_px=14)
    minor_ticks = _drop_ticks_too_close_to_zero(ax, minor_ticks, min_sep_px=10)

    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(signed_sci10_even_only))

    ax.tick_params(axis="y", which="minor", length=3)
    ax.tick_params(axis="y", which="major", length=4)

    # cosmetics
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$Q/\mathcal{M}$")
    ax.grid(False)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_linewidth(1.0)
    ax.set_title(title, pad=6)

    # ---------------- Legends (separate blocks) ----------------
    # Star legend
    star_handles = [
        mlines.Line2D(
            [0], [0], color="black", linestyle="-", linewidth=1.8, label="Light star"
        ),
        mlines.Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1.8, label="Heavy star"
        ),
    ]
    leg_star = ax.legend(
        handles=star_handles,
        loc="lower right",
        bbox_to_anchor=(1, 0.025),
        frameon=True,
    )
    ax.add_artist(leg_star)

    # Mode legend
    mode_handles = []
    for n in sorted(modes_plotted):
        mode_handles.append(
            mlines.Line2D(
                [0],
                [0],
                color=mode_colors.get(n, "black"),
                linestyle="-",
                linewidth=1.8,
                label=rf"$n={n}$",
            )
        )
    leg_mode = ax.legend(
        handles=mode_handles,
        loc="lower right",
        bbox_to_anchor=(1, 0.15),  # xi scan
        # bbox_to_anchor=(1, 0.3), #lambda scan
        frameon=True,
    )
    ax.add_artist(leg_mode)

    # Lambda legend (xi_scan only) — no handle column => no white space
    # if lambda_handles:
    #    leg_lambda = ax.legend(
    #        handles=lambda_handles,
    #        loc="right",
    #        bbox_to_anchor=(1, 0.4),
    #        frameon=True,
    #        handlelength=0.0,
    #        handletextpad=0.2,
    #        borderpad=0.6,
    #        labelspacing=0.4,
    #    )
    #    ax.add_artist(leg_lambda)

    # ---------------- X headroom ONLY on the right ----------------
    x_pad_frac = 10 if scan == "lambda_scan" else 0.0
    if xs_plotted:
        xs = np.concatenate([xx[np.isfinite(xx)] for xx in xs_plotted if xx.size])
        if xs.size:
            xmin, xmax = float(np.min(xs)), float(np.max(xs))
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                dx = (xmax - xmin) * float(x_pad_frac)
                ax.set_xlim(xmin, xmax + dx)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")

    return fig, ax
