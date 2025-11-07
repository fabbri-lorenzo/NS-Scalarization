# phase_diagram_filled.py
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
from Utils.params import hbar, c, M as M_CONST


# ---------- helpers ----------    
def sci_ticks(x, pos):
    if x == 0:
        return r"$0$"
    x_scaled = x * (hbar * c)  # natural-units relabeling only
    exp = int(np.floor(np.log10(abs(x_scaled))))
    return rf"$10^{{{exp}}}$"

    
def _p_from_lambda(lmbda, M=M_CONST):
    """Return p such that lmbda = 1/M^p. If lmbda<=0, return None."""
    if not np.isfinite(lmbda) or lmbda <= 0.0:
        return None
    return float(-np.log(lmbda) / np.log(M))

def _is_high(tag):
    """Return True if the mode index in tag is >= 3.
    Works with '3+', '3-', '2', 3, etc."""
    if isinstance(tag, str):
        core = tag[:-1] if tag and tag[-1] in "+-" else tag
    else:
        core = tag
    try:
        n = int(core)
    except Exception:
        return False
    return n >= 3

def _collapse_high(tpl):
    """If any tag in combo has n>=3, collapse the whole combo to the sentinel ('HIGH',)."""
    return ('HIGH',) if any(_is_high(t) for t in tpl) else tpl


def _lambda_tick_formatter(M=M_CONST, nautral_units=False, decimals=1):
    def _fmt(y, _pos):
        if y == 0:
            return "0"
        if nautral_units:
            return f"{y}"
        else:    
          p = _p_from_lambda(y, M=M)
          if p is None:
              return ""
          p = round(p, decimals)
          if abs(p) < 10 ** (-decimals):
              p = 0.0
          return r"$1/M_{Pl}^{%.*f}$" % (decimals, p)
    return FuncFormatter(_fmt)


def _tag_from_row(n, sgn):
    n = int(n)
    s = "+" if float(sgn) >= 0 else "-"
    return f"{n}{s}"


def _tag_sort_key(tag):
    n = int(tag[:-1])
    s = +1 if tag[-1] == "+" else -1
    return (n, -s)  # "+" before "-"


def _load(files):
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    for col in ("xi", "lambda", "mode_n", "scalarized", "vacuum_sign"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _grid_min_mode(df, xi_vals=None, lam_vals=None):
    if xi_vals is None:
        xi_vals = np.sort(df["xi"].unique())
    if lam_vals is None:
        lam_vals = np.sort(df["lambda"].unique())

    grid = np.full((len(lam_vals), len(xi_vals)), np.nan, dtype=float)

    dsol = df[(df["scalarized"] == 1) & np.isfinite(df["mode_n"])]
    grouped = dsol.groupby(["lambda", "xi"])["mode_n"].min()

    xi_index = {x: i for i, x in enumerate(xi_vals)}
    la_index = {a: i for i, a in enumerate(lam_vals)}
    for (lam, xi), nmin in grouped.items():
        if lam in la_index and xi in xi_index:
            grid[la_index[lam], xi_index[xi]] = float(nmin)

    return xi_vals, lam_vals, grid


# ---------- main ----------
def plot_phase_filled(
    path,
    xi_vals=None,
    lam_vals=None,
    vmax_n=None,
    figsize=(8.5, 6.5),
    out_path=None,
    nu=0.0,
    show_scatter=False,
    natural_units=False,
    combo_colors=None,
):
    if isinstance(path, (str, os.PathLike)):
        files = glob(os.path.join(path, "*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {path}")
    else:
        files = list(path)

        # normalize user palette keys and avoid mutating the caller's dict
    if combo_colors is None:
        combo_colors = {}

    def _norm_combo(tpl):
        # sort tags canonically (e.g., ("1-","0+") -> ("0+","1-"))
        return tuple(sorted(tpl, key=_tag_sort_key))

    palette = { _norm_combo(k): v for k, v in combo_colors.items() }


    df = _load(files)
    xi_vals, lam_vals, grid = _grid_min_mode(df, xi_vals, lam_vals)

    nmax = int(np.nanmax(grid)) if np.isfinite(np.nanmax(grid)) else 0
    if vmax_n is not None:
        nmax = min(nmax, int(vmax_n))

    fig, ax = plt.subplots(figsize=figsize)
    
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    X, Y = np.meshgrid(xi_vals, lam_vals)

    dsol = df[(df["scalarized"] == 1) & np.isfinite(df["mode_n"])]
    xi_index = {x: i for i, x in enumerate(xi_vals)}
    la_index = {a: i for i, a in enumerate(lam_vals)}

    # --- choose signed tags only if nu != 0 and we actually have non-NaN vacuum_sign ---
    use_signed = (nu != 0.0) and ("vacuum_sign" in df.columns and df["vacuum_sign"].notna().any())
    
    if use_signed:
        dsol_v = dsol[np.isfinite(dsol["vacuum_sign"])]
        tags = sorted(
            {_tag_from_row(n, s) for n, s in zip(dsol_v["mode_n"], dsol_v["vacuum_sign"])},
            key=_tag_sort_key
        )
        tag_to_idx = {t: k for k, t in enumerate(tags)}
        present = np.zeros((len(tags), len(lam_vals), len(xi_vals)), dtype=bool)
    
        for lam, xi, n, s in zip(dsol_v["lambda"], dsol_v["xi"], dsol_v["mode_n"], dsol_v["vacuum_sign"]):
            lam_key = lam 
            i = la_index.get(lam_key); j = xi_index.get(xi)
            if i is None or j is None:
                continue
            t = _tag_from_row(n, s)
            present[tag_to_idx[t], i, j] = True
    
        key_labels = tags
    else:
        present = np.zeros((nmax + 1, len(lam_vals), len(xi_vals)), dtype=bool)
        for lam, xi, n in zip(dsol["lambda"], dsol["xi"], dsol["mode_n"]):
            lam_key = lam 
            i = la_index.get(lam_key); j = xi_index.get(xi)

            if i is None or j is None:
                continue
            n_int = int(n)
            if 0 <= n_int <= nmax:
                present[n_int, i, j] = True
        key_labels = [str(k) for k in range(present.shape[0])]

        # ---- exact combo per cell (tuples of tags) ----
        # ---- exact combo per cell (tuples of tags) ----
    H, W = len(lam_vals), len(xi_vals)
    combo_grid = [
        [tuple(key_labels[k] for k in range(present.shape[0]) if present[k, i, j])
         for j in range(W)]
        for i in range(H)
    ]

    # collapse any combo containing n>=3 into a single sentinel ('HIGH',)
    combo_grid = [[_collapse_high(combo_grid[i][j]) for j in range(W)] for i in range(H)]
    observed_combos = set(combo_grid[i][j] for i in range(H) for j in range(W))

    # normalize palette keys and ensure a color for the sentinel
    def _norm_combo(tpl):
        # sort signed tags canonically; keep ('HIGH',) as is
        if tpl == ('HIGH',):
            return tpl
        return tuple(sorted(tpl, key=lambda t: (999, 0) if t == 'HIGH'
                            else (_tag_sort_key(t) if (isinstance(t, str) and (t.endswith('+') or t.endswith('-')))
                                  else (int(t), 0))))

    palette = { _norm_combo(k): v for k, v in (combo_colors or {}).items() }
    palette.setdefault(('HIGH',), '#000000')  # black for higher modes
    
    def _try_signed_color(key_tuple):
    # if all unsigned like ('0','1'), try ('0+','1+') then ('0-','1-')
        if all(isinstance(t, str) and not (t.endswith('+') or t.endswith('-')) and t != 'HIGH'
           for t in key_tuple):
               k_plus  = tuple(f"{t}+" for t in key_tuple)
               k_minus = tuple(f"{t}-" for t in key_tuple)
               if k_plus in palette:  return palette[k_plus]
               if k_minus in palette: return palette[k_minus]
        return None


    # fill any missing observed (excluding empty) with a fallback (won't affect NaN empties)
    for key in observed_combos:
        nk = _norm_combo(key)
        if len(nk) == 0:
            continue
        if nk not in palette:
            col = _try_signed_color(nk)
            palette[nk] = (col if col is not None else "#cccccc")


    # order only non-empty keys (so empties can stay NaN -> white background)
    def _sort_key_tuple(tpl):
        def _key(t):
            if t == 'HIGH':
                return (999, 0)
            return (_tag_sort_key(t) if (isinstance(t, str) and (t.endswith('+') or t.endswith('-')))
                    else (int(t), 0))
        return (len(tpl), tuple(_key(t) for t in tpl))

    ordered_keys = sorted((k for k in palette.keys() if len(k) > 0), key=_sort_key_tuple)
    key_to_int = {k: i for i, k in enumerate(ordered_keys)}

    # build integer grid; EMPTY -> NaN so it renders as white
    int_grid = np.full((H, W), np.nan, dtype=float)
    for i in range(H):
        for j in range(W):
            nk = _norm_combo(combo_grid[i][j])
            if len(nk) == 0:
                int_grid[i, j] = np.nan
            else:
                int_grid[i, j] = key_to_int[nk]

    # colormap for non-empty classes; NaNs will use white
    cmap = ListedColormap([palette[k] for k in ordered_keys])
    cmap.set_bad(color="white")
    levels = np.arange(-0.5, len(ordered_keys) + 0.5, 1.0)
    norm = BoundaryNorm(levels, cmap.N)

    # draw
    ax.pcolormesh(
        X, Y, np.ma.masked_invalid(int_grid),
        cmap=cmap, norm=norm,
        shading="nearest", edgecolors="None", linewidth=0
    )



    if show_scatter:
        ax.scatter(df["xi"], df["lambda"], c="k", s=6, alpha=0.25,
                   linewidths=0, zorder=1)

    ax.set_xlim(float(np.min(xi_vals)), float(np.max(xi_vals)))
    lam_min_ = float(np.nanmin(lam_vals)); lam_max_ = float(np.nanmax(lam_vals))
    pos = np.asarray(lam_vals)[np.asarray(lam_vals) > 0.0]
    linthresh = max(float(np.min(pos)) * 0.5, 1e-70) if pos.size else 1e-70
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    ax.set_ylim(lam_min_, lam_max_)
    
    pos = np.asarray(lam_vals)[np.asarray(lam_vals) > 0.0]
    linthresh = max(float(np.min(pos)) * 0.5, 1e-70) if pos.size else 1e-70
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    
    yticks = sorted(set(float(x) for x in pos))
    if np.any(np.asarray(lam_vals) == 0.0):
        yticks = [0.0] + yticks
    
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_minor_locator(NullLocator())
    
    if natural_units:
        ax.yaxis.set_major_formatter(FuncFormatter(sci_ticks))  # uses hbar*c inside
        ax.yaxis.get_offset_text().set_visible(False)
    else:
        ax.yaxis.set_major_formatter(_lambda_tick_formatter(M_CONST, False, decimals=1))

       

    xmin, xmax = float(np.min(xi_vals)), float(np.max(xi_vals))
    step = abs(max(xmin, xmax)) * 0.1
    xticks = np.arange(xmin, xmax + 0.5 * step, step)
    xticks = np.round(xticks, 12)
    xticks[np.isclose(xticks, 0.0, atol=1e-12)] = 0.0
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, i: f"{v:g}" if (i is None or i % 2 == 0) else ""))

    ax.set_xlabel(r"$\xi$")
    if natural_units:
       ax.set_ylabel(r"$\lambda$")
    else:
       ax.set_ylabel(r"$\lambda\,[\mathrm{s^2\,kg^{-1}\,m^{-3}}]$")
    ax.grid(ls=":", lw=0.6, color="k", alpha=0.25, zorder=0)

    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_linewidth(1.1)
    ax.tick_params(axis="x", which="both", top=True, labeltop=False, direction="in", length=4, width=0.9)
    ax.tick_params(axis="y", which="both", right=True, labelright=False, direction="in", length=4, width=0.9)

    def _lab(tpl):
        if tpl == ('HIGH',):
            return "Higher modes"
        if len(tpl) == 1:
            t = tpl[0]
            return (rf"${t[:-1]}^{{{t[-1]}}}$" if (isinstance(t, str) and t[-1] in "+-") else rf"$n={t}$")
        return r"$" + " + ".join(
            (rf"{t[:-1]}^{{{t[-1]}}}" if (isinstance(t, str) and t[-1] in "+-") else rf"{t}")
            for t in tpl
        ) + r"$"


        # Legend: only combos actually present in the plotted grid
    present_keys = set()
    for i in range(H):
        for j in range(W):
            nk = _norm_combo(combo_grid[i][j])
            if len(nk) > 0:
                present_keys.add(nk)

    legend_handles, legend_labels = [], []
    for k in ordered_keys:
        if k not in present_keys:
            continue
        legend_handles.append(Patch(facecolor=palette[k], edgecolor="none"))
        legend_labels.append(_lab(k))

    if legend_handles:
        ncol = min(len(legend_handles), 4)  # compact, up to 4 columns
        ax.legend(
            legend_handles,
            legend_labels,
            ncol=ncol,
            loc ="upper center",
            #bbox_to_anchor=(0.275, 0.975),
            fontsize=10.5,
            frameon=True,
            framealpha=0.95,
            fancybox=True,
            edgecolor="black",
            title="Present modes",
            title_fontsize=11,
            columnspacing=0.9,
            handletextpad=0.4,
            handlelength=1.4,
            borderpad=0.6
        ) 
    ax.grid(False)
    if nu == 0.0:
        plt.title(r"$v = 0$")
    else:    
       plt.title(r"$v = M_{Pl} \cdot 10^{-5}$")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")  
    plt.show()
    return fig, ax


if __name__ == "__main__":
    sw = 'L'
    nu = 2.e+16
    custom = {
        ("0+",): "#f4828f",
        ("0-",): "#f0a3ab",
        ("1+",): "#9305FF",
        ("1-",): "#a185d3",
        ("2+",): "#1e988a",
        ("2-",): "#5fb4a9",

        ("0+", "0-"): "#d12136",
        ("0+", "1+"): "#8658C2",
        ("0-", "1+"): "#EDB600",
        ("0+", "1-"): "#DF3392",
        ("0-", "1-"): "#43034d",
        ("0+", "0-", "1+"): "#1D440E",
        ("1-", "2+"): "#07ab51",
        ("0+", "2-"): "#6DC217",

        ("0+", "1+", "2+"): "#0A406D",
        ("0+", "1-", "2+"): "#160A6D",
        ("0+", "0-", "1-"): "#A50187",
        ("0+", "1+", "1-"): "#1C87DF",

        tuple(): "#F3F3F4",
    }

    plot_phase_filled(
        f"Results/scan/{sw}/nu={nu:.0e}_ZOOM",
        out_path=f"Results/scan/{sw}/nu={nu:.0e}_ZOOM/scan.png",
        nu=nu,
        show_scatter=False,
        natural_units=True,
        combo_colors=custom,
    )