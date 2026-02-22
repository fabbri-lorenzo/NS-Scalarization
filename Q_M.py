# q_vs_lambda_at_xi.py
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator, ScalarFormatter, LogFormatterMathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from Utils.params import M as M_CONST , hbar, c


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

def sci10_ticks_symlog(x, _pos):
    if x == 0:
        return r"$0$"
    x_scaled = x * (hbar * c)  # relabel-only scaling
    exp = int(np.floor(np.log10(abs(x_scaled))))
    return rf"$10^{{{exp}}}$"

def sci_a10_ticks(y, _pos):
    if y == 0:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(y))))
    mant = y / (10 ** exp)
    # keep a compact mantissa (avoid trailing zeros); 3 sig figs is usually plenty
    return rf"${mant:.3g}\times 10^{{{exp}}}$"


# ---------- IO ----------
_NUMERIC_COLS = (
    "xi", "lambda", "nu", "scalarized", "mode_n",
    "ADM_over_Msun", "Q_over_Msun", "Q_over_ADM"
)

def _load_csvs(path_or_list):
    files = (glob(os.path.join(path_or_list, "*.csv"))
             if isinstance(path_or_list, (str, os.PathLike))
             else list(path_or_list))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path_or_list!r}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
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
        raise ValueError("Need Q_over_ADM or (Q_over_Msun & ADM_over_Msun) to compute Q/M.")
    return df


# ---------- plotting ----------
def plot_q_over_m_vs_lambda_at_xi(
    path,
    *,
    xi_sel,                 # required
    zoom = False,
    natural_units=False,
    xi_tol=1e-12,
    rho0_tag=None,
    scalarized_only=True,
    include_modes=None,     # e.g. (0,1,2)
    figsize=(7.4, 4.4), dpi=150,
    out_path=None,
):
    df = _load_csvs(path)

    # --- filter slice (fixed xi) ---
    m = np.isfinite(df["xi"]) & np.isfinite(df["Q_over_M"]) & np.isfinite(df["lambda"])
    if scalarized_only and "scalarized" in df:
        m &= (df["scalarized"] == 1)
    if rho0_tag is not None and "rho0_tag" in df:
        m &= (df["rho0_tag"].astype(str) == str(rho0_tag))
    # xi selection (tolerant)
    m &= np.isclose(df["xi"], float(xi_sel), rtol=0.0, atol=xi_tol)

    d = df[m].copy()
    if d.empty:
        raise ValueError("No rows found for the requested slice.")

    # --- group duplicates: one value per (mode, lambda) ---
    if "mode_n" not in d:
        raise ValueError("Column 'mode_n' is required.")
    d = (d
         .dropna(subset=["mode_n", "lambda", "Q_over_M"])
         .assign(mode_n=d["mode_n"].astype(int))
         .groupby(["mode_n", "lambda"], as_index=False)["Q_over_M"]
         .median())

    modes = sorted(d["mode_n"].unique())
    if include_modes is not None:
        modes = [n for n in modes if n in set(include_modes)]
        
    d["lambda_plot"] = d["lambda"]


    # --- prepare axis ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    # symlog x with custom ticks at sampled lambdas
    lam_vals = np.sort(d["lambda_plot"].unique())
    pos = lam_vals[lam_vals > 0]
    linthresh = max((pos.min() * 0.5) if pos.size else 1e-70, 1e-70)
    ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
    ax.set_xlim(lam_vals.min(), lam_vals.max())
    
    xticks = lam_vals.tolist()
    if 0.0 not in xticks and np.any(lam_vals == 0.0):
        xticks = [0.0] + xticks
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(
        FuncFormatter(sci10_ticks_symlog) if natural_units
        else _lambda_tick_formatter(M_CONST, decimals=1)
    )

    # --- styles: discrete markers, no lines ---
    mode_colors = {0: "#f4828f", 1: "#8658C2", 2: "#1e988a"}  # per-mode colors

    def _mk(marker, n):
        c = mode_colors.get(n, "black")
        return dict(marker=marker, mfc=c, mec=c, mew=0.8, ms=6, ls="none")    

    style_map = {
        0: _mk("o", 0),
        1: _mk("s", 1),
        2: _mk("^", 2),
        3: _mk("D", 3),  # fallback color for unseen modes -> black
        4: _mk("v", 4),
    }

    # --- plot: scatter per node ---
    handles = []
    labels = []
    for n in modes:
        dn = d[d["mode_n"] == n].sort_values("lambda")
        if dn.empty: 
            continue
        st = style_map.get(n, dict(marker="o", mfc="none", mec="black", mew=1.2, ms=5, ls="none"))
        h = ax.plot(dn["lambda_plot"].to_numpy(), dn["Q_over_M"].to_numpy(), **st)[0]
        handles.append(h)
        labels.append(rf"$n={n}$")
        
    # --- horizontal bounds (±6×10⁻⁴) ---
    ax.axhline(6e-4, color="green", linestyle="--", linewidth=1.0)
    ax.axhline(-6e-4, color="green", linestyle="--", linewidth=1.0)
    
    if zoom:
            # inset zoom below the legend
        def _ticks_from_p(pmin, pmax, M=M_CONST, step=0.1):
           # exact p grid with 1-decimal steps -> second decimal = 0 by construction
           P = np.round(np.arange(np.ceil(pmin*10)/10, np.floor(pmax*10)/10 + 1e-9, step), 1)
           X = M ** (-P)  # exact λ positions for those exponents
           return X, P  
         
        iax = inset_axes(
        ax,
        width="100%", height="100%", loc="center left",
        bbox_to_anchor=(0.60, 0.40, 0.36, 0.30),  # (x0, y0, w, h) in axes fraction
        bbox_transform=ax.transAxes,
        borderpad=0.0,
                      )
        iax.tick_params(right=False)
    
        for n in modes:
            dn = d[d["mode_n"] == n].sort_values("lambda")
            if dn.empty:
                continue
            st = style_map.get(n, dict(marker="o", mfc="none", mec="black", mew=1.2, ms=5, ls="none"))
            iax.plot(dn["lambda_plot"].to_numpy(), dn["Q_over_M"].to_numpy(), **st)
    
        iax.set_ylim(-10e-4, 10e-4)
        
        iax.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
    
        iax.axhline(6e-4, color="green", linestyle="--", linewidth=1.0)
        iax.axhline(-6e-4, color="green", linestyle="--", linewidth=1.0)
        iax.set_yticks([-6e-4, 6e-4])
        iax.set_yticklabels([r"$-6\times10^{-4}$", r"$6\times10^{-4}$"])
        iax.tick_params(axis="y", which="both", direction="in", labelsize=8)
        iax.yaxis.set_major_formatter(FuncFormatter(sci_a10_ticks))
        iax.yaxis.get_offset_text().set_visible(False)

    
        iax.set_ylim(-10e-4, 10e-4)
        dd = d[np.abs(d["Q_over_M"]) <= 10e-4]
        if not dd.empty:
            xmin, xmax = dd["lambda_plot"].min(), dd["lambda_plot"].max()
            iax.set_xlim(xmin, xmax)
            iax.set_xscale("log")  # inset has no zero
        
            iax.xaxis.set_minor_locator(NullLocator())
            if natural_units:
                # Ticks EXACTLY at the sampled λ’s shown in the inset (so they align with points).
                xs = np.sort(dd["lambda_plot"].unique())
                xs = xs[(xs >= xmin) & (xs <= xmax)]
                # Limit to ~10 ticks (auto thinning)
                if xs.size > 10:
                    stride = int(np.ceil(xs.size / 10.0))
                    xs = xs[::stride]
                iax.xaxis.set_major_locator(FixedLocator(xs.tolist()))
                iax.xaxis.set_major_formatter(FuncFormatter(sci10_ticks_symlog))  # labels show 10^exp of (λ·ħc)
            else:
                # SI: ticks at exact λ = M^{-p}
                pmin_i = _p_from_lambda(xmax / 1.0)
                pmax_i = _p_from_lambda(xmin / 1.0)
                xticks_in, _ = _ticks_from_p(min(pmin_i, pmax_i), max(pmin_i, pmax_i), M_CONST, step=0.1)
                xticks_in = [t for t in xticks_in if xmin <= t <= xmax]
                iax.xaxis.set_major_locator(FixedLocator(xticks_in))
                iax.xaxis.set_major_formatter(_lambda_tick_formatter(M_CONST, decimals=1))
            
        for s in ("top", "right", "bottom", "left"):
            iax.spines[s].set_linewidth(0.8)
    

    # --- labels & cosmetics ---
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$Q/\mathcal{M}$")
    # inside plot_q_over_m_vs_lambda_at_xi(...), after ax.set_ylabel(...)
    ax.yaxis.set_major_formatter(FuncFormatter(sci_a10_ticks))
    ax.yaxis.get_offset_text().set_visible(False)


    ax.grid(False)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_linewidth(1.0)

    # title centered at top
    ax.set_title(rf"$\xi={xi_sel:g}$", pad=6)    

    if handles:
        ax.legend(
            handles, labels,
            ncol=1,                       # one column
            loc="upper right",            # inside, top-right
            bbox_to_anchor=(0.98, 0.98),  # slight inset
            frameon=True,
            handlelength=1.0, columnspacing=0.8
        )

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.show()    
    return fig, ax


if __name__ == "__main__":
    sw = 'L'
    plot_q_over_m_vs_lambda_at_xi(
        path=f"Results/scan/{sw}/nu=2e+16_ZOOM",
        xi_sel=1,
        zoom=False,
        natural_units=True,
        rho0_tag=f"{sw}",
        include_modes=(0, 1, 2, 3),
        out_path=f"Results/scan/{sw}/nu=2e+16_ZOOM/Q_M.png",
    )
