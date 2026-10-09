"""Plotting helpers for single-star solutions."""

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    AutoMinorLocator,
    FuncFormatter,
    MultipleLocator,
    SymmetricalLogLocator,
)
import numpy as np
from Utils.params import M

R_star_color = "c"
dpi_val = 600

#param_colors_base = [
#    "#1F4C25",
#    "#78b41f",
#    "#008035",
#    "#1F4C25",
#]
param_colors_base = [
   "#9847e9",
   "#E370F2",
   "#ac0081",
   "#9847e9",
]
# mode line styles
mode_linestyles = ["-", "--", ":", "-."]
single_curve_color = param_colors_base[0]


def clean_x(val, pos=None):
    if np.isclose(val, round(val), atol=1e-10):
        return str(int(round(val)))
    return f"{val:g}"


def even_decade_only(val, pos=None):
    if val == 0:
        return "0"
    sign = "-" if val < 0 else ""
    v = abs(val)

    exp = np.log10(v)
    if not np.isfinite(exp):
        return ""

    e = int(np.round(exp))
    if not np.isclose(v, 10**e, rtol=0, atol=v * 1e-12):
        return ""
    if (e % 2) != 0:
        return ""

    return rf"${sign}10^{{{e}}}$"


def _apply_common_axis_style(ax):
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(clean_x))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.tick_params(axis="x", which="major", length=4, width=1.2)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)


def _apply_symlog_y_style(ax, linthresh):
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.yaxis.set_major_locator(
        SymmetricalLogLocator(base=10, linthresh=linthresh, subs=(1.0,))
    )
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(base=10, linthresh=linthresh, subs=np.arange(2, 10) * 0.1)
    )
    ax.yaxis.set_major_formatter(FuncFormatter(even_decade_only))
    ax.tick_params(axis="y", which="major", length=6, width=1)
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)
    ax.margins(y=0.05)


def _mode_handles(entries, nu, color="black"):
    modes_seen = sorted({_extract_mode_and_sign(e, nu)[0] for e in entries})
    return [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=mode_linestyles[n % len(mode_linestyles)],
            linewidth=1.6,
            label=rf"$n={n}$",
        )
        for n in modes_seen
    ]


def _set_mode_legend(entries, nu, *, loc="best", bbox_to_anchor=None):
    handles = _mode_handles(entries, nu)
    _set_legend(handles, loc=loc, bbox_to_anchor=bbox_to_anchor)


def _set_legend(handles, *, loc="best", bbox_to_anchor=None):
    if not handles:
        return

    unique_handles = []
    seen_labels = set()

    for h in handles:
        lbl = h.get_label()
        # Filter out duplicates and internal Matplotlib unlabeled items (which start with '_')
        if lbl and not lbl.startswith("_") and lbl not in seen_labels:
            seen_labels.add(lbl)
            unique_handles.append(h)

    plt.legend(
        unique_handles,
        [h.get_label() for h in unique_handles],
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
    )


def nu_line(nu, vacuum_sols):
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    if nu != 0.0:
        if vacuum_sols["+"] > 0:
            plt.axhline(
                y=nu / M,
                color="pink",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=r"$v/M_{Pl}$",
            )
        if vacuum_sols["-"] > 0:
            plt.axhline(
                y=-nu / M,
                color="magenta",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=r"$-v/M_{Pl}$",
            )


def _extract_mode_and_sign(entry, nu):
    # n from "n=..." label; default 0
    try:
        n = int(entry["label"].split("=")[1])
    except Exception:
        n = 0
    vac_sign = entry.get("vacuum_sign", 0)
    return n, vac_sign


def _signed_label(base_math, n, vac_sign, nu):
    # base_math is a raw math string, e.g. r"\mu_{\rm eff}^2" or r"\sigma/M_{Pl}"
    if nu == 0.0 or vac_sign == 0:
        return rf"${base_math} \, [n={n}]$"
    s = "+" if vac_sign > 0 else "-"
    return rf"${base_math} \, [n={n}^{{{s}}}]$"


def _math_label(base_math):
    label = str(base_math)
    if label.startswith("$") and label.endswith("$"):
        return label
    return rf"${label}$"


def _resample_sol_component(sol, idx, r_plot):
    if hasattr(sol, "sol") and callable(sol.sol):
        return sol.sol(r_plot)[idx]
    return np.interp(r_plot, sol.t, sol.y[idx])


def _mode_entries(entries, nu):
    return sorted(entries, key=lambda e: _extract_mode_and_sign(e, nu))


def _radius_km(r_m):
    return np.asarray(r_m, dtype=float) / 1e3


def _max_radius_km(entries):
    xmax = 0.0
    for entry in entries:
        sol = entry["sol"]
        xmax = max(xmax, float(sol.t[-1]) / 1e3)
    return xmax


def _format_full_radius_axis(entries, *, xmin=0.0):
    xmax = _max_radius_km(entries)
    xmin = min(float(xmin), xmax) if xmax > 0.0 else 0.0
    plt.xlim(left=xmin, right=xmax)
    plt.xlabel("r [km]")
    ax = plt.gca()
    _apply_common_axis_style(ax)
    if xmax > 5.0:
        major_step = 5.0 if xmax <= 50.0 else 50.0 if xmax <= 500.0 else 100.0
        ax.xaxis.set_major_locator(MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction="in", which="both", top=True, right=True)


def _rescale_y_to_visible_x(ax, *, pad_frac=0.05):
    xmin, xmax = ax.get_xlim()
    visible_y = []

    for line in ax.lines:
        # Skip backgrounds, vlines, hlines that use blended axis transforms
        if line.get_transform() != ax.transData:
            continue

        # --- CRITICAL FIX: IGNORE ANALYTICAL BLOWUP ---
        # Only compute y-limits from lines matching your core numerical color.
        # This completely ignores your blue analytical curves for scaling purposes.
        if line.get_color() != single_curve_color:
            continue

        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.shape != y.shape:
            continue

        mask = (x >= xmin) & (x <= xmax) & np.isfinite(y)
        if np.any(mask):
            visible_y.append(y[mask])

    if not visible_y:
        return

    y_all = np.concatenate(visible_y)
    ymin = float(np.nanmin(y_all))
    ymax = float(np.nanmax(y_all))

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return
    if np.isclose(ymin, ymax):
        delta = abs(ymin) * pad_frac if ymin != 0.0 else pad_frac
        ax.set_ylim(ymin - delta, ymax + delta)
        return

    pad = (ymax - ymin) * pad_frac
    ax.set_ylim(ymin - pad, ymax + pad)


def _remove_stale_sigma_infty(savepath):
    stale_path = os.path.join(savepath, "sigma_infty.png")
    if os.path.exists(stale_path):
        os.remove(stale_path)


def _latex_sci(value):
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    if np.isclose(mantissa, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:g}\times 10^{{{exponent}}}"

from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def _add_zoom_insets(
    ax,
    zoom_regions,
    *,
    max_windows=3,
    inset_size=("70%", "70%"),
    pad_x_frac=0.08,
    pad_y_frac=0.15,
    locs=None,
):
    """
    Draw zoom inset windows using manually specified coordinates.
    Automatically extracts and crops data curves from the main plot axes.
    """
    if not zoom_regions:
        return []

    if locs is None:
        locs = [
            dict(bbox_to_anchor=(0.1, 0.2, 0.25, 0.25)),
            dict(bbox_to_anchor=(0.4, 0.2, 0.25, 0.25)),
        ]

    insets = []

    # Process up to your max allowed window count
    for k, region in enumerate(zoom_regions[:max_windows]):
        if k >= len(locs):
            break

        # 1. Create the blank inset axes canvas
        axins = inset_axes(
            ax,
            width=inset_size[0],
            height=inset_size[1],
            loc="upper left",
            bbox_transform=ax.transAxes,
            borderpad=0.8,
            **locs[k],
        )

        # 2. Extract your manual limits and calculate padded boundaries
        x0, x1 = region["x_min"], region["x_max"]
        y0, y1 = region["y_min"], region["y_max"]

        dx = max(x1 - x0, 1e-30)
        dy = max(y1 - y0, 1e-30)

        xlim_min, xlim_max = x0 - pad_x_frac * dx, x1 + pad_x_frac * dx
        ylim_min, ylim_max = y0 - pad_y_frac * dy, y1 + pad_y_frac * dy

        # 3. Automatically find and clone curves from the main axes
        for line in ax.get_lines():
            # Force conversion to NumPy arrays in case they are standard Python lists
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            
            # Mask data to only plot what's inside the zoom window
            mask = (xdata >= xlim_min) & (xdata <= xlim_max) & np.isfinite(xdata) & np.isfinite(ydata)
            
            if np.count_nonzero(mask) >= 2:
                axins.plot(
                    xdata[mask],
                    ydata[mask],
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                    alpha=line.get_alpha()
                )

        # 4. Apply window limits and clean up styling
        axins.set_xlim(xlim_min, xlim_max)
        axins.set_ylim(ylim_min, ylim_max)
        axins.tick_params(direction="in", which="both", top=True, right=True, labelsize=8)
        def clean_exponential(x, pos):
            if x == 0:
                return "0"
            return f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e+")
        
        axins.yaxis.set_major_formatter(FuncFormatter(clean_exponential))
        axins.grid(False)

        # 5. Draw the connector box on the main plot
        pp, p1, p2 = mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")
        p1.set_visible(False)
        p2.set_visible(True)

        insets.append(axins)

    return insets

def _plot_extra_functions(
    plot_name,
    extra_functions,
    *,
    entries,
    xi,
    lmbda,
    m,
    nu,
    vacuum_sols,
    label_modes=True,
):
    """
    Overlay user-supplied functions on a figure.

    Each function receives r in km and must return y-values in the same
    units as the active plot:
      pressure -> Pa, mu2 -> km^-2, sigma -> sigma/M_Pl.
    """
    if not extra_functions:
        return []

    import re  # Added for regex string scanning fallback

    handles = []
    x_default = np.linspace(0.0, _max_radius_km(entries), 4000)
    context = {
        "entries": entries,
        "xi": xi,
        "lmbda": lmbda,
        "m": m,
        "nu": nu,
        "vacuum_sols": vacuum_sols,
    }

    for i, cfg in enumerate(extra_functions):
        if cfg.get("plot", "sigma") != plot_name:
            continue

        func = cfg.get("function")
        if func is None:
            raise ValueError(f"Missing function for extra {plot_name} curve #{i}.")

        entry_list = (
            _mode_entries(entries, nu) if cfg.get("per_entry", False) else [None]
        )
        for entry in entry_list:
            if entry is not None and "x_km" not in cfg:
                sol = entry["sol"]
                x_km = _radius_km(np.linspace(sol.t[0], sol.t[-1], 4000))
            else:
                x_km = np.asarray(cfg.get("x_km", x_default), dtype=float)

            try:
                y = func(x_km, entry=entry, **context)
            except TypeError:
                try:
                    y = func(x_km, **context)
                except TypeError:
                    y = func(x_km)
            y = np.asarray(y, dtype=float)
            if y.ndim == 0:
                y = np.full_like(x_km, float(y))

            # --- SMART TRAIT DETECTION ---
            n_val = None
            vac_sign = 0

            if entry is not None:
                # Scenario A: You specified "per_entry": True
                n_val, vac_sign = _extract_mode_and_sign(entry, nu)
            else:
                # Scenario B: Standalone dictionary entries.
                # Check for explicit 'n' or 'mode' keys in your config dictionary first
                if "n" in cfg:
                    n_val = int(cfg["n"])
                elif "mode" in cfg:
                    n_val = int(cfg["mode"])
                else:
                    # Check if you typed the mode inside your text label (e.g. "analytical n=1")
                    label_text = str(cfg.get("label", ""))
                    match = re.search(
                        r"(?:n|mode)\s*=\s*(\d+)|(?:n|mode)\s+(\d+)",
                        label_text,
                        re.IGNORECASE,
                    )
                    if match:
                        n_val = int(match.group(1) or match.group(2))

            # Match the resolved mode integer to its corresponding linestyle
            if n_val is not None:
                mode_ls = mode_linestyles[n_val % len(mode_linestyles)]
            else:
                mode_ls = "-"

            # Format label
            label = cfg.get("label", f"extra {i + 1}")
            if entry is not None and cfg.get("label_modes", label_modes):
                label = _signed_label(label, n_val, vac_sign, nu)
            else:
                label = _math_label(label)

            (line,) = plt.plot(
                x_km,
                y,
                label=label,
                color=cfg.get("color"),
                linestyle=cfg.get(
                    "linestyle", mode_ls
                ),  # Falls back to our smart style!
                linewidth=cfg.get("linewidth", 1.6),
                alpha=cfg.get("alpha", 1.0),
            )
            handles.append(line)

    return handles


def _has_extra_functions(extra_functions, plot_name):
    return any(cfg.get("plot", "sigma") == plot_name for cfg in extra_functions or [])


def _numerical_handle():
    return Line2D(
        [0],
        [0],
        color=single_curve_color,
        linestyle="-",
        linewidth=1.6,
        label="Numerical",
    )


def plotResults_multi(
    entries,
    xi,
    lmbda,
    m,
    nu,
    vacuum_sols,
    savepath,
    zoom_regions,
    star_label = None,
    extra_functions=None,
):
    """Render pressure, μ_eff² and σ overlays for one star and many modes."""
    os.makedirs(savepath, exist_ok=True)
    has_sigma_extras = _has_extra_functions(extra_functions, "sigma")

    # --- Pressure profile ---
    plt.figure()

    for e in _mode_entries(entries, nu):
        sol = e["sol"]
        r_plot = np.linspace(sol.t[0], sol.t[-1], 4000)
        p_plot = _resample_sol_component(sol, idx=0, r_plot=r_plot)

        n_val, vac_sign = _extract_mode_and_sign(e, nu)
        ls = mode_linestyles[n_val % len(mode_linestyles)]
        x_km = _radius_km(r_plot)
        plt.plot(
            x_km,
            p_plot,
            color=single_curve_color,
            linestyle=ls,
            linewidth=1.6,
        )

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    extra_handles = _plot_extra_functions(
        "pressure",
        extra_functions,
        entries=entries,
        xi=xi,
        lmbda=lmbda,
        m=m,
        nu=nu,
        vacuum_sols=vacuum_sols,
    )
    _format_full_radius_axis(entries)
    plt.ylabel("P [Pa]")
    plt.grid(False)
    _set_legend(_mode_handles(entries, nu) + extra_handles, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "pressure.png"), dpi=dpi_val)
    plt.close()
    # plt.show()

    # --- Effective mass squared profile (uniform resample of logged points) ---
    plt.figure()
    for e in _mode_entries(entries, nu):
        r_mu, mu2 = e["r_mu"], e["mu2"]
        if r_mu.size == 0:
            continue

        # ensure strictly increasing x for interpolation
        r_mu_m = np.asarray(r_mu, dtype=float)
        order = np.argsort(r_mu_m, kind="mergesort")
        r_mu_m = r_mu_m[order]
        mu2_m = mu2[order]

        unique_x, unique_idx = np.unique(r_mu_m, return_index=True)
        r_mu_m = unique_x
        mu2_m = mu2_m[unique_idx]

        if r_mu_m.size < 2:
            continue

        r_plot = np.linspace(r_mu_m[0], r_mu_m[-1], 4000)
        mu2_plot_km = np.interp(r_plot, r_mu_m, mu2_m) * 1e6  # m^-2 -> km^-2

        n_val, vac_sign = _extract_mode_and_sign(e, nu)
        ls = mode_linestyles[n_val % len(mode_linestyles)]
        x_km = _radius_km(r_plot)
        plt.plot(
            x_km,
            mu2_plot_km,
            color=single_curve_color,
            linestyle=ls,
            linewidth=1.6,
        )

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    extra_handles = _plot_extra_functions(
        "mu2",
        extra_functions,
        entries=entries,
        xi=xi,
        lmbda=lmbda,
        m=m,
        nu=nu,
        vacuum_sols=vacuum_sols,
    )
    _format_full_radius_axis(entries)
    plt.ylabel(r"$\mu_{\rm eff}^2$ [km$^{-2}$]")
    ax = plt.gca()
    _apply_symlog_y_style(ax, linthresh=1e-12)
    plt.grid(False)
    _set_legend(_mode_handles(entries, nu) + extra_handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "mu2.png"), dpi=dpi_val, bbox_inches="tight")
    plt.close()
    # plt.show()

    # --- Scalar field σ(r) ---
    plt.figure()
    for e in _mode_entries(entries, nu):
        sol = e["sol"]
        r_plot = np.linspace(sol.t[0], sol.t[-1], 4000)
        sigma_plot = _resample_sol_component(sol, idx=2, r_plot=r_plot)  # σ(r)

        n_val, vac_sign = _extract_mode_and_sign(e, nu)
        ls = mode_linestyles[n_val % len(mode_linestyles)]
        x_km = _radius_km(r_plot)
        plt.plot(
            x_km,
            sigma_plot / M,
            color=single_curve_color,
            linestyle=ls,
            linewidth=1.6,
        )

    nu_line(nu, vacuum_sols)
    extra_handles = _plot_extra_functions(
        "sigma",
        extra_functions,
        entries=entries,
        xi=xi,
        lmbda=lmbda,
        m=m,
        nu=nu,
        vacuum_sols=vacuum_sols,
        label_modes=not has_sigma_extras,
    )
    _format_full_radius_axis(entries, xmin=150 if has_sigma_extras else 0.0)
        
    # --- ZOOM CONFIGURATION ---
    zoom_max_windows = len(zoom_regions) 
    zoom_pad_x_frac = 0.1
    zoom_pad_y_frac = 0.01
    
    zoom_inset_size = ("70%", "70%") 
    zoom_locs = None

    ax = plt.gca()

    _add_zoom_insets(
        ax,
        zoom_regions,
        max_windows=zoom_max_windows,
        inset_size=zoom_inset_size,
        pad_x_frac=zoom_pad_x_frac,
        pad_y_frac=zoom_pad_y_frac,
        locs=zoom_locs,
    )
    
    # 2. CRITICAL FIX: Set the current active axes back to the main plot!
    plt.sca(ax)
    
    if has_sigma_extras:
        _rescale_y_to_visible_x(plt.gca())
        
    plt.ylabel(r"$\sigma/M_{Pl}$")
    numerical_handles = (
        _mode_handles(entries, nu) + [_numerical_handle()]
        if has_sigma_extras
        else _mode_handles(entries, nu)
    )
    star_handles = []
    if star_label:
        star_handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                color="black",
                markersize=10,
                label=f"{star_label} star",
            )
        )
    _set_legend(
        numerical_handles + extra_handles + star_handles,
        loc="upper right",
        bbox_to_anchor=(1, 0.9),
    )
    plt.grid(False)
    plt.title(
        rf"$\xi={float(xi):g}$, $\lambda={_latex_sci(float(lmbda))}$, $\mu={float(m):g}\,$eV, $v={float(nu):g}\,$eV"
    )
    
    # Save cleanly without tight_layout distorting the insets
    plt.savefig(os.path.join(savepath, "sigma.png"), dpi=dpi_val, bbox_inches="tight")
    _remove_stale_sigma_infty(savepath)
    plt.show()
    plt.close()


# --- Colored printing utility ---
import sys
from colorama import Fore, Style, init

# Initialize colorama (important for Windows, harmless on Mac/Linux)
init(autoreset=True)

def custom_print(text, color="white", style="normal", stream=None):
    colors = {
        "black": Fore.BLACK,
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
        "gray": Fore.LIGHTBLACK_EX,
    }

    styles = {
        "normal": "",
        "bold": Style.BRIGHT,
        "dim": Style.DIM,
    }

    color_code = colors.get(color.lower(), Fore.WHITE)
    style_code = styles.get(style.lower(), "")

    out = stream if stream is not None else sys.stdout
    print(f"{style_code}{color_code}{text}{Style.RESET_ALL}", file=out)
