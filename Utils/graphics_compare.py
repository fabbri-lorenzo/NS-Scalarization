"""Overlay plots comparing solutions at different λ or μ values."""

# Utils/graphics_compare.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.ticker import SymmetricalLogLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from Utils.params import M
from Utils.graphics_single import (
    nu_line,
    _extract_mode_and_sign,
    _signed_label,
    _resample_sol_component,
)

dpi_val = 600

def clean_x(val, pos=None):
    # integers -> no decimals
    if np.isclose(val, round(val), atol=1e-10):
        return str(int(round(val)))
    # others like 0.5 -> keep as is
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

    # label only exact decades (1 * 10^e)
    if not np.isclose(v, 10**e, rtol=0, atol=v * 1e-12):
        return ""

    # label only even exponents
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


def _add_zoom_inset(
    ax,
    plot_func,
    zoom_cfg,
    *,
    xscale=None,
    yscale=None,
    yformatter=None,
):
    """
    Add a zoom inset to an existing axes.

    zoom_cfg example:
    {
        "xlim": (0.0, 0.3),
        "ylim": (-1e-8, 1e-5),
        "loc": "upper left",
        "width": "38%",
        "height": "38%",
        "borderpad": 1.0,
        "mark": True,
        "mark_loc1": 2,
        "mark_loc2": 4,
    }
    """
    if zoom_cfg is None:
        return None

    axins = inset_axes(
        ax,
        width=zoom_cfg.get("width", "38%"),
        height=zoom_cfg.get("height", "38%"),
        loc=zoom_cfg.get("loc", "upper right"),
        borderpad=zoom_cfg.get("borderpad", 1.0),
    )

    plot_func(axins)

    axins.set_xlim(*zoom_cfg["xlim"])
    axins.set_ylim(*zoom_cfg["ylim"])

    if xscale is not None:
        if isinstance(xscale, tuple):
            scale_name, scale_kwargs = xscale
            axins.set_xscale(scale_name, **scale_kwargs)
        else:
            axins.set_xscale(xscale)

    if yscale is not None:
        if isinstance(yscale, tuple):
            scale_name, scale_kwargs = yscale
            axins.set_yscale(scale_name, **scale_kwargs)
        else:
            axins.set_yscale(yscale)

    _apply_common_axis_style(axins)

    if yformatter is not None:
        axins.yaxis.set_major_formatter(FuncFormatter(yformatter))

    axins.tick_params(axis="both", labelsize=8)
    axins.grid(False)

    if zoom_cfg.get("mark", True):
        pp, p1, p2 = mark_inset(
            ax,
            axins,
            loc1=zoom_cfg.get("mark_loc1", 2),
            loc2=zoom_cfg.get("mark_loc2", 4),
            fc="none",
            ec=zoom_cfg.get("mark_ec", "0.4"),
            lw=zoom_cfg.get("mark_lw", 0.8),
        )

        # optional control of connectors
        p1.set_visible(True)
        p2.set_visible(False)

    return axins


def plotResults_compare(
    entries_by_param,
    xi,
    nu,
    vacuum_sols,
    savepath,
    *,
    m=None,
    lmbda=None,
    star_label=None,
    param_labels=None,
    n_resample=4000,
    lw=1.6,
    param_symbol=None,
    param_colors=None,
    param=None,
    title=None,
    pressure_zoom=None,
    mu2_zoom=None,
    sigma_zoom=None,
):
    """
    Overlay plots for multiple parameter values (λ or μ).
    entries_by_param: dict {param_value: entries_list}
      where each entries_list is the same structure you already pass to plotResults_multi.
    Distinguishes parameter values via color, modes via line style.
    param_symbol: LaTeX symbol to use in legends;
    param_colors: optional mapping {param_value: color}; if missing, first/second/third
                  parameter get exact blue/red/green respectively.
    """
    os.makedirs(savepath, exist_ok=True)

    param_values = list(entries_by_param.keys())  # preserve insertion order

    if param_symbol is None:
        param_symbol = r"\lambda" if str(param).lower().startswith("l") else r"\mu"

    # keep backward compatibility with lambda_labels input

    param_labels = {p: rf"${param_symbol}={p:g}$" for p in param_values}

    if param == "lambda":
        param_colors_base = [
            "#78b41f",
            "#008035",
            "#1F4C25",
        ]
    elif param == "mu":
        param_colors_base = [
            "#E370F2",
            "#ac0081",
            "#9847e9",
        ]
        param_labels = {p: rf"${param_symbol}={p:g}\,$eV" for p in param_values}
    else:
        param_colors_base = [
            "#ffd504",
            "#ff7f00",
            "#ff6b6b",
        ]
        param_labels = {p: rf"${param_symbol}={p:g}\,$eV" for p in param_values}

    if param_colors is None:
        param_colors = {
            p: param_colors_base[i % len(param_colors_base)]
            for i, p in enumerate(param_values)
        }
    else:
        param_colors = dict(param_colors)
        next_idx = len(param_colors) % len(param_colors_base)
        for p in param_values:
            if p not in param_colors:
                param_colors[p] = param_colors_base[next_idx % len(param_colors_base)]
                next_idx += 1

    # star color (single star per plot; keep neutral)
    star_color = "black"

    # mode line styles
    mode_linestyles = ["-", "--", ":", "-."]

    # ========== PRESSURE ==========
    plt.figure()
    def _plot_pressure_on_axis(ax_):
        for param_val in param_values:
            entries = entries_by_param[param_val]
            color = param_colors[param_val]

            for e in entries:
                sol = e["sol"]
                r_plot_m = np.linspace(sol.t[0], sol.t[-1], n_resample)
                p_plot = _resample_sol_component(sol, idx=0, r_plot=r_plot_m)

                n_val, vac_sign = _extract_mode_and_sign(e, nu)
                ls = mode_linestyles[n_val % len(mode_linestyles)]
                label = _signed_label(r"P", n_val, vac_sign, nu)

                R_s_m = e["r_star"] * 1e3
                x_norm = r_plot_m / R_s_m

                ax_.plot(
                    x_norm,
                    p_plot,
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    label=label,
                )

    _plot_pressure_on_axis(plt.gca())

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xlabel(r"$r/R_s$")
    plt.xlim(left=0, right=2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.tick_params(axis="x", which="major", length=4, width=1.2)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)
    plt.ylabel("P [Pa]")
    plt.grid(False)

    # Legends: mode linestyle, lambda color
    modes_seen = sorted(
        {
            _extract_mode_and_sign(e, nu)[0]
            for entries in entries_by_param.values()
            for e in entries
        }
    )
    mode_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=mode_linestyles[n % len(mode_linestyles)],
            linewidth=1.6,
            label=rf"$n={n}$",
        )
        for n in modes_seen
    ]
    param_handles = [
        Line2D(
            [0],
            [0],
            color=param_colors[p],
            linestyle="-",
            linewidth=2.0,
            label=param_labels[p],
        )
        for p in param_values
    ]
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

    legend_handles = param_handles + mode_handles + star_handles
    plt.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="best",
        frameon=True,
    )
    # _add_zoom_inset(
    #    plt.gca(),
    #    _plot_pressure_on_axis,
    #    pressure_zoom,
    # )

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "pressure_compare_param.pdf"), dpi=dpi_val)

    # ========== MU_EFF^2 ==========
    plt.figure()
    def _plot_mu2_on_axis(ax_):
        for param_val in param_values:
            entries = entries_by_param[param_val]
            color = param_colors[param_val]

            for e in entries:
                r_mu = e.get("r_mu", np.array([]))
                mu2 = e.get("mu2", np.array([]))
                if r_mu is None or mu2 is None or r_mu.size < 2:
                    continue

                r_mu_km = r_mu / 1e3
                order = np.argsort(r_mu_km, kind="mergesort")
                r_mu_km = r_mu_km[order]
                mu2_m = mu2[order]

                r_mu_km, unique_idx = np.unique(r_mu_km, return_index=True)
                mu2_m = mu2_m[unique_idx]
                if r_mu_km.size < 2:
                    continue

                r_plot_km = np.linspace(r_mu_km[0], r_mu_km[-1], n_resample)
                mu2_plot_km = np.interp(r_plot_km, r_mu_km, mu2_m) * 1e6

                n_val, vac_sign = _extract_mode_and_sign(e, nu)
                ls = mode_linestyles[n_val % len(mode_linestyles)]

                R_s_km = e["r_star"]
                x_norm = r_plot_km / R_s_km

                ax_.plot(
                    x_norm,
                    mu2_plot_km,
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                )

    _plot_mu2_on_axis(plt.gca())

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xlabel(r"$r/R_s$")
    plt.ylabel(r"$\mu_{\rm eff}^2$ [Km$^{-2}$]")
    ax = plt.gca()
    linthresh = 1e-12

    plt.yscale("symlog", linthresh=linthresh)

    # ticks that work on BOTH positive and negative sides
    ax.yaxis.set_major_locator(
        SymmetricalLogLocator(base=10, linthresh=linthresh, subs=(1.0,))
    )
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(base=10, linthresh=linthresh, subs=np.arange(2, 10) * 0.1)
    )

    # label only even decades
    ax.yaxis.set_major_formatter(FuncFormatter(even_decade_only))

    ax.tick_params(axis="y", which="major", length=6, width=1)
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)

    plt.xlim(left=0, right=2)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(clean_x))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.tick_params(axis="x", which="major", length=4, width=1.2)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)
    plt.grid(False)

    modes_seen = sorted(
        {
            _extract_mode_and_sign(e, nu)[0]
            for entries in entries_by_param.values()
            for e in entries
        }
    )
    mode_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=mode_linestyles[n % len(mode_linestyles)],
            linewidth=1.6,
            label=rf"$n={n}$",
        )
        for n in modes_seen
    ]
    param_handles = [
        Line2D(
            [0],
            [0],
            color=param_colors[p],
            linestyle="-",
            linewidth=2.0,
            label=param_labels[p],
        )
        for p in param_values
    ]

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

    legend_handles = mode_handles + param_handles + star_handles
    plt.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper right",
        frameon=True,
    )

    ax = plt.gca()
    ax.margins(y=0.05)  # 5% headroom

    # _add_zoom_inset(
    #    plt.gca(),
    #    _plot_mu2_on_axis,
    #    mu2_zoom,
    #    yscale={"value": "symlog", "linthresh": linthresh},
    #    yformatter=even_decade_only,
    # )
    plt.tight_layout()
    plt.savefig(
        os.path.join(savepath, "mu2_compare_param.pdf"),
        dpi=dpi_val,
        bbox_inches="tight",
    )

    # ========== SIGMA ==========
    plt.figure()
    def _plot_sigma_on_axis(ax_):
        for param_val in param_values:
            entries = entries_by_param[param_val]
            color = param_colors[param_val]

            for e in entries:
                sol = e["sol"]
                r_plot_m = np.linspace(sol.t[0], sol.t[-1], n_resample)
                sigma_plot = _resample_sol_component(sol, idx=2, r_plot=r_plot_m)

                n_val, vac_sign = _extract_mode_and_sign(e, nu)
                ls = mode_linestyles[n_val % len(mode_linestyles)]

                R_s_m = e["r_star"] * 1e3
                x_norm = r_plot_m / R_s_m

                ax_.plot(
                    x_norm,
                    sigma_plot / M,
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                )

    _plot_sigma_on_axis(plt.gca())

    nu_line(nu, vacuum_sols)
    plt.xlabel(r"$r/R_s$")
    plt.ylabel(r"$\sigma_0 /M_{Pl}$")
    plt.xlim(left=0, right=2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(clean_x))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.tick_params(axis="x", which="major", length=4, width=1.2)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)
    ax = plt.gca()
    linthresh = 1e-8

    plt.yscale("symlog", linthresh=linthresh)

    # ticks that work on BOTH positive and negative sides
    ax.yaxis.set_major_locator(
        SymmetricalLogLocator(base=10, linthresh=linthresh, subs=(1.0,))
    )
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(base=10, linthresh=linthresh, subs=np.arange(2, 10) * 0.1)
    )
    # label only even decades (your function already supports negatives)
    ax.yaxis.set_major_formatter(FuncFormatter(even_decade_only))

    ax.tick_params(axis="y", which="major", length=6, width=1)
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)

    ax = plt.gca()
    ax.margins(y=0.05)  # 5% headroom

    modes_seen = sorted(
        {
            _extract_mode_and_sign(e, nu)[0]
            for entries in entries_by_param.values()
            for e in entries
        }
    )
    mode_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=mode_linestyles[n % len(mode_linestyles)],
            linewidth=1.6,
            label=rf"$n={n}$",
        )
        for n in modes_seen
    ]
    param_handles = [
        Line2D(
            [0],
            [0],
            color=param_colors[p],
            linestyle="-",
            linewidth=2.0,
            label=param_labels[p],
        )
        for p in param_values
    ]

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

    legend_handles = mode_handles + param_handles + star_handles
    plt.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper right",
        bbox_to_anchor=(1, 0.9),
        frameon=True,
    )

    plt.grid(False)

    # _add_zoom_inset(
    #    plt.gca(),
    #    _plot_sigma_on_axis,
    #    sigma_zoom,
    #    yscale=("symlog", {"linthresh": linthresh}),
    #    yformatter=even_decade_only,
    # )

    if title is None:
        if param == "lambda":
            title = (
                rf"$\xi={float(xi):g}$, $\mu={float(m):g}\,$eV, $v={float(nu):g}\,$eV"
            )
        else:
            title = rf"$\xi={float(xi):g}$, $\lambda={float(lmbda):g}$, $v={float(nu):g}\,$eV"

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "sigma_compare_param.pdf"), dpi=dpi_val)
