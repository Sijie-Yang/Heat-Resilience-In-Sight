"""Paper figures for Heat Resilience in Sight."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from scripts.config import CALIBRATE_KEY, CITIES, MONTH_LAB, MONTHS, PANEL_ORDER, SHR_RANK_ORDER, U0


# A restrained, print-friendly diverging scale shared by the two main figures.
# Warm colours denote supply shortfall; cool colours denote surplus.
SHR_COLORS = ("#8E1B2C", "#C94B3E", "#EDA57D", "#F7F4EF", "#9CC7D8", "#4A90B8", "#174A73")
INK = "#1F2933"
MUTED = "#53606B"
PANEL_BG = "#F6F8F9"
GRID = "#D8DEE3"
BLUE = "#174A73"
MID_BLUE = "#4A90B8"
LIGHT_BLUE = "#9CC7D8"
BRICK = "#C94B3E"
ORANGE = "#D7834B"
CITY_COLORS = (BLUE, BRICK, "#5E8C61", "#7C6AA6", ORANGE, "#4F8C8D", "#A6536F", "#7B8794")


def shr_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("hris_shr", SHR_COLORS, N=256)


def risk_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "hris_risk", ("#F7F4EF", "#F3C6A8", "#E8825B", "#C94B3E", "#8E1B2C"), N=256
    )


def polish_axes(ax, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=INK, width=0.7)
    ax.title.set_color(INK)
    ax.title.set_fontweight("bold")
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    if grid_axis:
        ax.grid(axis=grid_axis, color="white", linewidth=0.9)
        ax.set_axisbelow(True)


def add_panel_label(ax, label: str, fontsize: float = 8.0) -> None:
    """Place panel letters exactly as in main-text Figure 2."""
    ax.text(
        0.035, 0.95, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=MUTED,
        zorder=5,
    )


def style_horizontal_colorbar(cbar, zero: float | None = None, labelsize: float = 7.0) -> None:
    """Use the same thin bottom colour bar treatment as Figures 1 and 2."""
    cbar.ax.tick_params(labelsize=labelsize, length=2.5, width=0.6, colors=INK)
    cbar.outline.set_linewidth(0.6)
    if zero is not None:
        cbar.ax.axvline(zero, color=INK, lw=0.65)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 7.5,
            "axes.labelcolor": INK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "legend.fontsize": 6.2,
        }
    )


def draw_vata_box(stats: pd.DataFrame, path) -> None:
    keys = list(PANEL_ORDER)
    labels = [CITIES[k]["short"] for k in keys]
    means = [stats.loc[k, "mean"] for k in keys]
    stds = [stats.loc[k, "std"] for k in keys]
    fig, ax = plt.subplots(figsize=(5.50, 2.30))
    polish_axes(ax)
    colors = [BRICK if k == CALIBRATE_KEY else MID_BLUE for k in keys]
    ax.bar(range(len(keys)), means, yerr=stds, color=colors, ecolor=MUTED, capsize=3, width=0.72)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean VATA (0–5)")
    ax.set_ylim(0, 3.2)
    ax.axhline(stats.loc[CALIBRATE_KEY, "mean"], color=BRICK, ls="--", lw=1.0, alpha=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_utci(monthly: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(5.50, 2.56))
    polish_axes(ax)
    for i, key in enumerate(PANEL_ORDER):
        sub = monthly[monthly["city"] == key]
        ax.plot(
            sub["month"], sub["utci"], marker="o", ms=3.2, lw=1.5,
            color=CITY_COLORS[i], markeredgecolor="white", markeredgewidth=0.45,
            label=CITIES[key]["short"],
        )
    ax.axhline(U0, color=MUTED, ls="--", lw=1.0, label="UTCI 26 °C (moderate heat)")
    ax.axhline(32, color=BRICK, ls=":", lw=1.2, label="UTCI 32 °C (strong heat)")
    ax.set_xticks(MONTHS)
    ax.set_xticklabels(MONTH_LAB)
    ax.set_ylabel("Climatological outdoor UTCI (°C)")
    ax.set_xlabel("Month (2015–2024)")
    ax.legend(ncol=3, frameon=True, facecolor="white", edgecolor=GRID, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _matrix(monthly: pd.DataFrame, value: str, keys: list[str]) -> np.ndarray:
    mat = np.zeros((len(keys), 12))
    for i, key in enumerate(keys):
        sub = monthly[monthly["city"] == key].sort_values("month")
        mat[i] = sub[value].to_numpy()
    return mat


def draw_heatmap(monthly: pd.DataFrame, value: str, title: str, cbar: str, path, cmap="RdBu", vlim=None) -> None:
    keys = PANEL_ORDER
    mat = _matrix(monthly, value, keys)
    if vlim is None:
        vmax = float(np.nanmax(np.abs(mat)))
        vlim = (-vmax, vmax) if cmap.startswith("RdBu") else (0, float(np.nanmax(mat)))
    fig, ax = plt.subplots(figsize=(5.50, 2.62))
    plot_cmap = risk_cmap() if cmap == "YlOrRd" else (shr_cmap() if cmap.startswith("RdBu") else cmap)
    im = ax.imshow(mat, aspect="auto", cmap=plot_cmap, vmin=vlim[0], vmax=vlim[1])
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_LAB)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([CITIES[k]["short"] for k in keys])
    ax.tick_params(length=0, pad=3, colors=INK)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(keys), 1), minor=True)
    ax.grid(which="minor", color="#F7F4EF", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar_label = f"Low share     {cbar}     High share" if cbar == "Inadequate fraction" else cbar
    colorbar = fig.colorbar(
        im, ax=ax, orientation="horizontal", fraction=0.085, pad=0.15, aspect=55,
        label=colorbar_label,
    )
    style_horizontal_colorbar(colorbar, labelsize=7.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_shr_heatmap_wide(monthly: pd.DataFrame, path) -> None:
    """Landscape monthly SHR heatmap for the main-text Figure 1."""
    keys = SHR_RANK_ORDER
    mat = _matrix(monthly, "shr", keys)
    vmax = float(np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(5.50, 1.84))
    im = ax.imshow(mat, aspect="auto", cmap=shr_cmap(), vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_LAB, fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([CITIES[k]["short"] for k in keys], fontsize=8)
    ax.tick_params(length=0, pad=2.5, colors="#1F2933")
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(keys), 1), minor=True)
    ax.grid(which="minor", color="#F7F4EF", linewidth=0.75)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(keys)):
        for j in range(12):
            val = float(mat[i, j])
            ink = "white" if abs(val) > 0.50 * vmax else "#20262E"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=5.55, color=ink)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.085, pad=0.15, aspect=56)
    cbar.set_label("Supply shortfall     SHR (VATA units)     Supply surplus", fontsize=7.2, labelpad=2)
    style_horizontal_colorbar(cbar, zero=0, labelsize=6.8)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_warming(annual: pd.DataFrame, path) -> None:
    keys = PANEL_ORDER
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(5.50, 2.43))
    polish_axes(ax)
    w = 0.25
    frame = annual.reindex(keys)
    ax.bar(x - w, frame["shr"].to_numpy(), w, label="2015–2024", color=BLUE)
    ax.bar(x, frame["shr_plus15"].to_numpy(), w, label="+1.5 °C", color=ORANGE)
    ax.bar(x + w, frame["shr_plus25"].to_numpy(), w, label="+2.5 °C", color=BRICK)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([CITIES[k]["short"] for k in keys], rotation=25, ha="right")
    ax.set_ylabel("Annual-mean SHR")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_concept(path) -> None:
    fig, ax = plt.subplots(figsize=(5.50, 2.75))
    polish_axes(ax)
    v = np.linspace(0.6, 3.2, 80)
    for demand, lab, col in [
        (0.4, "Cool month (low UTCI demand)", "#5E8C61"),
        (1.0, "Hong Kong-summer demand", BLUE),
        (1.8, "Hotter / future climate", BRICK),
    ]:
        ax.plot(v, v - demand, color=col, lw=2, label=lab)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xlabel(r"Streetscape supply  $\bar{V}$  (mean VATA)")
    ax.set_ylabel(r"SHR $= \bar{V} - V^{\mathrm{need}}(U)$")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_paper_figures(stats: pd.DataFrame, monthly: pd.DataFrame, annual: pd.DataFrame, figs) -> None:
    style()
    figs.mkdir(parents=True, exist_ok=True)
    draw_concept(figs / "fig_shr_concept.png")
    draw_vata_box(stats, figs / "fig_raw_vata.png")
    draw_utci(monthly, figs / "fig_monthly_utci.png")
    draw_shr_heatmap_wide(monthly, figs / "fig_shr_heatmap.png")
    draw_heatmap(
        monthly,
        "inadequate_frac",
        "Share of streetscapes below climate-needed VATA",
        "Inadequate fraction",
        figs / "fig_inadequate_heatmap.png",
        cmap="YlOrRd",
        vlim=(0, 1),
    )
    draw_warming(annual, figs / "fig_warming_shr.png")


def draw_robustness(city_table: pd.DataFrame, path) -> None:
    from scripts.config import CITIES, PANEL_ORDER

    style()
    variants = sorted(city_table["variant"].unique())
    mat = np.zeros((len(PANEL_ORDER), len(variants)))
    for j, variant in enumerate(variants):
        part = city_table[city_table["variant"] == variant].set_index("city")
        mat[:, j] = part.reindex(PANEL_ORDER)["shr"].to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(5.50, 1.98))
    im = ax.imshow(mat, aspect="auto", cmap=shr_cmap(), vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(PANEL_ORDER)))
    ax.set_yticklabels([CITIES[k]["short"] for k in PANEL_ORDER])
    ax.set_xticks(range(len(variants)))
    compact_variants = []
    for variant in variants:
        window, slope, radiation = variant.split("|")
        slope_short = slope.replace("slope", "").replace("0.", ".")
        compact_variants.append(
            f"{window.upper()} / {slope_short} / {'sun' if radiation == 'open_sun' else 'shade'}"
        )
    ax.set_xticklabels(compact_variants, rotation=58, ha="right", fontsize=5.2)
    ax.tick_params(length=0, colors=INK)
    ax.set_xticks(np.arange(-0.5, len(variants), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PANEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="#F7F4EF", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.43)
    cax = fig.add_axes([0.22, 0.045, 0.56, 0.050])
    colorbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    colorbar.set_label("Shortfall     Annual-mean SHR     Surplus", fontsize=8.0, labelpad=2)
    style_horizontal_colorbar(colorbar, zero=0, labelsize=7.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
