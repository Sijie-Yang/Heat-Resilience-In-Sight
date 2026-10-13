"""Eight-city VATA / SHR maps on a regular metre hex grid.

matplotlib hexbin stretches with the axes. Here hexes are pointy-top
regular polygons in a local metre frame; each panel is a square with
equal aspect so they stay regular.

Street-level SHR is V_i - V^need(city, period). City-mean SHR has no map.
"""

from __future__ import annotations

from math import comb

from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import CALIBRATE_KEY, CITIES, CSV_DIR, FIGS, PANEL_ORDER, SHR_RANK_ORDER, TABLES
from scripts.figures import (
    BLUE,
    BRICK,
    GRID,
    INK,
    MID_BLUE,
    MUTED,
    PANEL_BG,
    add_panel_label,
    shr_cmap,
    style,
    style_horizontal_colorbar,
)

HEX_M = 800.0
HEX_SIZE = HEX_M / np.sqrt(3.0)  # pointy-top; neighbour spacing = 800 m
HEX_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
MINCNT = 4
LISA_P = 0.05
LISA_SHORTFALL = "#8E1B2C"
LISA_SHORTFALL_NS = "#EDA57D"
LISA_SURPLUS = "#174A73"
LISA_SURPLUS_NS = "#9CC7D8"
VMIN = 0.40
VMAX = 3.00
SHR_LIM = 2.00
BINS = 40
QUARTERS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def _load_points(key: str) -> pd.DataFrame:
    return pd.read_csv(
        CSV_DIR / f"{key}.csv",
        usecols=["longitude", "latitude", "thermal_affordance"],
    ).dropna()


def _to_local_m(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0 = np.deg2rad(float(np.median(lat)))
    lon0 = float(np.median(lon))
    latm = float(np.median(lat))
    x = (lon - lon0) * 111_320.0 * np.cos(lat0)
    y = (lat - latm) * 110_540.0
    return x, y


def _hk_mean(stats: pd.DataFrame) -> float:
    return float(stats.loc[CALIBRATE_KEY, "mean"])


def _cube_round(q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = -q - r
    qi = np.rint(q)
    ri = np.rint(r)
    si = np.rint(s)
    dq = np.abs(qi - q)
    dr = np.abs(ri - r)
    ds = np.abs(si - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = (~fix_q) & (dr > ds)
    qi = np.where(fix_q, -ri - si, qi)
    ri = np.where(fix_r, -qi - si, ri)
    return qi.astype(np.int32), ri.astype(np.int32)


def _xy_to_axial(x: np.ndarray, y: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    q = (np.sqrt(3.0) / 3.0 * x - y / 3.0) / size
    r = (2.0 / 3.0 * y) / size
    return _cube_round(q, r)


def _axial_to_xy(q: np.ndarray, r: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    cx = size * (np.sqrt(3.0) * q + np.sqrt(3.0) / 2.0 * r)
    cy = size * (1.5 * r)
    return cx, cy


def _hex_corners(q: np.ndarray, r: np.ndarray, size: float) -> np.ndarray:
    cx, cy = _axial_to_xy(q.astype(float), r.astype(float), size)
    angles = np.deg2rad(60.0 * np.arange(6) - 30.0)
    dx = size * np.cos(angles)
    dy = size * np.sin(angles)
    return np.stack([cx[:, None] + dx, cy[:, None] + dy], axis=2)


def hex_means(x: np.ndarray, y: np.ndarray, values: np.ndarray, mincnt: int = MINCNT) -> tuple[np.ndarray, np.ndarray]:
    q, r = _xy_to_axial(x, y, HEX_SIZE)
    frame = pd.DataFrame({"q": q, "r": r, "v": values})
    agg = frame.groupby(["q", "r"], sort=False)["v"].agg(["mean", "count"])
    agg = agg[agg["count"] >= mincnt]
    if agg.empty:
        return np.empty((0, 6, 2)), np.empty((0,))
    polys = _hex_corners(
        agg.index.get_level_values("q").to_numpy(),
        agg.index.get_level_values("r").to_numpy(),
        HEX_SIZE,
    )
    return polys, agg["mean"].to_numpy()


def hex_table(x: np.ndarray, y: np.ndarray, values: np.ndarray, mincnt: int = MINCNT) -> pd.DataFrame:
    q, r = _xy_to_axial(x, y, HEX_SIZE)
    frame = pd.DataFrame({"q": q, "r": r, "v": values})
    agg = frame.groupby(["q", "r"], sort=False)["v"].agg(["mean", "count"])
    agg = agg[agg["count"] >= mincnt].reset_index()
    return agg


def _hex_neighbor_index(q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key_to_i = {(int(qi), int(ri)): i for i, (qi, ri) in enumerate(zip(q, r))}
    n = len(q)
    neigh = np.full((n, 6), -1, dtype=np.int32)
    deg = np.zeros(n, dtype=np.int32)
    for i, (qi, ri) in enumerate(zip(q, r)):
        k = 0
        for dq, dr in HEX_NEIGHBORS:
            j = key_to_i.get((int(qi) + dq, int(ri) + dr))
            if j is not None:
                neigh[i, k] = j
                k += 1
        deg[i] = k
    return neigh, deg


def _neighbor_sum(values: np.ndarray, neigh: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    for k in range(neigh.shape[1]):
        j = neigh[:, k]
        valid = j >= 0
        out[valid] += values[j[valid]]
    return out


def _hypergeom_sf(k: int, population: int, successes: int, draws: int) -> float:
    if draws <= 0 or k <= 0:
        return 1.0
    max_x = min(draws, successes)
    if k > max_x:
        return 0.0
    if population < draws or successes < 0 or successes > population:
        return 1.0
    den = comb(population, draws)
    if den == 0:
        return 1.0
    total = 0
    failures = population - successes
    for x in range(k, max_x + 1):
        if x > successes or draws - x > failures:
            continue
        total += comb(successes, x) * comb(failures, draws - x)
    return total / den


def morans_i_binary(x: np.ndarray, neigh: np.ndarray, deg: np.ndarray) -> float:
    z = x.astype(float) - float(x.mean())
    denom = float(np.dot(z, z))
    s0 = float(deg.sum())
    if denom == 0.0 or s0 == 0.0:
        return float("nan")
    wz = _neighbor_sum(z, neigh)
    return (len(x) / s0) * float(np.dot(z, wz)) / denom


def shortfall_join_count(shr: np.ndarray, q: np.ndarray, r: np.ndarray) -> dict:
    """City-internal join-count clusters on hex-mean SHR < 0."""
    x = (shr < 0.0).astype(np.int32)
    neigh, deg = _hex_neighbor_index(q, r)
    shortfall_nb = _neighbor_sum(x.astype(float), neigh)
    n = len(x)
    n_short = int(x.sum())
    p = np.ones(n, dtype=float)
    for i in range(n):
        d = int(deg[i])
        if d == 0 or n <= 1:
            continue
        if x[i] == 1:
            p[i] = _hypergeom_sf(int(shortfall_nb[i]), n - 1, n_short - 1, d)
        else:
            p[i] = _hypergeom_sf(d - int(shortfall_nb[i]), n - 1, n - n_short - 1, d)
    sig = (deg > 0) & (p < LISA_P)
    cls = np.full(n, "ns_surplus", dtype=object)
    cls[x == 1] = "ns_shortfall"
    cls[(x == 1) & sig] = "cluster_shortfall"
    cls[(x == 0) & sig] = "cluster_surplus"
    return {
        "x": x,
        "p": p,
        "cls": cls,
        "n_hex": n,
        "shortfall_share": float(x.mean()) if n else float("nan"),
        "cluster_shortfall_share": float(np.mean(cls == "cluster_shortfall")) if n else float("nan"),
        "cluster_surplus_share": float(np.mean(cls == "cluster_surplus")) if n else float("nan"),
        "moran_i": morans_i_binary(x, neigh, deg),
    }


def _square_frame(ax, polys: np.ndarray, panel_frame: bool = False) -> None:
    if len(polys) == 0:
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        return
    xy = polys.reshape(-1, 2)
    mid = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    half = 0.5 * float(np.max(xy.max(axis=0) - xy.min(axis=0))) * 1.04
    half = max(half, HEX_M)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(panel_frame)
        if panel_frame:
            spine.set_color("#D8DEE3")
            spine.set_linewidth(0.55)


def _paint_hexes(
    ax,
    polys: np.ndarray,
    means: np.ndarray,
    cmap,
    norm,
    *,
    edgecolors="0.92",
    linewidths: float = 0.15,
    panel_frame: bool = False,
) -> None:
    coll = PolyCollection(
        polys,
        array=means,
        cmap=cmap,
        norm=norm,
        edgecolors=edgecolors,
        linewidths=linewidths,
    )
    ax.add_collection(coll)
    _square_frame(ax, polys, panel_frame=panel_frame)


def _paint_hexes_cat(ax, polys: np.ndarray, colors: np.ndarray, *, panel_frame: bool = False) -> None:
    coll = PolyCollection(
        polys,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
    )
    ax.add_collection(coll)
    _square_frame(ax, polys, panel_frame=panel_frame)


EIGHT_FIGSIZE = (5.50, 3.06)
EIGHT_ADJUST = dict(left=0.015, right=0.985, top=0.90, bottom=0.14, wspace=0.08, hspace=0.25)
EIGHT_CBAR = [0.22, 0.038, 0.56, 0.035]


def _eight_city_figure():
    style()
    fig, axes = plt.subplots(2, 4, figsize=EIGHT_FIGSIZE, subplot_kw={"box_aspect": 1})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(**EIGHT_ADJUST)
    return fig, axes


def _style_map_panel(ax, title: str, index: int) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, fontsize=7.6, pad=2.2, fontweight="bold", color=INK)
    add_panel_label(ax, chr(65 + index), fontsize=6.5)


def _save_eight(fig, path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_panel_points() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for key in PANEL_ORDER:
        pts = _load_points(key)
        x, y = _to_local_m(pts["longitude"].to_numpy(), pts["latitude"].to_numpy())
        out[key] = (x, y, pts["thermal_affordance"].to_numpy(dtype=float))
    return out


def v_need_period(monthly: pd.DataFrame, city: str, months: tuple[int, ...]) -> float:
    sub = monthly[(monthly["city"] == city) & monthly["month"].isin(months)]
    return float(sub["v_need"].mean())


def peak_quarter(key: str) -> str:
    return "DJF" if CITIES[key]["hemisphere"] == "S" else "JJA"


def draw_vata_distributions(stats: pd.DataFrame, path, v_ref: float | None = None) -> None:
    style()
    v_ref = _hk_mean(stats) if v_ref is None else v_ref
    fig, axes = plt.subplots(2, 4, figsize=(5.50, 2.95), sharex=True, sharey=True)
    ymax = 0.0
    for index, (ax, key) in enumerate(zip(axes.ravel(), PANEL_ORDER)):
        values = pd.read_csv(CSV_DIR / f"{key}.csv", usecols=["thermal_affordance"])[
            "thermal_affordance"
        ].to_numpy(dtype=float)
        ax.set_facecolor(PANEL_BG)
        ax.hist(values, bins=BINS, range=(0, 5), density=True, color=MID_BLUE, edgecolor="white", linewidth=0.25)
        ax.axvline(v_ref, color=BRICK, ls="-", lw=1.4, label="Hong Kong mean")
        ax.axvline(float(stats.loc[key, "mean"]), color=BLUE, ls="--", lw=1.1, label="City mean")
        n = int(stats.loc[key, "n"])
        mu = float(stats.loc[key, "mean"])
        med = float(stats.loc[key, "p50"])
        ax.set_title(
            f"{CITIES[key]['short']}  $n$={n:,}\nmean {mu:.2f}  median {med:.2f}",
            fontsize=5.7, fontweight="bold", color=INK, pad=1.8,
        )
        add_panel_label(ax, chr(65 + index), fontsize=5.8)
        ax.grid(axis="y", color="white", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ymax = max(ymax, ax.get_ylim()[1])
    axes[0, 0].set_ylim(0, ymax * 1.05)
    for ax in axes[1]:
        ax.set_xlabel("VATA (0–5)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        fontsize=5.8,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_vata_hexmaps(panel, stats: pd.DataFrame, path, v_ref: float | None = None) -> None:
    v_ref = _hk_mean(stats) if v_ref is None else v_ref
    cmap = shr_cmap()
    norm = mcolors.TwoSlopeNorm(vmin=VMIN, vcenter=v_ref, vmax=VMAX)
    cache = {}

    def values_for(key: str):
        if key not in cache:
            x, y, v = panel[key]
            cache[key] = hex_means(x, y, v)
        return cache[key]

    fig, axes = _eight_city_figure()
    for index, (ax, key) in enumerate(zip(axes.ravel(), SHR_RANK_ORDER)):
        _paint_hexes(
            ax, *values_for(key), cmap, norm,
            edgecolors="none", linewidths=0.0, panel_frame=True,
        )
        _style_map_panel(ax, CITIES[key]["short"], index)
    cax = fig.add_axes(EIGHT_CBAR)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    colorbar.set_label("Below anchor     Hex-mean VATA     Above anchor", fontsize=7.1, labelpad=2)
    style_horizontal_colorbar(colorbar, zero=v_ref, labelsize=6.3)
    _save_eight(fig, path)


def draw_shr_hexmaps(panel, monthly: pd.DataFrame, path) -> None:
    cmap = shr_cmap()
    norm = mcolors.TwoSlopeNorm(vmin=-SHR_LIM, vcenter=0.0, vmax=SHR_LIM)
    fig, axes = _eight_city_figure()
    for index, (ax, key) in enumerate(zip(axes.ravel(), SHR_RANK_ORDER)):
        x, y, v = panel[key]
        qname = peak_quarter(key)
        need = v_need_period(monthly, key, QUARTERS[qname])
        _paint_hexes(
            ax,
            *hex_means(x, y, v - need),
            cmap,
            norm,
            edgecolors="none",
            linewidths=0.0,
            panel_frame=True,
        )
        _style_map_panel(ax, f"{CITIES[key]['short']}  {qname}", index)
    cax = fig.add_axes(EIGHT_CBAR)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"Shortfall     Hex-mean SHR $= V_i - V^{\mathrm{need}}$     Surplus", fontsize=7.1, labelpad=2)
    style_horizontal_colorbar(cbar, zero=0, labelsize=6.3)
    _save_eight(fig, path)


def draw_shr_quarterly(panel, monthly: pd.DataFrame, keys: list[str], path) -> None:
    style()
    cmap = shr_cmap()
    norm = mcolors.TwoSlopeNorm(vmin=-SHR_LIM, vcenter=0.0, vmax=SHR_LIM)
    qnames = list(QUARTERS)
    n = len(keys)
    # Four square map columns need about 4.6 in of panel height at paper width;
    # keeping the canvas tighter prevents appendix pagination without shrinking type.
    fig, axes = plt.subplots(n, 4, figsize=(5.50, 4.55), subplot_kw={"box_aspect": 1})
    if n == 1:
        axes = np.array([axes])
    panel_index = 0
    for i, key in enumerate(keys):
        x, y, v = panel[key]
        for j, qname in enumerate(qnames):
            ax = axes[i, j]
            need = v_need_period(monthly, key, QUARTERS[qname])
            ax.set_facecolor(PANEL_BG)
            _paint_hexes(
                ax, *hex_means(x, y, v - need), cmap, norm,
                edgecolors="none", linewidths=0.0, panel_frame=True,
            )
            if i == 0:
                ax.set_title(qname, fontsize=7.2, fontweight="bold", color=INK)
            if j == 0:
                ax.set_ylabel(CITIES[key]["short"], fontsize=6.5, fontweight="bold", color=INK)
            add_panel_label(ax, chr(65 + panel_index), fontsize=5.5)
            panel_index += 1
    fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.10, wspace=0.08, hspace=0.18)
    cax = fig.add_axes([0.22, 0.026, 0.56, 0.022])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    colorbar.set_label(r"Shortfall     SHR $= V_i - V^{\mathrm{need}}$     Surplus", fontsize=8.0, labelpad=2)
    style_horizontal_colorbar(colorbar, zero=0, labelsize=7.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


LISA_COLORS = {
    "cluster_shortfall": LISA_SHORTFALL,
    "ns_shortfall": LISA_SHORTFALL_NS,
    "ns_surplus": LISA_SURPLUS_NS,
    "cluster_surplus": LISA_SURPLUS,
}


def city_hot_quarter_lisa(x: np.ndarray, y: np.ndarray, v: np.ndarray, need: float) -> tuple[np.ndarray, np.ndarray, dict]:
    table = hex_table(x, y, v - need)
    if table.empty:
        return np.empty((0, 6, 2)), np.empty((0,), dtype=object), {
            "n_hex": 0,
            "shortfall_share": float("nan"),
            "cluster_shortfall_share": float("nan"),
            "cluster_surplus_share": float("nan"),
            "moran_i": float("nan"),
        }
    lisa = shortfall_join_count(
        table["mean"].to_numpy(dtype=float),
        table["q"].to_numpy(),
        table["r"].to_numpy(),
    )
    polys = _hex_corners(table["q"].to_numpy(), table["r"].to_numpy(), HEX_SIZE)
    return polys, lisa["cls"], lisa


def draw_shr_lisa_maps(panel, monthly: pd.DataFrame, path) -> pd.DataFrame:
    fig, axes = _eight_city_figure()
    rows = []
    for index, (ax, key) in enumerate(zip(axes.ravel(), SHR_RANK_ORDER)):
        x, y, v = panel[key]
        qname = peak_quarter(key)
        need = v_need_period(monthly, key, QUARTERS[qname])
        polys, cls, lisa = city_hot_quarter_lisa(x, y, v, need)
        colors = np.array([LISA_COLORS[c] for c in cls], dtype=object) if len(cls) else np.empty((0,))
        _paint_hexes_cat(ax, polys, colors, panel_frame=True)
        _style_map_panel(ax, f"{CITIES[key]['short']}  {qname}", index)
        rows.append(
            {
                "city": key,
                "short": CITIES[key]["short"],
                "quarter": qname,
                "n_hex": lisa["n_hex"],
                "shortfall_share": lisa["shortfall_share"],
                "cluster_shortfall_share": lisa["cluster_shortfall_share"],
                "cluster_surplus_share": lisa["cluster_surplus_share"],
                "moran_i": lisa["moran_i"],
            }
        )
    cax = fig.add_axes(EIGHT_CBAR)
    cmap = mcolors.ListedColormap(
        [LISA_SHORTFALL, LISA_SHORTFALL_NS, LISA_SURPLUS_NS, LISA_SURPLUS]
    )
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.set_ticklabels(["Shortfall cluster", "Shortfall (n.s.)", "Surplus (n.s.)", "Surplus cluster"])
    cbar.ax.tick_params(labelsize=6.3, length=0, width=0.6, colors=INK)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("City-internal join-count     $p<0.05$", fontsize=7.1, labelpad=2)
    _save_eight(fig, path)
    return pd.DataFrame(rows)


def write_vata_spatial_figures(stats: pd.DataFrame, figs=None, panel=None) -> None:
    figs = FIGS if figs is None else figs
    figs.mkdir(parents=True, exist_ok=True)
    if panel is None:
        panel = load_panel_points()
    draw_vata_distributions(stats, figs / "fig_vata_distributions.png")
    draw_vata_hexmaps(panel, stats, figs / "fig_vata_maps.png")


def write_shr_spatial_figures(monthly: pd.DataFrame, figs=None, panel=None) -> None:
    figs = FIGS if figs is None else figs
    figs.mkdir(parents=True, exist_ok=True)
    if panel is None:
        panel = load_panel_points()
    draw_shr_hexmaps(panel, monthly, figs / "fig_shr_maps.png")
    lisa = draw_shr_lisa_maps(panel, monthly, figs / "fig_shr_lisa.png")
    TABLES.mkdir(parents=True, exist_ok=True)
    lisa.to_csv(TABLES / "lisa_hot_quarter.csv", index=False)
    mid = len(PANEL_ORDER) // 2
    draw_shr_quarterly(panel, monthly, PANEL_ORDER[:mid], figs / "fig_shr_quarterly_a.png")
    draw_shr_quarterly(panel, monthly, PANEL_ORDER[mid:], figs / "fig_shr_quarterly_b.png")
