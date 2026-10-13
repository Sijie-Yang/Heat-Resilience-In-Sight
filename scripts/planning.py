"""Indicative planning strategies from the hottest-quarter shortfall clusters.

Built on the same 800 m hexes, hottest-quarter V_need and city-internal
join-count test as the LISA cluster map. Rules:
  priority  = significant shortfall cluster, or any shortfall hex in a city
              where >= 50% of hexes are short (shortfall too widespread to cluster)
    - plant trees     : priority hex with green-view fraction below the city median
    - add built shade : priority hex already at or above the city median green view
  spot retrofit = other shortfall hex
  protect       = significant surplus cluster (keep existing canopy and shade)
  maintain      = other surplus hex
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import CITIES, CSV_DIR, FIGS, SHR_RANK_ORDER, TABLES
from scripts.figures import INK
from scripts.maps import (HEX_SIZE, MINCNT, QUARTERS, EIGHT_CBAR, _eight_city_figure, _hex_corners,
                          _paint_hexes_cat, _save_eight, _style_map_panel, _to_local_m, _xy_to_axial,
                          peak_quarter, shortfall_join_count, v_need_period)

WIDESPREAD = 0.50
STRATEGY = {
    "trees": ("Priority: plant trees", "#B5452E"),
    "shade": ("Priority: add built shade", "#6E3F8C"),
    "spot": ("Spot retrofit", "#EDA57D"),
    "maintain": ("Maintain", "#D5DEE5"),
    "protect": ("Protect canopy and shade", "#174A73"),
}


def city_strategy(key: str, monthly: pd.DataFrame):
    f = pd.read_csv(CSV_DIR / f"{key}.csv", usecols=["longitude", "latitude", "thermal_affordance", "seg_vegetation"]).dropna()
    need = v_need_period(monthly, key, QUARTERS[peak_quarter(key)])
    x, y = _to_local_m(f.longitude.to_numpy(), f.latitude.to_numpy())
    q, r = _xy_to_axial(x, y, HEX_SIZE)
    t = (pd.DataFrame({"q": q, "r": r, "shr": f.thermal_affordance.to_numpy() - need, "green": f.seg_vegetation.to_numpy()})
         .groupby(["q", "r"]).agg(shr=("shr", "mean"), green=("green", "mean"), n=("shr", "size")).reset_index())
    t = t[t.n >= MINCNT].reset_index(drop=True)
    lisa = shortfall_join_count(t.shr.to_numpy(), t.q.to_numpy(), t.r.to_numpy())
    cls = lisa["cls"]
    short = t.shr.to_numpy() < 0
    widespread = short.mean() >= WIDESPREAD
    priority = (cls == "cluster_shortfall") | (short & widespread)
    low_green = t.green.to_numpy() < np.median(t.green)
    s = np.full(len(t), "maintain", dtype=object)
    s[short] = "spot"
    s[priority & low_green] = "trees"
    s[priority & ~low_green] = "shade"
    s[cls == "cluster_surplus"] = "protect"
    polys = _hex_corners(t.q.to_numpy(), t.r.to_numpy(), HEX_SIZE)
    shares = {k: float(np.mean(s == k)) for k in STRATEGY}
    return polys, s, {"city": key, "label": CITIES[key]["label"], "quarter": peak_quarter(key), "n_hex": len(t),
                      "widespread": bool(widespread), **shares}


def draw(path=FIGS / "fig_planning.png") -> pd.DataFrame:
    monthly = pd.read_csv(TABLES / "monthly_climate_vata.csv")
    fig, axes = _eight_city_figure()
    rows = []
    for i, (ax, key) in enumerate(zip(axes.ravel(), SHR_RANK_ORDER)):
        polys, s, st = city_strategy(key, monthly)
        _paint_hexes_cat(ax, polys, np.array([STRATEGY[c][1] for c in s], dtype=object), panel_frame=True)
        _style_map_panel(ax, f"{CITIES[key]['short']}  {st['quarter']}", i)
        rows.append(st)
    order = ["trees", "shade", "spot", "maintain", "protect"]
    cax = fig.add_axes(EIGHT_CBAR)
    cmap = mcolors.ListedColormap([STRATEGY[k][1] for k in order])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.BoundaryNorm(range(6), cmap.N))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
    cbar.set_ticklabels(["Plant trees", "Add built shade", "Spot retrofit", "Maintain", "Protect"])
    cbar.ax.tick_params(labelsize=6.0, length=0, colors=INK)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("Priority = shortfall cluster or citywide shortfall; trees vs built shade by city-median green view", fontsize=6.6, labelpad=2)
    _save_eight(fig, path)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "planning_strategy.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(draw().round(3).to_string(index=False))
