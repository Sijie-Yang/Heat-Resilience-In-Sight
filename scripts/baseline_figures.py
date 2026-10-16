"""Figures for the simple-indicator baselines.

draw_baselines         -> figures/fig_baselines.png         (appendix, city-average margins)
draw_street_agreement  -> figures/fig_street_agreement.png  (main-text Figure 4, hex-level agreement)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import CITIES, FIGS, SHR_RANK_ORDER, TABLES
from scripts.figures import BLUE, GRID, INK, MUTED, polish_axes, style
from scripts.maps import _to_local_m, peak_quarter

def draw_baselines(path) -> None:
    """City-average margins by indicator, drawn exactly like the warming bar chart (draw_warming)."""
    from scripts.config import PANEL_ORDER
    from scripts.figures import ORANGE
    style()
    keys = PANEL_ORDER
    x = np.arange(len(keys))
    b = pd.read_csv(TABLES / "baselines_city.csv").set_index("city").reindex(keys)
    fig, ax = plt.subplots(figsize=(5.50, 2.43))
    polish_axes(ax)
    w = 0.25
    ax.bar(x - w, b["vata_shr_rel"].to_numpy(), w, label="Supply score (VATA)", color=BLUE)
    ax.bar(x, b["green_view_shr_rel"].to_numpy(), w, label="Green-view fraction", color="#5E8C61")
    ax.bar(x + w, b["shade_proxy_shr_rel"].to_numpy(), w, label="1 $-$ sky fraction", color=ORANGE)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([CITIES[k]["short"] for k in keys], rotation=25, ha="right")
    ax.set_ylabel("Annual-mean margin\n(HK-mean units)")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def hex_margins() -> dict:
    """Hottest-quarter hex-mean margins (HK-mean units) for VATA and green view, per city."""
    import json
    from scripts.config import CSV_DIR
    from scripts.maps import HEX_SIZE, MINCNT, QUARTERS, _xy_to_axial
    monthly = pd.read_csv(TABLES / "monthly_climate_vata.csv")
    d_ref = json.loads((TABLES / "calibration.json").read_text())["d_ref"]
    frames = {k: pd.read_csv(CSV_DIR / f"{k}.csv", usecols=["longitude", "latitude", "thermal_affordance",
                                                             "seg_vegetation"]).dropna() for k in SHR_RANK_ORDER}
    hk_v, hk_g = frames["svi_hk"].thermal_affordance.mean(), frames["svi_hk"].seg_vegetation.mean()
    out = {}
    for k, f in frames.items():
        need = monthly[(monthly.city == k) & monthly.month.isin(QUARTERS[peak_quarter(k)])]["demand"].mean() / d_ref
        x, y = _to_local_m(f.longitude.to_numpy(), f.latitude.to_numpy())
        q, r = _xy_to_axial(x, y, HEX_SIZE)
        t = pd.DataFrame({"q": q, "r": r, "v": f.thermal_affordance.to_numpy() / hk_v,
                          "g": f.seg_vegetation.to_numpy() / hk_g}).groupby(["q", "r"])
        m = t.mean()[t.size() >= MINCNT]
        out[k] = (m["g"].to_numpy() - need, m["v"].to_numpy() - need)
    return out


def draw_street_agreement(path) -> pd.DataFrame:
    from scripts.figures import BRICK, ORANGE
    style()
    data = hex_margins()
    fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.12, wspace=0.08, hspace=0.30)
    lo, hi = -1.25, 2.75
    c_short, c_ok, c_diff = BRICK, "#4A90B8", ORANGE
    rows = []
    for i, (ax, k) in enumerate(zip(axes.ravel(), SHR_RANK_ORDER)):
        g, v = data[k]
        both_short, both_ok = (g < 0) & (v < 0), (g >= 0) & (v >= 0)
        diff = ~(both_short | both_ok)
        ax.set_facecolor("#F6F8F9")
        ax.axhline(0, color=INK, lw=0.6, zorder=1)
        ax.axvline(0, color=INK, lw=0.6, zorder=1)
        for mask, col, a in ((both_ok, c_ok, 0.35), (both_short, c_short, 0.45), (diff, c_diff, 0.85)):
            ax.scatter(g[mask], v[mask], s=1.6, color=col, alpha=a, lw=0, rasterized=True, zorder=2)
        share = float(diff.mean())
        ax.text(0.96, 0.06, f"{share:.0%}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                fontweight="bold", color=c_diff)
        ax.set_title(CITIES[k]["short"], fontsize=7.4, pad=2.5, color=INK, fontweight="bold")
        ax.set_xlim(lo, hi); ax.set_ylim(-1.15, 1.65)
        ax.set_xticks([-1, 0, 1, 2]); ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=6.2, length=2, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(GRID); sp.set_linewidth(0.6)
        rows.append({"city": k, "n_hex": len(g), "both_short": float(both_short.mean()),
                     "both_ok": float(both_ok.mean()), "disagree": share})
    fig.supxlabel("Street margin by green-view fraction (HK-mean units)", fontsize=7.2, y=0.015)
    fig.supylabel("Street margin by VATA", fontsize=7.2, x=0.01)
    h = [plt.Line2D([], [], marker="o", ls="", color=c_short, ms=5, label="Both: shortfall"),
         plt.Line2D([], [], marker="o", ls="", color=c_ok, ms=5, label="Both: surplus"),
         plt.Line2D([], [], marker="o", ls="", color=c_diff, ms=5, label="Judged differently (share in corner)")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=3, fontsize=6.6, framealpha=1.0,
               edgecolor=GRID, handletextpad=0.2, columnspacing=1.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "street_agreement_green.csv", index=False)
    return out


if __name__ == "__main__":
    draw_baselines(FIGS / "fig_baselines.png")
    print(draw_street_agreement(FIGS / "fig_street_agreement.png").round(3).to_string(index=False))
