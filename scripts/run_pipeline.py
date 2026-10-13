"""Regenerate climate, SHR, robustness, domain-shift, figures, and checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.checks import run_all
from scripts.climate import panel_climatology
from scripts.config import FIGS, PANEL_ORDER, TABLES
from scripts.domain_shift import write_domain_shift
from scripts.figures import draw_robustness, write_paper_figures
from scripts.maps import load_panel_points, write_shr_spatial_figures, write_vata_spatial_figures
from scripts.panel import summarise_panel
from scripts.robustness import write_robustness
from scripts.shr import attach_shr, write_tables


def main() -> None:
    stats = summarise_panel(write=True)
    monthly_raw = panel_climatology()
    monthly, annual, meta = attach_shr(stats, monthly_raw)
    write_tables(monthly, annual, meta)
    write_paper_figures(stats, monthly, annual, FIGS)
    robust_meta = write_robustness(stats, monthly_raw)
    import pandas as pd

    city_table = pd.read_csv(TABLES / "robustness_city_annual.csv")
    draw_robustness(city_table, FIGS / "fig_robustness.png")
    domain = write_domain_shift()
    panel = load_panel_points()
    write_vata_spatial_figures(stats, FIGS, panel=panel)
    write_shr_spatial_figures(monthly, FIGS, panel=panel)
    run_all(monthly, annual, meta)
    print("calibration", {k: meta[k] for k in ("calibrate_city", "v_ref", "d_ref", "demand_column", "cache_version")})
    print(annual.loc[PANEL_ORDER, ["short", "n", "v_bar", "utci", "demand", "shr", "inadequate_frac"]].round(3))
    print("robustness", robust_meta)
    print("domain shift")
    print(domain.round(3))


if __name__ == "__main__":
    main()
