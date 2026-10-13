"""Calibration and radiation robustness for SHR."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.config import CALIBRATE_KEY, CALIBRATE_MONTHS, PANEL_ORDER, TABLES
from scripts.panel import load_all_vata
from scripts.shr import attach_shr


WINDOWS = {
    "july": (7,),
    "jja": (6, 7, 8),
    "mjjas": (5, 6, 7, 8, 9),
}
SLOPES = (0.8, 1.0, 1.2)
RADIATION = {
    "open_sun": ("demand", "demand_plus15", "demand_plus25"),
    "shaded": ("demand_shade", "demand_shade_plus15", "demand_shade_plus25"),
}


def _variant_label(window: str, slope: float, radiation: str) -> str:
    return f"{window}|slope{slope:.1f}|{radiation}"


def run_variants(stats: pd.DataFrame, monthly_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vata_values = load_all_vata()
    rows = []
    annual_rows = []
    for window_name, months in WINDOWS.items():
        for slope in SLOPES:
            for rad_name, (d0, d15, d25) in RADIATION.items():
                _, annual, _ = attach_shr(
                    stats,
                    monthly_raw,
                    calibrate_key=CALIBRATE_KEY,
                    calibrate_months=months,
                    slope=slope,
                    demand_col=d0,
                    demand_plus15_col=d15,
                    demand_plus25_col=d25,
                    vata_values=vata_values,
                )
                label = _variant_label(window_name, slope, rad_name)
                annual = annual.copy()
                annual["variant"] = label
                annual["window"] = window_name
                annual["slope"] = slope
                annual["radiation"] = rad_name
                annual_rows.append(annual.reset_index())
                for city in PANEL_ORDER:
                    rec = annual.loc[city]
                    rows.append(
                        {
                            "variant": label,
                            "window": window_name,
                            "slope": slope,
                            "radiation": rad_name,
                            "city": city,
                            "short": rec["short"],
                            "shr": rec["shr"],
                            "inadequate_frac": rec["inadequate_frac"],
                            "sign": int(np.sign(rec["shr"])),
                        }
                    )
    city_table = pd.DataFrame(rows)
    annual_long = pd.concat(annual_rows, ignore_index=True)
    return city_table, annual_long


def summarise_robustness(city_table: pd.DataFrame) -> pd.DataFrame:
    base = city_table[
        (city_table["window"] == "jja")
        & (city_table["slope"] == 1.0)
        & (city_table["radiation"] == "open_sun")
    ].set_index("city")
    rows = []
    for variant, part in city_table.groupby("variant"):
        part = part.set_index("city").reindex(PANEL_ORDER)
        base_shr = base.reindex(PANEL_ORDER)["shr"].to_numpy(dtype=float)
        var_shr = part["shr"].to_numpy(dtype=float)
        base_sign = np.sign(base_shr)
        var_sign = np.sign(var_shr)
        spearman = pd.Series(base_shr).corr(pd.Series(var_shr), method="spearman")
        rows.append(
            {
                "variant": variant,
                "window": part["window"].iloc[0],
                "slope": float(part["slope"].iloc[0]),
                "radiation": part["radiation"].iloc[0],
                "spearman_vs_baseline": float(spearman),
                "cities_sign_flip": int(np.sum(base_sign != var_sign)),
                "mean_abs_shr_delta": float(np.mean(np.abs(var_shr - base_shr))),
                "mean_inad_delta": float(
                    np.mean(part["inadequate_frac"].to_numpy() - base["inadequate_frac"].reindex(PANEL_ORDER).to_numpy())
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("variant")


def write_robustness(stats: pd.DataFrame, monthly_raw: pd.DataFrame) -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    city_table, annual_long = run_variants(stats, monthly_raw)
    summary = summarise_robustness(city_table)
    city_table.to_csv(TABLES / "robustness_city_annual.csv", index=False)
    summary.to_csv(TABLES / "robustness_summary.csv", index=False)
    annual_long.to_csv(TABLES / "robustness_annual_long.csv", index=False)

    baseline = city_table[
        (city_table.window == "jja") & (city_table.slope == 1.0) & (city_table.radiation == "open_sun")
    ]
    invariant = []
    for city, part in city_table.groupby("city"):
        signs = set(part["sign"].tolist())
        invariant.append(
            {
                "city": city,
                "short": part["short"].iloc[0],
                "sign_stable": len(signs) == 1,
                "always_surplus": signs == {1},
                "always_deficit": signs == {-1},
                "min_shr": float(part["shr"].min()),
                "max_shr": float(part["shr"].max()),
            }
        )
    invariant_df = pd.DataFrame(invariant)
    invariant_df.to_csv(TABLES / "robustness_sign_stability.csv", index=False)
    meta = {
        "n_variants": int(city_table["variant"].nunique()),
        "baseline": "jja|slope1.0|open_sun",
        "cities_always_surplus": invariant_df.loc[invariant_df.always_surplus, "short"].tolist(),
        "cities_always_deficit": invariant_df.loc[invariant_df.always_deficit, "short"].tolist(),
        "cities_sign_unstable": invariant_df.loc[~invariant_df.sign_stable, "short"].tolist(),
        "min_spearman": float(summary["spearman_vs_baseline"].min()),
    }
    (TABLES / "robustness_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
