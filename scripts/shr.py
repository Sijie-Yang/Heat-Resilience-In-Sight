"""Hong Kong summer-calibrated streetscape heat resilience (SHR)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.climate import CACHE_VERSION
from scripts.config import (
    CALIBRATE_KEY,
    CALIBRATE_MONTHS,
    CITIES,
    DAYTIME_HOURS,
    SOLAR_ASSUMPTIONS,
    TABLES,
    U0,
)
from scripts.panel import load_all_vata


def inadequate_fraction(values: np.ndarray, v_need: float) -> float:
    if v_need <= 0:
        return 0.0
    return float(np.mean(values < v_need))


def attach_shr(
    stats: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    calibrate_key: str = CALIBRATE_KEY,
    calibrate_months: tuple[int, ...] = CALIBRATE_MONTHS,
    slope: float = 1.0,
    demand_col: str = "demand",
    demand_plus15_col: str = "demand_plus15",
    demand_plus25_col: str = "demand_plus25",
    vata_values: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Scale climate demand onto the VATA axis using a stated reference.

    V_need = slope * V_ref * D / D_ref. Default D is the open-sun daytime
    proxy; pass demand_shade* columns for the shaded bound.
    """
    calib = monthly[monthly["city"] == calibrate_key]
    summer = calib[calib["month"].isin(calibrate_months)]
    d_ref = float(summer[demand_col].mean())
    v_ref = float(stats.loc[calibrate_key, "mean"])
    if d_ref <= 0:
        raise RuntimeError("reference-city summer demand is zero; check UTCI")

    monthly = monthly.copy()
    monthly["v_bar"] = monthly["city"].map(stats["mean"])
    monthly["v_need"] = slope * v_ref * monthly[demand_col] / d_ref
    monthly["v_need_plus15"] = slope * v_ref * monthly[demand_plus15_col] / d_ref
    monthly["v_need_plus25"] = slope * v_ref * monthly[demand_plus25_col] / d_ref
    monthly["shr"] = monthly["v_bar"] - monthly["v_need"]
    monthly["shr_plus15"] = monthly["v_bar"] - monthly["v_need_plus15"]
    monthly["shr_plus25"] = monthly["v_bar"] - monthly["v_need_plus25"]
    monthly["rta"] = np.where(monthly["v_need"] > 1e-6, monthly["v_bar"] / monthly["v_need"], np.nan)

    vata_values = load_all_vata() if vata_values is None else vata_values
    monthly["inadequate_frac"] = [
        inadequate_fraction(vata_values[rec.city], rec.v_need) for rec in monthly.itertuples()
    ]

    annual = (
        monthly.groupby("city")
        .agg(
            demand=(demand_col, "mean"),
            v_need=("v_need", "mean"),
            shr=("shr", "mean"),
            shr_plus15=("shr_plus15", "mean"),
            shr_plus25=("shr_plus25", "mean"),
            inadequate_frac=("inadequate_frac", "mean"),
            rta=("rta", "mean"),
        )
        .join(stats[["n", "mean", "std", "label", "short", "koppen"]])
    )
    utci_col = "utci_shade" if demand_col.startswith("demand_shade") and "utci_shade" in monthly.columns else "utci"
    if utci_col in monthly.columns:
        annual["utci"] = monthly.groupby("city")[utci_col].mean()
    annual["v_bar"] = annual["mean"]

    meta = {
        "u0_utci_c": U0,
        "calibrate_city": calibrate_key,
        "calibrate_months": list(calibrate_months),
        "calibration_slope": slope,
        "demand_column": demand_col,
        "v_ref": v_ref,
        "d_ref": d_ref,
        "daytime_hours_local": list(DAYTIME_HOURS),
        "tdb_proxy": "hourly_temperature_2m",
        "tmrt_open_sun": "T_air + SolarCal delta_mrt (standardised open-sun pedestrian)",
        "tmrt_shaded": "T_air",
        "demand": (
            "mean of daily daytime-max max(UTCI-26, 0); "
            "open-sun unless demand_column is shaded"
        ),
        "solar_assumptions": SOLAR_ASSUMPTIONS,
        "cache_version": CACHE_VERSION,
        "climate_source": "Open-Meteo archive ERA5-land hourly 2015-2024 city centroid",
        "vata_note": "Singapore-trained TCIS two-stage MTNNL; aligned 1024x512 inference",
        "cities": {k: CITIES[k] for k in CITIES},
    }
    return monthly, annual, meta


def write_tables(monthly: pd.DataFrame, annual: pd.DataFrame, meta: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(TABLES / "monthly_climate_vata.csv", index=False)
    annual.to_csv(TABLES / "annual_climate_vata.csv")
    (TABLES / "calibration.json").write_text(json.dumps(meta, indent=2))
