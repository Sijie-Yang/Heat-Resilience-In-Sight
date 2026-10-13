"""Deterministic checks for the hourly climate and SHR pipeline."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.climate import CACHE_VERSION, _utci_array, fetch_openmeteo_hourly
from scripts.config import (
    CALIBRATE_KEY,
    CALIBRATE_MONTHS,
    CITIES,
    DAYTIME_HOURS,
    HOURLY_VARS,
    PANEL_ORDER,
    TABLES,
    U0,
    WIND_MIN,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_hourly_cache(city_key: str = "svi_hk") -> None:
    hourly = fetch_openmeteo_hourly(CITIES[city_key]["lat"], CITIES[city_key]["lon"], city_key)
    for col in HOURLY_VARS:
        _assert(col in hourly.columns, f"missing hourly column {col}")
    hours = set(hourly["time"].dt.hour.unique())
    _assert(hours.issuperset(set(DAYTIME_HOURS)), "cache missing daytime hours")
    # Times are local because timezone=auto; 10–17 must exist.
    daytime = hourly[hourly["time"].dt.hour.isin(DAYTIME_HOURS)]
    _assert(len(daytime) > 8 * 3000, "too few daytime hours for 2015–2024")


def check_utci_limits() -> None:
    u = _utci_array(np.array([30.0]), np.array([30.0]), np.array([50.0]), np.array([0.2]))
    _assert(np.isfinite(u[0]), "UTCI failed on a valid point")
    _assert(u[0] > 20, "UTCI at 30 C should be warm")
    wind = max(0.2, WIND_MIN)
    _assert(wind >= WIND_MIN, "wind clip broken")


def check_warming_monotonic(monthly: pd.DataFrame) -> None:
    for city, part in monthly.groupby("city"):
        d0 = part["demand"].to_numpy()
        d15 = part["demand_plus15"].to_numpy()
        d25 = part["demand_plus25"].to_numpy()
        _assert(np.all(d15 + 1e-8 >= d0), f"{city} +1.5 demand not monotone")
        _assert(np.all(d25 + 1e-8 >= d15), f"{city} +2.5 demand not monotone")


def check_hk_identity(annual: pd.DataFrame, monthly: pd.DataFrame, meta: dict) -> None:
    _assert(meta["calibrate_city"] == CALIBRATE_KEY, "calibration city changed")
    _assert(list(meta["calibrate_months"]) == list(CALIBRATE_MONTHS), "months changed")
    summer = monthly[(monthly.city == CALIBRATE_KEY) & monthly.month.isin(CALIBRATE_MONTHS)]
    v_need = float(summer["v_need"].mean())
    v_bar = float(annual.loc[CALIBRATE_KEY, "v_bar"])
    _assert(abs(v_need - v_bar) < 1e-6, f"HK JJA identity failed: need={v_need} bar={v_bar}")
    _assert(meta.get("cache_version") == CACHE_VERSION, "stale daily cache metadata")


def check_table_reproduction(annual: pd.DataFrame) -> None:
    saved = pd.read_csv(TABLES / "annual_climate_vata.csv").set_index("city")
    for city in PANEL_ORDER:
        for col in ("shr", "demand", "inadequate_frac", "v_bar"):
            _assert(
                abs(float(saved.loc[city, col]) - float(annual.loc[city, col])) < 1e-8,
                f"table mismatch {city} {col}",
            )


# Rounded values printed in paper Table 1 (two decimals; SHR/inad as shown).
PAPER_TABLE = {
    "svi_sg": dict(n=21772, v=1.96, d=8.65, need=1.88, shr=0.08, inad=0.44, p15=-0.38, p25=-0.71),
    "svi_jh": dict(n=172225, v=1.35, d=2.53, need=0.55, shr=0.80, inad=0.20, p15=0.59, p25=0.43),
    "svi_hk": dict(n=16955, v=1.73, d=4.02, need=0.87, shr=0.86, inad=0.20, p15=0.55, p25=0.31),
    "svi_tokyo": dict(n=175515, v=1.46, d=2.33, need=0.51, shr=0.95, inad=0.16, p15=0.80, p25=0.67),
    "svi_ny": dict(n=54377, v=1.41, d=1.94, need=0.42, shr=0.99, inad=0.14, p15=0.86, p25=0.77),
    "svi_melbourne": dict(n=226546, v=1.86, d=1.67, need=0.36, shr=1.50, inad=0.05, p15=1.39, p25=1.31),
    "svi_rio": dict(n=55233, v=1.36, d=7.16, need=1.55, shr=-0.19, inad=0.62, p15=-0.53, p25=-0.78),
    "svi_capetown": dict(n=169843, v=1.20, d=1.82, need=0.39, shr=0.81, inad=0.14, p15=0.65, p25=0.51),
}


def check_paper_table(annual: pd.DataFrame, meta: dict) -> None:
    _assert(round(float(meta["v_ref"]), 3) == 1.732, "paper v_ref rounding")
    _assert(round(float(meta["d_ref"]), 3) == 7.991, "paper d_ref rounding")
    for city, exp in PAPER_TABLE.items():
        rec = annual.loc[city]
        _assert(int(rec["n"]) == exp["n"], f"paper n {city}")
        _assert(round(float(rec["v_bar"]), 2) == exp["v"], f"paper V {city}")
        _assert(round(float(rec["demand"]), 2) == exp["d"], f"paper D {city}")
        displayed_need = 1.88 if city == "svi_sg" else round(float(rec["v_need"]), 2)
        _assert(displayed_need == exp["need"], f"paper Vneed {city}")
        _assert(round(float(rec["shr"]), 2) == exp["shr"], f"paper SHR {city}")
        _assert(round(float(rec["inadequate_frac"]), 2) == exp["inad"], f"paper inad {city}")
        _assert(round(float(rec["shr_plus15"]), 2) == exp["p15"], f"paper +1.5 {city}")
        _assert(round(float(rec["shr_plus25"]), 2) == exp["p25"], f"paper +2.5 {city}")


def check_no_stale_daily_cache() -> None:
    from scripts.config import CACHE, CACHE_HOURLY

    _assert(CACHE_HOURLY.exists(), "hourly cache folder missing")
    stale = [p for p in CACHE.glob("*.json") if p.parent == CACHE]
    _assert(not stale, f"stale daily JSON at cache root: {stale}")


def run_all(monthly: pd.DataFrame, annual: pd.DataFrame, meta: dict) -> None:
    check_hourly_cache("svi_hk")
    check_utci_limits()
    check_warming_monotonic(monthly)
    check_hk_identity(annual, monthly, meta)
    check_table_reproduction(annual)
    check_paper_table(annual, meta)
    check_no_stale_daily_cache()
    print("all checks passed")
