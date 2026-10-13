"""Hourly ERA5-land climatology and radiation-aware pedestrian UTCI.

Demand is a standardised city-centroid proxy: daytime (10:00–17:00 local)
maximum UTCI, then the monthly mean of max(UTCI − 26, 0). Open-sun MRT
adds a SolarCal increment; the shaded bound uses T_mrt = T_air.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pysolar.solar import get_altitude
from pythermalcomfort.models import solar_gain, utci

from scripts.config import (
    CACHE_HOURLY,
    CITIES,
    DAYTIME_HOURS,
    HOURLY_VARS,
    PANEL_ORDER,
    SOLAR_ASSUMPTIONS,
    U0,
    WARMING,
    WIND_MIN,
)

CACHE_VERSION = "hourly_v1"


def _openmeteo_url(lat: float, lon: float, start: str, end: str) -> str:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }
    return "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)


def _get_json(url: str, retries: int = 6) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "hris-workshop-draft/2.0"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise last


def fetch_openmeteo_hourly(lat: float, lon: float, city_key: str) -> pd.DataFrame:
    """Fetch or load versioned hourly cache. Old daily JSON is never reused."""
    CACHE_HOURLY.mkdir(parents=True, exist_ok=True)
    dest = CACHE_HOURLY / f"{city_key}.json"
    payload = None
    if dest.exists():
        payload = json.loads(dest.read_text())
        if payload.get("hris_cache") != CACHE_VERSION or "hourly" not in payload:
            dest.unlink()
            payload = None

    if payload is None:
        payload = _get_json(_openmeteo_url(lat, lon, "2015-01-01", "2024-12-31"))
        if "hourly" not in payload or not payload.get("hourly", {}).get("time"):
            chunks = []
            for year in range(2015, 2025):
                part = _get_json(_openmeteo_url(lat, lon, f"{year}-01-01", f"{year}-12-31"))
                chunks.append(part)
                time.sleep(1.5)
            payload = dict(chunks[0])
            hourly = {k: [] for k in chunks[0]["hourly"]}
            for part in chunks:
                for k, vals in part["hourly"].items():
                    hourly[k].extend(vals)
            payload["hourly"] = hourly
        payload["hris_cache"] = CACHE_VERSION
        dest.write_text(json.dumps(payload))
        time.sleep(1.5)

    hourly = payload["hourly"]
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"])
    frame.attrs["timezone"] = payload.get("timezone", "UTC")
    frame.attrs["latitude"] = float(payload.get("latitude", lat))
    frame.attrs["longitude"] = float(payload.get("longitude", lon))
    return frame


def _solar_altitude(lat: float, lon: float, times: pd.Series, tz_name: str) -> np.ndarray:
    tz = ZoneInfo(tz_name)
    out = np.empty(len(times), dtype=float)
    for i, ts in enumerate(times):
        dt = ts.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        try:
            out[i] = float(get_altitude(lat, lon, dt))
        except Exception:
            out[i] = np.nan
    return out


def _delta_mrt(sol_alt: np.ndarray, dni: np.ndarray) -> np.ndarray:
    """Standardised open-sun SolarCal increment; zero when the sun is down."""
    alt = np.clip(np.asarray(sol_alt, dtype=float), 0.0, 90.0)
    dni = np.clip(np.asarray(dni, dtype=float), 0.0, None)
    day = alt > 0.0
    delta = np.zeros(alt.shape, dtype=float)
    idx = np.flatnonzero(day)
    if idx.size == 0:
        return delta
    chunk = 4000
    for start in range(0, idx.size, chunk):
        sl = idx[start : start + chunk]
        result = solar_gain(
            sol_altitude=alt[sl].tolist(),
            sharp=SOLAR_ASSUMPTIONS["sharp"],
            sol_radiation_dir=dni[sl].tolist(),
            sol_transmittance=SOLAR_ASSUMPTIONS["sol_transmittance"],
            f_svv=SOLAR_ASSUMPTIONS["f_svv"],
            f_bes=SOLAR_ASSUMPTIONS["f_bes"],
            asw=SOLAR_ASSUMPTIONS["asw"],
            posture=SOLAR_ASSUMPTIONS["posture"],
            floor_reflectance=SOLAR_ASSUMPTIONS["floor_reflectance"],
            round_output=False,
        )
        delta[sl] = np.asarray(result.delta_mrt, dtype=float)
    return np.nan_to_num(delta, nan=0.0)


def _utci_array(tdb: np.ndarray, tr: np.ndarray, rh: np.ndarray, wind: np.ndarray) -> np.ndarray:
    tdb = np.asarray(tdb, dtype=float)
    tr = np.asarray(tr, dtype=float)
    rh = np.clip(np.asarray(rh, dtype=float), 1.0, 99.0)
    wind = np.maximum(np.asarray(wind, dtype=float), WIND_MIN)
    ok = np.isfinite(tdb) & np.isfinite(tr) & np.isfinite(rh) & np.isfinite(wind)
    out = np.full(tdb.shape, np.nan)
    if not np.any(ok):
        return out
    try:
        result = utci(
            tdb=tdb[ok].tolist(),
            tr=tr[ok].tolist(),
            v=wind[ok].tolist(),
            rh=rh[ok].tolist(),
            limit_inputs=False,
            round_output=False,
        )
        out[ok] = np.asarray(result.utci, dtype=float)
    except Exception:
        for i in np.flatnonzero(ok):
            try:
                out[i] = float(
                    utci(
                        tdb=float(tdb[i]),
                        tr=float(tr[i]),
                        v=float(wind[i]),
                        rh=float(rh[i]),
                        limit_inputs=False,
                        round_output=False,
                    ).utci
                )
            except Exception:
                out[i] = np.nan
    return out


def hourly_utci(hourly: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    frame = hourly.copy()
    tz_name = hourly.attrs.get("timezone", "UTC")
    frame = frame[frame["time"].dt.hour.isin(DAYTIME_HOURS)].copy()
    if frame.empty:
        raise RuntimeError("no daytime hourly rows after local-time filter")

    tdb = frame["temperature_2m"].to_numpy(dtype=float)
    rh = frame["relative_humidity_2m"].to_numpy(dtype=float)
    wind = frame["wind_speed_10m"].to_numpy(dtype=float)
    dni = frame["direct_normal_irradiance"].to_numpy(dtype=float)
    sol_alt = _solar_altitude(lat, lon, frame["time"], tz_name)
    delta = _delta_mrt(sol_alt, dni)
    tr_open = tdb + delta
    tr_shade = tdb.copy()

    rec = {
        "time": frame["time"].to_numpy(),
        "tdb": tdb,
        "rh": rh,
        "wind": wind,
        "dni": dni,
        "sol_alt": sol_alt,
        "delta_mrt": delta,
    }
    for dT in WARMING:
        suffix = "" if dT == 0 else f"_plus{str(dT).replace('.', '')}"
        rec[f"utci_open{suffix}"] = _utci_array(tdb + dT, tr_open + dT, rh, wind)
        rec[f"utci_shade{suffix}"] = _utci_array(tdb + dT, tr_shade + dT, rh, wind)
    return pd.DataFrame(rec)


def daily_from_hourly(hourly_utci_df: pd.DataFrame) -> pd.DataFrame:
    work = hourly_utci_df.copy()
    work["date"] = pd.to_datetime(work["time"]).dt.normalize()
    agg = {"n_hours": ("tdb", "size"), "t_mean": ("tdb", "mean"), "rh": ("rh", "mean"), "wind": ("wind", "mean")}
    for col in work.columns:
        if col.startswith("utci_"):
            agg[col] = (col, "max")
    daily = work.groupby("date", as_index=False).agg(**{k: v for k, v in agg.items()})
    return daily


def monthly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["month"] = pd.to_datetime(daily["date"]).dt.month
    rows = []
    for month, part in daily.groupby("month"):
        rec = {
            "month": int(month),
            "n_days": int(len(part)),
            "t_mean": float(part["t_mean"].mean()),
            "rh": float(part["rh"].mean()),
            "wind": float(part["wind"].mean()),
        }
        for dT in WARMING:
            suffix = "" if dT == 0 else f"_plus{str(dT).replace('.', '')}"
            open_u = part[f"utci_open{suffix}"].to_numpy(dtype=float)
            shade_u = part[f"utci_shade{suffix}"].to_numpy(dtype=float)
            rec[f"utci{suffix}"] = float(np.nanmean(open_u))
            rec[f"demand{suffix}"] = float(np.nanmean(np.clip(open_u - U0, 0.0, None)))
            rec[f"utci_shade{suffix}"] = float(np.nanmean(shade_u))
            rec[f"demand_shade{suffix}"] = float(np.nanmean(np.clip(shade_u - U0, 0.0, None)))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("month")


def city_climatology(city_key: str) -> pd.DataFrame:
    meta = CITIES[city_key]
    hourly = fetch_openmeteo_hourly(meta["lat"], meta["lon"], city_key)
    lat = hourly.attrs.get("latitude", meta["lat"])
    lon = hourly.attrs.get("longitude", meta["lon"])
    utcis = hourly_utci(hourly, lat, lon)
    daily = daily_from_hourly(utcis)
    monthly = monthly_from_daily(daily)
    monthly["city"] = city_key
    return monthly


def panel_climatology() -> pd.DataFrame:
    return pd.concat([city_climatology(key) for key in PANEL_ORDER], ignore_index=True)
