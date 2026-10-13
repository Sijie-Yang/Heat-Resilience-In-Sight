"""Eight-city VATA panel: load aligned TCIS scores and summarise supply."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.config import CITIES, CSV_DIR, PANEL_ORDER, TABLES


def load_vata_values(path) -> np.ndarray:
    return pd.read_csv(path, usecols=["thermal_affordance"])["thermal_affordance"].to_numpy(dtype=float)


def city_vata(path) -> dict:
    frame = pd.read_csv(path, usecols=["thermal_affordance"])
    s = frame["thermal_affordance"].astype(float)
    return {
        "n": int(s.notna().sum()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
    }


def summarise_panel(write: bool = True) -> pd.DataFrame:
    rows = []
    for key in PANEL_ORDER:
        meta = CITIES[key]
        path = CSV_DIR / f"{key}.csv"
        rec = city_vata(path)
        rec.update({"city": key, **{k: meta[k] for k in ("label", "short", "koppen")}})
        rows.append(rec)
    stats = pd.DataFrame(rows).set_index("city")
    if write:
        TABLES.mkdir(parents=True, exist_ok=True)
        stats.to_csv(TABLES / "city_vata_stats.csv")
    return stats


def load_all_vata() -> dict[str, np.ndarray]:
    return {key: load_vata_values(CSV_DIR / f"{key}.csv") for key in PANEL_ORDER}
