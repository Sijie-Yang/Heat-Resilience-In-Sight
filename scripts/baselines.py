"""Simple visual baselines through the same Hong Kong summer calibration, plus capture metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.config import CITIES, CSV_DIR, PANEL_ORDER, TABLES
from scripts.maps import _to_local_m, hex_table

INDICATORS = {
    "vata": lambda f: f["thermal_affordance"],
    "green_view": lambda f: f["seg_vegetation"],
    "shade_proxy": lambda f: 1.0 - f["seg_sky"],  # 1 - visible sky fraction
}


def load(key: str) -> pd.DataFrame:
    cols = ["thermal_affordance", "seg_vegetation", "seg_sky", "longitude", "latitude", "captured_at"]
    return pd.read_csv(CSV_DIR / f"{key}.csv", usecols=cols).dropna(subset=cols[:5])


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(TABLES / "monthly_climate_vata.csv")
    import json
    cal = json.loads((TABLES / "calibration.json").read_text())
    d_ref = cal["d_ref"]
    frames = {k: load(k) for k in PANEL_ORDER}
    means = {k: {n: float(fn(f).mean()) for n, fn in INDICATORS.items()} for k, f in frames.items()}
    rows, hexcorr, cap = [], [], []
    for k, f in frames.items():
        mon = monthly[monthly.city == k].sort_values("month")
        D = mon["demand"].to_numpy()
        rec = {"city": k, "label": CITIES[k]["label"]}
        for n in INDICATORS:
            ref = means["svi_hk"][n]
            need = ref * D / d_ref
            shr = means[k][n] - need
            rec[f"{n}_mean"] = means[k][n]
            rec[f"{n}_shr_annual"] = float(shr.mean())
            rec[f"{n}_shr_rel"] = float(shr.mean() / ref)  # in units of the HK reference mean
            rec[f"{n}_neg_months"] = int((shr < 0).sum())
        rows.append(rec)
        x, y = _to_local_m(f.longitude.to_numpy(), f.latitude.to_numpy())
        hv = {n: hex_table(x, y, fn(f).to_numpy(float)).set_index(["q", "r"])["mean"] for n, fn in INDICATORS.items()}
        h = pd.concat(hv, axis=1).dropna()
        hexcorr.append({"label": CITIES[k]["label"], "n_hex": len(h),
                        "rho_green": spearmanr(h["vata"], h["green_view"]).statistic,
                        "rho_shade": spearmanr(h["vata"], h["shade_proxy"]).statistic,
                        "rho_img_green": spearmanr(f["thermal_affordance"], f["seg_vegetation"]).statistic,
                        "rho_img_shade": spearmanr(f["thermal_affordance"], 1 - f["seg_sky"]).statistic})
        dt = pd.to_datetime(f["captured_at"], format="%Y-%m", errors="coerce")
        mo = dt.dt.month
        warm = (6, 7, 8) if CITIES[k]["hemisphere"] == "N" else (12, 1, 2)
        leafoff = (11, 12, 1, 2, 3) if CITIES[k]["hemisphere"] == "N" else (5, 6, 7, 8, 9)
        cap.append({"label": CITIES[k]["label"], "n": len(f), "dated": float(dt.notna().mean()),
                    "year_p10": int(dt.dt.year.quantile(0.1)), "year_median": int(dt.dt.year.median()),
                    "year_p90": int(dt.dt.year.quantile(0.9)),
                    "warm_quarter": float(mo.isin(warm).mean()), "cool_months": float(mo.isin(leafoff).mean())})
    base = pd.DataFrame(rows)
    for n in INDICATORS:
        base[f"{n}_rank"] = base[f"{n}_shr_rel"].rank(ascending=False).astype(int)
    hexc, capt = pd.DataFrame(hexcorr), pd.DataFrame(cap)
    base.to_csv(TABLES / "baselines_city.csv", index=False)
    hexc.to_csv(TABLES / "baselines_hexcorr.csv", index=False)
    capt.to_csv(TABLES / "capture_metadata.csv", index=False)
    return base, hexc, capt


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    b, h, c = run()
    print(b[["label"] + [f"{n}_{s}" for n in INDICATORS for s in ("mean", "shr_rel", "neg_months", "rank")]].round(3).to_string(index=False))
    print(h.round(3).to_string(index=False))
    print(c.round(3).to_string(index=False))
    print("Spearman city rank VATA vs green:", spearmanr(b.vata_shr_rel, b.green_view_shr_rel).statistic,
          " VATA vs shade:", spearmanr(b.vata_shr_rel, b.shade_proxy_shr_rel).statistic)


def street_disagreement() -> pd.DataFrame:
    """Hottest-quarter share of 800 m hexes where two indicators disagree on shortfall vs surplus."""
    import json
    from scripts.maps import HEX_SIZE, MINCNT, QUARTERS, _xy_to_axial, peak_quarter
    monthly = pd.read_csv(TABLES / "monthly_climate_vata.csv")
    d_ref = json.loads((TABLES / "calibration.json").read_text())["d_ref"]
    frames = {k: load(k) for k in PANEL_ORDER}
    hk = {n: float(fn(frames["svi_hk"]).mean()) for n, fn in INDICATORS.items()}
    rows = []
    for k, f in frames.items():
        D = monthly[(monthly.city == k) & monthly.month.isin(QUARTERS[peak_quarter(k)])]["demand"].mean()
        x, y = _to_local_m(f.longitude.to_numpy(), f.latitude.to_numpy())
        q, r = _xy_to_axial(x, y, HEX_SIZE)
        t = pd.DataFrame({"q": q, "r": r, **{n: fn(f).to_numpy(float) for n, fn in INDICATORS.items()}})
        g = t.groupby(["q", "r"])
        t = g.mean()[g.size() >= MINCNT]
        short = {n: (t[n] < hk[n] * D / d_ref).to_numpy() for n in INDICATORS}
        rows.append({"city": k, "label": CITIES[k]["label"], "n_hex": len(t),
                     **{f"{n}_short": float(v.mean()) for n, v in short.items()},
                     "vata_vs_green": float((short["vata"] != short["green_view"]).mean()),
                     "vata_vs_shade": float((short["vata"] != short["shade_proxy"]).mean()),
                     "green_vs_shade": float((short["green_view"] != short["shade_proxy"]).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "baselines_street_disagreement.csv", index=False)
    return out
