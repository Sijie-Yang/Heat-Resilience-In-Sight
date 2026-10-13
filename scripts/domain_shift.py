"""Out-of-distribution diagnostics for Singapore-trained VATA, not validation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.config import CSV_DIR, PANEL_ORDER, TABLES

SEG = [
    "seg_road",
    "seg_sidewalk",
    "seg_building",
    "seg_wall",
    "seg_fence",
    "seg_pole",
    "seg_traffic light",
    "seg_traffic sign",
    "seg_vegetation",
    "seg_terrain",
    "seg_sky",
    "seg_person",
    "seg_rider",
    "seg_car",
    "seg_truck",
    "seg_bus",
    "seg_train",
    "seg_motorcycle",
    "seg_bicycle",
]
DET = [
    "det_person",
    "det_bicycle",
    "det_car",
    "det_motorcycle",
    "det_bus",
    "det_truck",
    "det_traffic light",
    "det_fire hydrant",
    "det_stop sign",
    "det_bench",
]
PIX = [
    "Colorfulness",
    "Canny_Edges",
    "Hue_Mean",
    "Hue_Std",
    "Saturation_Mean",
    "Saturation_Std",
    "Lightness_Mean",
    "Lightness_Std",
    "Contrast",
    "Sharpness",
    "Entropy",
    "Image_Variance",
]
SCENE = [
    "scene_downtown",
    "scene_office_building",
    "scene_apartment_building/outdoor",
    "scene_residential_neighborhood",
    "scene_food_court",
    "scene_parking_lot",
    "scene_driveway",
    "scene_highway",
    "scene_plaza",
    "scene_market/outdoor",
    "scene_campus",
    "scene_promenade",
    "scene_field/wild",
    "scene_forest_path",
    "scene_forest/broadleaf",
    "scene_park",
    "scene_construction_site",
    "scene_industrial_area",
]
FEATURES = SEG + DET + PIX + SCENE
REF_CITY = "svi_sg"
N_PCS = 10
LOW = 0.05
HIGH = 4.95


def _load_features(key: str) -> pd.DataFrame:
    cols = FEATURES + ["thermal_affordance"]
    frame = pd.read_csv(CSV_DIR / f"{key}.csv", usecols=lambda c: c in cols)
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{key} missing features: {missing}")
    return frame


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    train_z = (train - mu) / sd
    test_z = (test - mu) / sd
    train_z = np.nan_to_num(train_z, nan=0.0)
    test_z = np.nan_to_num(test_z, nan=0.0)
    return train_z, test_z, mu, sd


def _pca_fit(x: np.ndarray, n_pcs: int) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:n_pcs]
    scores = x @ components.T
    return components, scores


def domain_shift_table() -> pd.DataFrame:
    ref = _load_features(REF_CITY)
    x_ref = ref[FEATURES].to_numpy(dtype=float)
    lo = np.nanpercentile(x_ref, 1, axis=0)
    hi = np.nanpercentile(x_ref, 99, axis=0)
    z_ref, _, _, _ = _standardize(x_ref, x_ref)
    components, scores_ref = _pca_fit(z_ref, N_PCS)
    dist_ref = np.sqrt(np.sum(scores_ref**2, axis=1))
    thresh = float(np.percentile(dist_ref, 95))

    rows = []
    for key in PANEL_ORDER:
        frame = _load_features(key)
        x = frame[FEATURES].to_numpy(dtype=float)
        outside = np.any((x < lo) | (x > hi), axis=1)
        z_ref2, z, _, _ = _standardize(x_ref, x)
        scores = (z - z.mean(axis=0) * 0) @ components.T
        # project with Singapore centering already in _standardize
        dist = np.sqrt(np.sum(scores**2, axis=1))
        v = frame["thermal_affordance"].to_numpy(dtype=float)
        rows.append(
            {
                "city": key,
                "n": int(len(frame)),
                "frac_outside_sg_p1_p99": float(np.mean(outside)),
                "frac_above_sg_pca95": float(np.mean(dist > thresh)),
                "mean_pca_distance": float(np.mean(dist)),
                "frac_near_0": float(np.mean(v <= LOW)),
                "frac_near_5": float(np.mean(v >= HIGH)),
                "v_mean": float(np.nanmean(v)),
                "v_min": float(np.nanmin(v)),
                "v_max": float(np.nanmax(v)),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["pca_distance_p95_singapore"] = thresh
    return table


def write_domain_shift() -> pd.DataFrame:
    TABLES.mkdir(parents=True, exist_ok=True)
    table = domain_shift_table()
    table.to_csv(TABLES / "domain_shift.csv", index=False)
    meta = {
        "reference_city": REF_CITY,
        "n_features": len(FEATURES),
        "n_pcs": N_PCS,
        "envelope": "Singapore 1st–99th percentile on 59 pre-model features",
        "pca_threshold": "Singapore 95th percentile of PCA Euclidean distance",
        "note": "Diagnostics of input shift, not external validation of VATA labels.",
    }
    (TABLES / "domain_shift_meta.json").write_text(json.dumps(meta, indent=2))
    return table
