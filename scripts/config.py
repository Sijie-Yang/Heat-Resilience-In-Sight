"""Shared paths and city metadata for Heat Resilience in Sight."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "svi_vata"
CACHE = ROOT / "data" / "climate_cache"
CACHE_HOURLY = CACHE / "hourly_v1"
TABLES = ROOT / "data" / "outputs"
FIGS = ROOT / "figures"

U0 = 26.0  # UTCI moderate heat-stress onset (Bröde et al. 2012)
WIND_MIN = 0.5
WARMING = (0.0, 1.5, 2.5)
MONTHS = list(range(1, 13))
MONTH_LAB = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
DAYTIME_HOURS = tuple(range(10, 18))  # 10:00–17:00 local inclusive
HOURLY_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "shortwave_radiation",
)
# Standardised open-sun pedestrian, independent of local street geometry.
SOLAR_ASSUMPTIONS = {
    "posture": "standing",
    "asw": 0.7,
    "floor_reflectance": 0.2,
    "f_svv": 1.0,
    "f_bes": 0.50,
    "sol_transmittance": 1.0,
    "sharp": 90.0,
    "hours_local": "10-17",
    "daily_utci": "maximum of daytime hours",
}

CITIES = {
    "svi_sg": {
        "label": "Singapore",
        "short": "Singapore",
        "lat": 1.3521,
        "lon": 103.8198,
        "koppen": "Af",
        "hemisphere": "N",
    },
    "svi_jh": {
        "label": "Johannesburg",
        "short": "Johannesburg",
        "lat": -26.2041,
        "lon": 28.0473,
        "koppen": "Cwb",
        "hemisphere": "S",
    },
    "svi_hk": {
        "label": "Hong Kong",
        "short": "Hong Kong",
        "lat": 22.3193,
        "lon": 114.1694,
        "koppen": "Cwa",
        "hemisphere": "N",
    },
    "svi_tokyo": {
        "label": "Tokyo",
        "short": "Tokyo",
        "lat": 35.6762,
        "lon": 139.6503,
        "koppen": "Cfa",
        "hemisphere": "N",
    },
    "svi_ny": {
        "label": "New York",
        "short": "New York",
        "lat": 40.7128,
        "lon": -74.0060,
        "koppen": "Cfa",
        "hemisphere": "N",
    },
    "svi_melbourne": {
        "label": "Melbourne",
        "short": "Melbourne",
        "lat": -37.8136,
        "lon": 144.9631,
        "koppen": "Cfb",
        "hemisphere": "S",
    },
    "svi_rio": {
        "label": "Rio de Janeiro",
        "short": "Rio",
        "lat": -22.9068,
        "lon": -43.1729,
        "koppen": "Aw",
        "hemisphere": "S",
    },
    "svi_capetown": {
        "label": "Cape Town",
        "short": "Cape Town",
        "lat": -33.9249,
        "lon": 18.4241,
        "koppen": "Csb",
        "hemisphere": "S",
    },
}

CALIBRATE_KEY = "svi_hk"
CALIBRATE_MONTHS = (6, 7, 8)
PANEL_ORDER = [
    "svi_sg",
    "svi_jh",
    "svi_hk",
    "svi_tokyo",
    "svi_ny",
    "svi_melbourne",
    "svi_rio",
    "svi_capetown",
]
# Paper Figs 1–2 and Table 1: annual SHR descending
SHR_RANK_ORDER = [
    "svi_melbourne",
    "svi_ny",
    "svi_tokyo",
    "svi_hk",
    "svi_capetown",
    "svi_jh",
    "svi_sg",
    "svi_rio",
]
