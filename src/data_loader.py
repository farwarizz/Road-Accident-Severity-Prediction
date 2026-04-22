import pandas as pd

TARGET_CANDIDATES = [
    "Accident_Severity",
    "accident_severity",
    "Severity",
    "severity",
]

LAT_CANDIDATES = ["Latitude", "latitude", "lat"]
LON_CANDIDATES = ["Longitude", "longitude", "lon"]


def load_data(source):
    if isinstance(source, str):
        return pd.read_csv(source)
    return pd.read_csv(source)


def detect_target_column(df):
    matches = []
    for col in df.columns:
        if col in TARGET_CANDIDATES and df[col].nunique() > 1:
            matches.append(col)
    return matches


def detect_lat_lon(df):
    lat = next((c for c in df.columns if c in LAT_CANDIDATES), None)
    lon = next((c for c in df.columns if c in LON_CANDIDATES), None)
    return lat, lon
