"""Public-domain IEM METAR weather, joined strictly at or before each UTC hour."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/external/iem_metar"
OUTPUT = ROOT / "data/processed/weather_hourly.parquet"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
FIELDS = ["tmpf", "dwpf", "drct", "sknt", "alti", "vsby", "gust", "wxcodes"]
PERIODS = [("2024-12-31", "2026-02-01"), ("2026-06-30", "2026-08-01")]
LICENSE_URL = "https://mesonet.agron.iastate.edu/disclaimer.php"
LIMIT = 100_000_000


def fetch():
    CACHE.mkdir(parents=True, exist_ok=True)
    frames, sources = [], []
    for start, end in PERIODS:
        params = dict(station=AIRPORTS, data=FIELDS, sts=start + "T00:00Z", ets=end + "T00:00Z",
                      tz="Etc/UTC", format="onlycomma", latlon="no", elev="no", missing="M",
                      trace="T", report_type=[3, 4])
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urlencode(params, doseq=True)
        path = CACHE / f"metar_{start}_{end}.csv"
        if not path.exists():
            print(f"Downloading {start} to {end} for {len(AIRPORTS)} airports", flush=True)
            total = sum(p.stat().st_size for p in CACHE.glob("*.csv"))
            temporary = path.with_suffix(".part")
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(65536):
                    total += len(chunk)
                    if total > LIMIT:
                        raise ValueError("Weather download exceeds 100 MB limit")
                    output.write(chunk)
            temporary.replace(path)
        frame = pd.read_csv(path, na_values=["M"], dtype={"station": str, "wxcodes": str})
        assert {"station", "valid", *FIELDS}.issubset(frame.columns), path
        assert set(frame.station.unique()) == set(AIRPORTS), (path, frame.station.unique())
        frames.append(frame)
        sources.append(dict(url=url, path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), observations=len(frame)))
        print(f"Cached {path.name}: {len(frame):,} observations, {path.stat().st_size / 1e6:.2f} MB", flush=True)
    return pd.concat(frames, ignore_index=True), sources


def make_features(raw):
    raw["observed_at"] = pd.to_datetime(raw.valid, utc=True)
    raw = raw.sort_values(["station", "observed_at"]).drop_duplicates(["station", "observed_at"], keep="last")
    obs = raw[["station", "observed_at"]].copy()
    for source, name, scale, offset in [
        ("tmpf", "temp_c", 5 / 9, -32), ("dwpf", "dewpoint_c", 5 / 9, -32),
        ("sknt", "wind_knots", 1, 0), ("gust", "gust_knots", 1, 0),
        ("drct", "wind_deg", 1, 0), ("vsby", "visibility_km", 1.609344, 0),
        ("alti", "pressure_hpa", 33.8638866667, 0)]:
        obs["wx_" + name] = (pd.to_numeric(raw[source], errors="coerce") + offset) * scale
    codes = raw.wxcodes.fillna("")
    for name, pattern in {"snow": "SN|SG|PL|IC", "freezing": "FZ", "rain": "RA|DZ",
                          "fog": "FG|BR", "thunder": "TS", "hail": "GR|GS"}.items():
        obs["wx_reported_" + name] = codes.str.contains(pattern, regex=True).astype(float)
    obs["wx_dewpoint_spread_c"] = obs.wx_temp_c - obs.wx_dewpoint_c
    obs["wx_freezing_temp"] = obs.wx_temp_c.le(0).where(obs.wx_temp_c.notna()).astype(float)
    # A continuous cold-and-humid interaction; it is not an observed deicing flag.
    obs["wx_cold_humid"] = (3 - obs.wx_temp_c).clip(0) * (3 - obs.wx_dewpoint_spread_c).clip(0)
    radians = np.deg2rad(obs.wx_wind_deg)
    obs["wx_wind_north_knots"] = obs.wx_wind_knots * np.cos(radians)
    obs["wx_wind_east_knots"] = obs.wx_wind_knots * np.sin(radians)
    hours = pd.date_range("2025-01-01", "2026-02-01", inclusive="left", freq="h", tz="UTC").append(
        pd.date_range("2026-07-01", "2026-08-01", inclusive="left", freq="h", tz="UTC"))
    results = []
    for airport in AIRPORTS:
        right = obs[obs.station.eq(airport)].drop(columns="station").sort_values("observed_at")
        matched = pd.merge_asof(pd.DataFrame({"hour": hours}), right, left_on="hour", right_on="observed_at",
                                direction="backward", tolerance=pd.Timedelta(hours=3))
        matched["wx_age_minutes"] = (matched.hour - matched.observed_at).dt.total_seconds() / 60
        assert matched.wx_age_minutes.dropna().between(0, 180).all(), "Weather may not come from the future"
        matched["ADEP_mvt"] = airport
        results.append(matched.drop(columns="observed_at"))
    result = pd.concat(results, ignore_index=True)
    columns = [c for c in result if c.startswith("wx_")]
    result[columns] = result[columns].astype("float32")
    assert not result.duplicated(["ADEP_mvt", "hour"]).any()
    return result


def self_check():
    hours = pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T01:00Z"])
    observations = pd.to_datetime(["2024-12-31T23:50Z", "2025-01-01T00:20Z"])
    result = pd.merge_asof(pd.DataFrame({"hour": hours}),
        pd.DataFrame({"observed_at": observations, "temp": [1, 99]}),
        left_on="hour", right_on="observed_at", direction="backward")
    assert result.temp.tolist() == [1, 99], "An observation after the hour must not leak backward"


def main():
    self_check()
    raw, sources = fetch()
    features = make_features(raw)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT, index=False)
    metadata = dict(retrieved_at=datetime.now(timezone.utc).isoformat(),
        source="Iowa Environmental Mesonet ASOS/METAR archive", license_url=LICENSE_URL,
        rights="IEM states its website materials are public domain and freely usable for lawful purposes.",
        api_documentation="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=",
        sources=sources, total_download_bytes=sum(s["bytes"] for s in sources), hourly_rows=len(features),
        missing_temperature_by_airport=features.groupby("ADEP_mvt").wx_temp_c.apply(lambda s: float(s.isna().mean())).to_dict(),
        missing_by_column=features.filter(regex="^wx_").isna().mean().to_dict())
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}: {len(features):,} airport-hours", flush=True)
    print(json.dumps(metadata["missing_temperature_by_airport"], indent=2), flush=True)


if __name__ == "__main__":
    main()
