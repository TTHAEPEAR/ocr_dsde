"""
Fetch Google Trends data and OSM POI density for Kamphaeng Phet tambons.

Combines two external data sources:
1. Google Trends — search interest for party names before election
2. OpenStreetMap — POI counts per tambon as urbanization proxy

Usage:
    python 05_analysis/fetch_trends_and_poi.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from config import REFERENCE_DIR, FIGURES_DIR

OUTPUT_DIR = FIGURES_DIR


# ---------------------------------------------------------------------------
# 1. Google Trends (direct API, no pytrends dependency issues)
# ---------------------------------------------------------------------------

FOCUS_PARTIES = ["กล้าธรรม", "พรรคประชาชน", "ประชาธิปัตย์", "ภูมิใจไทย", "เพื่อไทย"]


def fetch_google_trends_direct(
    keywords: list[str] | None = None,
    timeframe: str = "today 3-m",
    geo: str = "TH",
) -> pd.DataFrame:
    """Fetch Google Trends interest-over-time using the internal API."""
    keywords = keywords or FOCUS_PARTIES

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "th,en-US;q=0.9",
    })

    explore_url = "https://trends.google.com/trends/api/explore"
    req_body = {
        "comparisonItem": [
            {"keyword": kw, "geo": geo, "time": timeframe}
            for kw in keywords
        ],
        "category": 0,
        "property": "",
    }

    logger.info(f"Fetching Google Trends explore for {keywords}...")
    time.sleep(2)  # be polite

    try:
        resp = session.get(explore_url, params={
            "hl": "th", "tz": "-420", "req": json.dumps(req_body),
        }, timeout=30)

        if resp.status_code == 429:
            logger.warning("Google Trends rate limited (429). Wait a few minutes and retry.")
            return pd.DataFrame()

        if resp.status_code != 200:
            logger.warning(f"Google Trends explore failed: {resp.status_code}")
            return pd.DataFrame()

        text = resp.text
        # Strip JSONP prefix
        for prefix in [")]}'\\n", ")]}\\'\\n", ")]}',\\n", ")]}\n", ")]}'\n"]:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        if text.startswith(")]}'"):
            text = text[5:]
        elif text.startswith(")]}"):
            text = text[4:]

        data = json.loads(text)
        widgets = data.get("widgets", [])

        # Find the TIMESERIES widget
        timeline_widget = None
        for w in widgets:
            if w.get("id") == "TIMESERIES":
                timeline_widget = w
                break

        if timeline_widget is None:
            logger.warning("No TIMESERIES widget found in Google Trends response")
            return pd.DataFrame()

        # Fetch actual data
        token = timeline_widget["token"]
        req_data = timeline_widget["request"]

        time.sleep(2)
        multiline_url = "https://trends.google.com/trends/api/widgetdata/multiline"
        resp2 = session.get(multiline_url, params={
            "hl": "th", "tz": "-420",
            "req": json.dumps(req_data),
            "token": token,
        }, timeout=30)

        if resp2.status_code != 200:
            logger.warning(f"Google Trends data request failed: {resp2.status_code}")
            return pd.DataFrame()

        text2 = resp2.text
        for prefix in [")]}'\\n", ")]}\\'\\n", ")]}',\\n", ")]}\n", ")]}'\n"]:
            if text2.startswith(prefix):
                text2 = text2[len(prefix):]
                break
        if text2.startswith(")]}'"):
            text2 = text2[5:]
        elif text2.startswith(")]}"):
            text2 = text2[4:]

        result = json.loads(text2)
        timeline_data = result.get("default", {}).get("timelineData", [])

        if not timeline_data:
            logger.warning("No timeline data in response")
            return pd.DataFrame()

        rows = []
        for point in timeline_data:
            row = {"date": point.get("formattedAxisTime", point.get("formattedTime", ""))}
            for i, kw in enumerate(keywords):
                vals = point.get("value", [])
                row[kw] = vals[i] if i < len(vals) else 0
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Google Trends: {len(df)} time points fetched")
        return df

    except Exception as e:
        logger.warning(f"Google Trends fetch failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. OpenStreetMap POI Density (Overpass API)
# ---------------------------------------------------------------------------

def fetch_osm_poi_for_tambons() -> pd.DataFrame:
    """Fetch POI counts near each tambon centroid using Overpass API."""
    ref = pd.read_csv(REFERENCE_DIR / "tambon_reference.csv")
    ref = ref.drop_duplicates("official_subdistrict")[
        ["official_subdistrict", "tambon_lat", "tambon_lon"]
    ].copy()

    cache_path = REFERENCE_DIR / "tambon_osm_poi.csv"
    if cache_path.exists():
        logger.info(f"Loading cached OSM POI data: {cache_path}")
        return pd.read_csv(cache_path)

    logger.info("Fetching OSM POI counts for each tambon centroid...")
    overpass_url = "https://overpass-api.de/api/interpreter"

    results = []
    for _, row in ref.iterrows():
        tambon = row["official_subdistrict"]
        lat, lon = row["tambon_lat"], row["tambon_lon"]

        if pd.isna(lat) or pd.isna(lon):
            results.append({"official_subdistrict": tambon})
            continue

        # Simple query: count all amenities and shops within 3km
        query = (
            f'[out:json][timeout:30];'
            f'('
            f'  node["amenity"](around:3000,{lat},{lon});'
            f'  node["shop"](around:3000,{lat},{lon});'
            f');'
            f'out count;'
        )
        try:
            logger.info(f"  Querying OSM for {tambon} ({lat}, {lon})...")
            resp = requests.post(overpass_url, data={"data": query}, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                elements = data.get("elements", [])
                total = 0
                if elements:
                    total = int(elements[0].get("tags", {}).get("total", elements[0].get("tags", {}).get("nodes", 0)))
                results.append({
                    "official_subdistrict": tambon,
                    "tambon_lat": lat,
                    "tambon_lon": lon,
                    "osm_poi_count_3km": total,
                })
                logger.info(f"    → {total} POIs found")
            else:
                logger.warning(f"  Overpass returned {resp.status_code} for {tambon}")
                results.append({"official_subdistrict": tambon, "osm_poi_count_3km": np.nan})
            time.sleep(2)  # respect rate limit
        except Exception as e:
            logger.warning(f"  OSM query failed for {tambon}: {e}")
            results.append({"official_subdistrict": tambon, "osm_poi_count_3km": np.nan})
            time.sleep(3)

    out = pd.DataFrame(results)

    # Derive urbanization index from POI count
    if "osm_poi_count_3km" in out.columns:
        valid = out["osm_poi_count_3km"].dropna()
        if len(valid) > 0 and valid.sum() > 0:
            vmin, vmax = valid.min(), valid.max()
            if vmax > vmin:
                out["osm_urbanization_index"] = (out["osm_poi_count_3km"] - vmin) / (vmax - vmin)
            else:
                out["osm_urbanization_index"] = 0.5

            out["osm_urban_class"] = pd.cut(
                out["osm_urbanization_index"].fillna(0),
                bins=[-0.01, 0.25, 0.55, 1.01],
                labels=["rural", "peri-urban", "urban"],
            )

    out.to_csv(cache_path, index=False)
    logger.info(f"Saved OSM POI data: {cache_path}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Fetching external data: Google Trends + OSM POI")
    logger.info("=" * 60)

    # Step 1: Google Trends
    trends_path = OUTPUT_DIR / "google_trends_over_time.csv"
    if trends_path.exists():
        logger.info(f"Google Trends data already cached: {trends_path}")
        trends = pd.read_csv(trends_path)
    else:
        trends = fetch_google_trends_direct()
        if not trends.empty:
            trends.to_csv(trends_path, index=False)
            logger.info(f"Saved Google Trends: {trends_path}")
        else:
            logger.warning("No Google Trends data obtained. Try again later if rate-limited.")

    # Step 2: OSM POI
    poi = fetch_osm_poi_for_tambons()
    has_poi = not poi.empty and "osm_poi_count_3km" in poi.columns
    has_valid_poi = has_poi and poi["osm_poi_count_3km"].notna().any()

    if has_valid_poi:
        logger.info("OSM POI summary:")
        for _, row in poi.iterrows():
            label = row.get("osm_urban_class", "?")
            count = row.get("osm_poi_count_3km", "?")
            logger.info(f"  {row['official_subdistrict']}: {count} POIs → {label}")

    # Step 3: Integrate into multi-source features
    logger.info("Integrating OSM data into multi-source features...")
    features_path = OUTPUT_DIR / "tambon_multisource_features.csv"
    if features_path.exists() and has_valid_poi:
        features = pd.read_csv(features_path)
        osm_cols = [c for c in ["official_subdistrict", "osm_poi_count_3km",
                                "osm_urbanization_index", "osm_urban_class"] if c in poi.columns]
        poi_merge = poi[osm_cols].copy()
        poi_merge = poi_merge.dropna(subset=["osm_poi_count_3km"])
        if not poi_merge.empty:
            # Drop old OSM columns if they exist
            for col in ["osm_poi_count_3km", "osm_urbanization_index", "osm_urban_class"]:
                if col in features.columns:
                    features = features.drop(columns=[col])
            features = features.merge(poi_merge, on="official_subdistrict", how="left")
            features.to_csv(features_path, index=False)
            logger.info(f"Updated multi-source features with OSM data: {features_path}")

    logger.info("External data fetch complete!")


if __name__ == "__main__":
    main()

