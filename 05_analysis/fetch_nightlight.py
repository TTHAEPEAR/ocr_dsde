"""
Download VIIRS nighttime light data and extract radiance at tambon coordinates.

Uses the EOG (Earth Observation Group) public HTTPS server to download
a VIIRS DNB annual composite tile covering Kamphaeng Phet (tile 75N060E).
Then samples the radiance raster at each tambon centroid.

No Google Earth Engine authentication required.

Usage:
    python 05_analysis/fetch_nightlight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from config import REFERENCE_DIR, FIGURES_DIR

VIIRS_DIR = REFERENCE_DIR / "viirs"
VIIRS_DIR.mkdir(parents=True, exist_ok=True)

# The EOG VIIRS annual composite (v2.2) — tile covering Thailand
# Kamphaeng Phet is around 16.5°N, 99.5°E → tile 75N060E covers it
# We use the average radiance (avg_rade9h) product
VIIRS_TILE_URL = (
    "https://eogdata.mines.edu/nighttime_light/annual/v22/2024/average/"
    "VNL_v22_npp-j01_2024_global_vcmslcfg_c202502131200.average.tif"
)

# Alternative: use a smaller regional subset if the global file is too large
# For the project, we'll sample directly using rasterio's windowed reading
# to avoid downloading the full ~3GB global file.


def sample_viirs_from_url(
    lat_lon_pairs: list[tuple[float, float]],
    url: str = VIIRS_TILE_URL,
) -> list[float]:
    """Sample VIIRS radiance at given lat/lon pairs using HTTP range requests.

    Uses rasterio's GDAL virtual filesystem (vsicurl) to read only the
    pixels we need from the remote GeoTIFF without downloading the whole file.
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        logger.error("rasterio not installed — run `pip install rasterio`")
        return [np.nan] * len(lat_lon_pairs)

    # Use GDAL's vsicurl for HTTP range-request based reading
    vsicurl_url = f"/vsicurl/{url}"

    try:
        logger.info(f"Opening VIIRS raster via HTTP range requests: {url[:80]}...")
        with rasterio.open(vsicurl_url) as src:
            logger.info(f"Raster CRS: {src.crs}, Shape: {src.shape}, Bands: {src.count}")
            values = []
            for lat, lon in lat_lon_pairs:
                try:
                    row, col = rowcol(src.transform, lon, lat)
                    # Read a small window around the point (3x3 for averaging)
                    window = rasterio.windows.Window(
                        max(0, col - 1), max(0, row - 1), 3, 3
                    )
                    data = src.read(1, window=window)
                    # Filter out fill values (negative or very high values)
                    valid = data[(data > 0) & (data < 1000)]
                    val = float(np.nanmean(valid)) if len(valid) > 0 else np.nan
                    values.append(val)
                except Exception as e:
                    logger.warning(f"Failed to sample ({lat}, {lon}): {e}")
                    values.append(np.nan)
            return values
    except Exception as e:
        logger.warning(f"Could not open VIIRS raster: {e}")
        logger.info("Falling back to alternative VIIRS source...")
        return try_alternative_viirs(lat_lon_pairs)


def try_alternative_viirs(lat_lon_pairs: list[tuple[float, float]]) -> list[float]:
    """Try alternative VIIRS tile sources (smaller files)."""
    # Try the 15-arc-second tiled version which is much smaller
    # Thailand falls in tile h03v05 (MODIS-style tiling) or the simpler lat/lon tiles
    alt_urls = [
        # Annual composite 2023 tile for Southeast Asia
        "https://eogdata.mines.edu/nighttime_light/annual/v22/2023/average_masked/"
        "VNL_v22_npp-j01_2023_global_vcmslcfg_c202503201300.average_masked.tif",
        # 2022 version
        "https://eogdata.mines.edu/nighttime_light/annual/v22/2022/average_masked/"
        "VNL_v22_npp-j01_2022_global_vcmslcfg_c202503201300.average_masked.tif",
    ]

    import rasterio
    from rasterio.transform import rowcol

    for url in alt_urls:
        try:
            vsicurl_url = f"/vsicurl/{url}"
            logger.info(f"Trying alternative: {url[-60:]}...")
            with rasterio.open(vsicurl_url) as src:
                values = []
                for lat, lon in lat_lon_pairs:
                    try:
                        row, col = rowcol(src.transform, lon, lat)
                        window = rasterio.windows.Window(max(0, col - 1), max(0, row - 1), 3, 3)
                        data = src.read(1, window=window)
                        valid = data[(data > 0) & (data < 1000)]
                        val = float(np.nanmean(valid)) if len(valid) > 0 else np.nan
                        values.append(val)
                    except Exception:
                        values.append(np.nan)
                return values
        except Exception as e:
            logger.warning(f"Alternative failed: {e}")
            continue

    logger.warning("All VIIRS sources failed — returning NaN")
    return [np.nan] * len(lat_lon_pairs)


def fetch_viirs_for_tambons() -> pd.DataFrame:
    """Fetch VIIRS radiance values for all tambons in the reference file."""
    ref = pd.read_csv(REFERENCE_DIR / "tambon_reference.csv")
    ref = ref.drop_duplicates("official_subdistrict")[
        ["official_subdistrict", "tambon_lat", "tambon_lon"]
    ].copy()

    # Check for cached result
    cache_path = REFERENCE_DIR / "tambon_viirs_radiance.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        logger.info(f"Loaded cached VIIRS data: {cache_path}")
        return cached

    lat_lon_pairs = list(zip(ref["tambon_lat"], ref["tambon_lon"]))
    logger.info(f"Sampling VIIRS radiance at {len(lat_lon_pairs)} tambon centroids...")

    radiance_values = sample_viirs_from_url(lat_lon_pairs)

    ref["viirs_mean_radiance"] = radiance_values

    # Derive urbanization index from VIIRS if we got valid data
    valid_count = ref["viirs_mean_radiance"].notna().sum()
    if valid_count > 0:
        logger.info(f"Got valid VIIRS values for {valid_count}/{len(ref)} tambons")
        vmin = ref["viirs_mean_radiance"].min()
        vmax = ref["viirs_mean_radiance"].max()
        if vmax > vmin:
            ref["viirs_urbanization_index"] = (ref["viirs_mean_radiance"] - vmin) / (vmax - vmin)
        else:
            ref["viirs_urbanization_index"] = 0.5

        # Classify
        ref["viirs_urban_class"] = pd.cut(
            ref["viirs_urbanization_index"],
            bins=[-0.01, 0.33, 0.66, 1.01],
            labels=["rural", "peri-urban", "urban"],
        )
    else:
        logger.warning("No valid VIIRS data — urbanization index not computed from satellite")

    ref.to_csv(cache_path, index=False)
    logger.info(f"Saved VIIRS radiance data: {cache_path}")
    return ref


def main():
    logger.info("=" * 50)
    logger.info("Fetching VIIRS nighttime light data")
    logger.info("=" * 50)

    result = fetch_viirs_for_tambons()

    if "viirs_mean_radiance" in result.columns:
        print("\n=== VIIRS Night Light Results ===")
        print(result[["official_subdistrict", "viirs_mean_radiance",
                       "viirs_urbanization_index", "viirs_urban_class"]].to_string())
    else:
        print("No VIIRS data obtained.")

    return result


if __name__ == "__main__":
    main()
