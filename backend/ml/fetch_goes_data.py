"""
fetch_goes_data.py — Fetch real NOAA GOES XRS data and convert to training schema.

Downloads X-ray flux data from NOAA SWPC and generates labeled training data
for solar flare classification.

Labeling Strategy:
1. PRIMARY: Event-based labeling using NOAA SWPC flare event list
   - Labels observations within flare event windows (begin_time to end_time)
   - Includes flare class from event data (e.g., C2.5, M1.0)

2. FALLBACK: Threshold-based labeling (MVP baseline only)
   - Used when flare event API is unavailable
   - Labels based on soft X-ray flux >= 1e-6 W/m² (C-class threshold)
   - NOTE: This causes data leakage and should not be used for production

Data sources:
- X-ray flux: https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json
- Flare events: https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json
"""

import json
import logging
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# NOAA SWPC API endpoints
GOES_XRS_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
GOES_FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"

# Output path for training data
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "training_data.csv"

# Flare threshold for fallback labeling: C1.0 class = 1e-6 W/m²
# NOTE: Threshold-based labeling is only for MVP baseline when event data is unavailable.
# It causes data leakage because soft_xray_flux directly determines the label.
FLARE_THRESHOLD = 1e-6


def fetch_json_api(url: str, description: str) -> list[dict]:
    """
    Fetch JSON data from a NOAA SWPC API endpoint.

    Args:
        url: API endpoint URL.
        description: Human-readable description for logging.

    Returns:
        List of records from the API.

    Raises:
        ConnectionError: If the API is unreachable.
        ValueError: If the response is invalid.
    """
    logger.info("Fetching %s...", description)
    logger.info("URL: %s", url)

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "SolarFlare-ML-Pipeline/1.0"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise ConnectionError(f"API returned status {response.status}")

            data = json.loads(response.read().decode("utf-8"))

    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Failed to connect to NOAA SWPC API: {e}. "
            "Check your internet connection."
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from API: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected list from API, got {type(data).__name__}")

    logger.info("Received %d records from API", len(data))
    return data


def fetch_goes_xrs_data() -> list[dict]:
    """Fetch GOES XRS flux data from NOAA SWPC API."""
    return fetch_json_api(GOES_XRS_URL, "GOES XRS flux data")


def fetch_flare_events() -> Optional[list[dict]]:
    """
    Fetch GOES flare event list from NOAA SWPC API.

    Returns:
        List of flare event records, or None if unavailable.

    Event record fields:
    - begin_time: Flare start time (ISO format)
    - max_time: Flare peak time
    - end_time: Flare end time
    - max_class: Flare class at peak (e.g., "C2.5", "M1.0", "X1.5")
    - max_xrlong: Peak X-ray flux in W/m²
    """
    try:
        return fetch_json_api(GOES_FLARES_URL, "GOES flare events")
    except (ConnectionError, ValueError) as e:
        logger.warning("Could not fetch flare events: %s", e)
        return None


def parse_flare_events(raw_events: list[dict]) -> pd.DataFrame:
    """
    Parse flare event list into a structured DataFrame.

    Args:
        raw_events: List of flare event records from API.

    Returns:
        DataFrame with columns: begin_time, end_time, max_class
    """
    events = []

    for event in raw_events:
        begin_time = event.get("begin_time")
        end_time = event.get("end_time")
        max_class = event.get("max_class", "")

        # Skip incomplete events
        if not begin_time or not end_time:
            continue

        try:
            events.append({
                "begin_time": pd.to_datetime(begin_time),
                "end_time": pd.to_datetime(end_time),
                "max_class": max_class,
            })
        except (ValueError, TypeError):
            continue

    if not events:
        return pd.DataFrame(columns=["begin_time", "end_time", "max_class"])

    df = pd.DataFrame(events)
    df = df.sort_values("begin_time").reset_index(drop=True)

    return df


def parse_goes_data(raw_data: list[dict]) -> pd.DataFrame:
    """
    Parse raw GOES XRS API response into a structured DataFrame.

    The API returns records with fields:
    - time_tag: ISO timestamp
    - satellite: GOES satellite number
    - flux: X-ray flux value in W/m²
    - energy: channel identifier (e.g., "0.05-0.4nm" or "0.1-0.8nm")

    Channel mapping:
    - "0.05-0.4nm" (0.5-4 Å) = XRS-A = hard X-ray
    - "0.1-0.8nm" (1-8 Å) = XRS-B = soft X-ray

    Args:
        raw_data: List of records from GOES API.

    Returns:
        DataFrame with columns: timestamp, soft_xray_flux, hard_xray_flux
    """
    logger.info("Parsing GOES XRS data...")

    # Separate records by energy channel
    soft_records = []  # XRS-B (1-8 Å / 0.1-0.8nm)
    hard_records = []  # XRS-A (0.5-4 Å / 0.05-0.4nm)

    for record in raw_data:
        energy = record.get("energy", "")
        time_tag = record.get("time_tag")
        flux = record.get("flux")

        if time_tag is None or flux is None:
            continue

        # Skip invalid flux values
        try:
            flux_val = float(flux)
            if flux_val < 0:
                continue
        except (TypeError, ValueError):
            continue

        entry = {"timestamp": time_tag, "flux": flux_val}

        # Map energy channels
        if "0.1-0.8" in energy:
            soft_records.append(entry)
        elif "0.05-0.4" in energy:
            hard_records.append(entry)

    logger.info("Found %d soft X-ray records (XRS-B, 1-8 Å)", len(soft_records))
    logger.info("Found %d hard X-ray records (XRS-A, 0.5-4 Å)", len(hard_records))

    if not soft_records or not hard_records:
        raise ValueError("No valid X-ray flux data found in API response")

    # Create DataFrames
    df_soft = pd.DataFrame(soft_records)
    df_soft["timestamp"] = pd.to_datetime(df_soft["timestamp"])
    df_soft = df_soft.rename(columns={"flux": "soft_xray_flux"})

    df_hard = pd.DataFrame(hard_records)
    df_hard["timestamp"] = pd.to_datetime(df_hard["timestamp"])
    df_hard = df_hard.rename(columns={"flux": "hard_xray_flux"})

    # Merge on timestamp (inner join to ensure both channels present)
    df = pd.merge(df_soft, df_hard, on="timestamp", how="inner")

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Merged dataset: %d aligned observations", len(df))

    return df


def apply_event_based_labels(
    df: pd.DataFrame,
    events_df: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """
    Apply flare labels based on actual flare event windows.

    For each observation timestamp:
    - label = 1 if timestamp falls within any [begin_time, end_time] window
    - label = 0 otherwise
    - flare_class = max_class of the matching event (or empty string)

    Args:
        df: DataFrame with timestamp column.
        events_df: DataFrame with begin_time, end_time, max_class columns.

    Returns:
        Tuple of (labeled DataFrame, number of events matched)
    """
    logger.info("Applying event-based labels using %d flare events...", len(events_df))

    df = df.copy()
    df["label"] = 0
    df["flare_class"] = ""

    events_matched = 0

    # Make timestamps timezone-naive for comparison if needed
    df_timestamps = df["timestamp"].dt.tz_localize(None) if df["timestamp"].dt.tz is not None else df["timestamp"]

    for _, event in events_df.iterrows():
        begin = event["begin_time"]
        end = event["end_time"]
        flare_class = event["max_class"]

        # Make event times timezone-naive for comparison
        if begin.tz is not None:
            begin = begin.tz_localize(None)
        if end.tz is not None:
            end = end.tz_localize(None)

        # Find observations within this flare window
        mask = (df_timestamps >= begin) & (df_timestamps <= end)
        matches = mask.sum()

        if matches > 0:
            events_matched += 1
            df.loc[mask, "label"] = 1
            # Only set flare_class if not already set (keep highest class)
            empty_class_mask = mask & (df["flare_class"] == "")
            df.loc[empty_class_mask, "flare_class"] = flare_class

    labeled_count = df["label"].sum()
    logger.info("Labeled %d observations as flare (from %d events)", labeled_count, events_matched)

    return df, events_matched


def apply_threshold_labels(
    df: pd.DataFrame,
    threshold: float = FLARE_THRESHOLD
) -> pd.DataFrame:
    """
    Apply flare labels based on soft X-ray flux threshold (FALLBACK ONLY).

    WARNING: This method causes data leakage because soft_xray_flux is both
    an input feature and determines the label. Use only when event-based
    labeling is unavailable.

    GOES flare classification:
    - A-class: < 1e-7 W/m²
    - B-class: 1e-7 to 1e-6 W/m²
    - C-class: 1e-6 to 1e-5 W/m² (threshold for label=1)
    - M-class: 1e-5 to 1e-4 W/m²
    - X-class: >= 1e-4 W/m²

    Args:
        df: DataFrame with soft_xray_flux column.
        threshold: Flux threshold for flare label (default: 1e-6 for C-class).

    Returns:
        DataFrame with added 'label' and 'flare_class' columns.
    """
    logger.warning(
        "Using threshold-based labeling (FALLBACK). "
        "This causes data leakage and is only for MVP baseline."
    )

    df = df.copy()
    df["label"] = (df["soft_xray_flux"] >= threshold).astype(int)

    # Derive flare class from flux level
    def classify_flux(flux: float) -> str:
        if flux >= 1e-4:
            return "X"
        elif flux >= 1e-5:
            return "M"
        elif flux >= 1e-6:
            return "C"
        elif flux >= 1e-7:
            return "B"
        else:
            return "A"

    df["flare_class"] = df.apply(
        lambda row: classify_flux(row["soft_xray_flux"]) if row["label"] == 1 else "",
        axis=1
    )

    return df


def print_statistics(df: pd.DataFrame, labeling_method: str, events_count: int = 0) -> None:
    """Print summary statistics of the dataset."""

    print("\n" + "=" * 70)
    print("GOES XRS DATA INGESTION SUMMARY")
    print("=" * 70)

    print(f"\nLabeling method:    {labeling_method}")
    if events_count > 0:
        print(f"Flare events used:  {events_count}")

    print(f"\nTotal rows:         {len(df):,}")
    print(f"Timestamp range:    {df['timestamp'].min()} to {df['timestamp'].max()}")

    duration = df["timestamp"].max() - df["timestamp"].min()
    print(f"Duration:           {duration}")

    flare_count = df["label"].sum()
    no_flare_count = len(df) - flare_count

    print(f"\nFlare labels (1):   {flare_count:,} ({100 * flare_count / len(df):.1f}%)")
    print(f"No-flare labels (0): {no_flare_count:,} ({100 * no_flare_count / len(df):.1f}%)")

    # Show flare class distribution
    if "flare_class" in df.columns:
        flare_classes = df[df["label"] == 1]["flare_class"].value_counts()
        if len(flare_classes) > 0:
            print("\nFlare class distribution:")
            for cls, count in flare_classes.items():
                print(f"  {cls}: {count}")

    print(f"\nSoft X-ray flux range: {df['soft_xray_flux'].min():.2e} to {df['soft_xray_flux'].max():.2e} W/m²")
    print(f"Hard X-ray flux range: {df['hard_xray_flux'].min():.2e} to {df['hard_xray_flux'].max():.2e} W/m²")

    print("\n" + "-" * 70)
    print("SAMPLE ROWS (first 5):")
    print("-" * 70)
    print(df.head().to_string(index=False))

    print("\n" + "-" * 70)
    print("SAMPLE ROWS (last 5):")
    print("-" * 70)
    print(df.tail().to_string(index=False))

    # Show flare examples
    flare_samples = df[df["label"] == 1].head(5)
    if len(flare_samples) > 0:
        print("\n" + "-" * 70)
        print("SAMPLE FLARE EVENTS (label=1):")
        print("-" * 70)
        print(flare_samples.to_string(index=False))

    print("\n" + "=" * 70)


def main() -> int:
    """
    Main entry point for GOES data fetching.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        # Fetch X-ray flux data
        raw_xrs_data = fetch_goes_xrs_data()

        # Parse and align channels
        df = parse_goes_data(raw_xrs_data)

        if len(df) == 0:
            logger.error("No data available after parsing")
            return 1

        # Try to fetch flare events for event-based labeling
        raw_flare_events = fetch_flare_events()

        labeling_method = ""
        events_matched = 0

        if raw_flare_events and len(raw_flare_events) > 0:
            # PRIMARY: Event-based labeling
            events_df = parse_flare_events(raw_flare_events)

            if len(events_df) > 0:
                df, events_matched = apply_event_based_labels(df, events_df)
                labeling_method = "EVENT-BASED (NOAA SWPC flare event list)"
                logger.info("Successfully applied event-based labels")
            else:
                logger.warning("No valid flare events parsed, falling back to threshold")
                df = apply_threshold_labels(df)
                labeling_method = "THRESHOLD-BASED (fallback, causes data leakage)"
        else:
            # FALLBACK: Threshold-based labeling
            logger.warning("Flare event data unavailable, using threshold-based labeling")
            df = apply_threshold_labels(df)
            labeling_method = "THRESHOLD-BASED (fallback, causes data leakage)"

        # Ensure output directory exists
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV with new schema: timestamp, soft_xray_flux, hard_xray_flux, label, flare_class
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info("Saved training data to %s", OUTPUT_CSV)

        # Print statistics
        print_statistics(df, labeling_method, events_matched)

        print(f"\nOutput file: {OUTPUT_CSV}")
        print("Output schema: timestamp, soft_xray_flux, hard_xray_flux, label, flare_class")
        print("\nReady for model training with: python ml/train_model.py")

        return 0

    except ConnectionError as e:
        logger.error("Connection error: %s", e)
        print("\nFailed to fetch data. Please check your internet connection.")
        return 1

    except ValueError as e:
        logger.error("Data error: %s", e)
        return 1

    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
