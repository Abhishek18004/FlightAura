from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


def get_project_path(environment_variable: str, default: str) -> Path:
    """Resolve a project-relative path from an environment variable."""
    raw_path = os.getenv(environment_variable, default)
    path = Path(raw_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


DATASET_PATH = get_project_path(
    "FLIGHTAURA_DATASET",
    "dataset/processed/airline_reviews_unique.csv",
)

DL_DIR = get_project_path(
    "FLIGHTAURA_DL_DIR",
    "models/deep_learning",
)

CLASSICAL_DIR = get_project_path(
    "FLIGHTAURA_CLASSICAL_DIR",
    "models/classical",
)

MAX_EVIDENCE_REVIEWS = int(os.getenv("MAX_EVIDENCE_REVIEWS", "6"))
MIN_RETRIEVAL_SIMILARITY = float(
    os.getenv("MIN_RETRIEVAL_SIMILARITY", "0.03")
)


# ---------------------------------------------------------------------
# TEXT UTILITIES
# ---------------------------------------------------------------------

def clean_text(value: Any) -> str:
    """Convert a value into clean, whitespace-normalized text."""
    if value is None or pd.isna(value):
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_column_name(value: str) -> str:
    """Normalize a column name for flexible matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_airline_name(value: str) -> str:
    """Normalize airline names for comparisons."""
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None when conversion fails."""
    if value is None or pd.isna(value):
        return None

    try:
        result = float(str(value).replace(",", "").strip())

        if result != result:
            return None

        return result

    except (TypeError, ValueError):
        return None


def safe_string(value: Any, default: str = "Not available") -> str:
    """Convert a value to a display-safe string."""
    text = clean_text(value)

    return text if text else default


def split_sentences(text: str) -> list[str]:
    """Split review text into readable sentence-like segments."""
    text = clean_text(text)

    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)

    return [part.strip() for part in parts if part.strip()]


# ---------------------------------------------------------------------
# DATASET COLUMN RESOLUTION
# ---------------------------------------------------------------------

COLUMN_CANDIDATES = {
    "airline": [
        "airline",
        "airline_name",
        "carrier",
    ],
    "review_text": [
        "review_text_clean",
        "review_text",
        "reviews_clean",
        "reviews",
        "review",
        "cleaned_review",
        "text",
    ],
    "review_date": [
        "review_date",
        "date",
        "date_published",
    ],
    "month_flown": [
        "month_flown",
        "date_flown",
        "flight_date",
        "month_of_travel",
    ],
    "overall_rating": [
        "overall_rating",
        "overall",
        "rating",
        "overall_score",
    ],
    "recommended": [
        "recommended_binary",
        "recommended",
        "recommendation",
    ],
    "name": [
        "name",
        "reviewer_name",
        "author",
    ],
    "verified": [
        "verified",
        "verification_status",
    ],
    "aircraft": [
        "aircraft",
        "plane",
    ],
    "traveller_type": [
        "type_of_traveller",
        "traveller_type",
        "travel_type",
    ],
    "seat_type": [
        "seat_type",
        "cabin_class",
        "class",
    ],
    "route": [
        "route",
        "flight_route",
    ],
    "seat_comfort": [
        "seat_comfort",
        "seat_comfort_rating",
    ],
    "cabin_staff": [
        "cabin_staff_service",
        "cabin_crew_service",
        "staff_service",
    ],
    "food": [
        "food_beverages",
        "food_and_beverages",
        "food",
        "meal_rating",
    ],
    "ground_service": [
        "ground_service",
        "airport_service",
    ],
    "wifi": [
        "wifi_connectivity",
        "wifi_and_connectivity",
        "wifi",
    ],
    "entertainment": [
        "inflight_entertainment",
        "in_flight_entertainment",
        "entertainment",
    ],
    "value": [
        "value_for_money",
        "value",
    ],
}


def resolve_dataset_columns(
    dataframe: pd.DataFrame,
) -> dict[str, str | None]:
    """
    Match canonical application fields to actual dataframe columns.

    Matching ignores spaces, underscores, punctuation and capitalization.
    """
    normalized_lookup = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    resolved: dict[str, str | None] = {}

    for canonical_name, candidates in COLUMN_CANDIDATES.items():
        resolved[canonical_name] = None

        for candidate in candidates:
            normalized_candidate = normalize_column_name(candidate)

            if normalized_candidate in normalized_lookup:
                resolved[canonical_name] = normalized_lookup[
                    normalized_candidate
                ]
                break

    return resolved


def load_review_dataset(
    dataset_path: Path = DATASET_PATH,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Load the review dataset and resolve its important columns."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Review dataset was not found: {dataset_path}"
        )

    dataframe = pd.read_csv(dataset_path, low_memory=False)

    if dataframe.empty:
        raise ValueError("The review dataset contains no rows.")

    columns = resolve_dataset_columns(dataframe)

    required = ["airline", "review_text"]
    missing = [
        field
        for field in required
        if columns.get(field) is None
    ]

    if missing:
        raise ValueError(
            "Could not identify required dataset columns: "
            f"{missing}. Available columns: {list(dataframe.columns)}"
        )

    # Remove rows that cannot be used for claim investigation.
    airline_column = columns["airline"]
    review_column = columns["review_text"]

    dataframe[airline_column] = dataframe[airline_column].apply(clean_text)
    dataframe[review_column] = dataframe[review_column].apply(clean_text)

    dataframe = dataframe[
        (dataframe[airline_column] != "")
        & (dataframe[review_column] != "")
    ].copy()

    dataframe.reset_index(drop=True, inplace=True)

    return dataframe, columns


# ---------------------------------------------------------------------
# DISPLAY / LABEL UTILITIES
# ---------------------------------------------------------------------

def normalize_recommendation(value: Any) -> bool | None:
    """Convert common recommendation values into True/False."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"yes", "y", "true", "1", "recommended"}:
        return True

    if text in {"no", "n", "false", "0", "not recommended"}:
        return False

    number = safe_float(value)

    if number is not None:
        if number == 1:
            return True
        if number == 0:
            return False

    return None


def json_safe(value: Any) -> Any:
    """Convert common pandas and NumPy-style values into JSON-safe values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)