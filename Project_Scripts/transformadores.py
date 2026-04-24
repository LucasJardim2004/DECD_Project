"""
Utilities for CVD data preparation.

This module centralizes categorical and numeric transformations used to build
fully categorical and fully numeric dataset versions, plus normalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


AGE_CATEGORY_GROUPS: dict[str, str] = {
    "18-24": "Young Adult",
    "25-29": "Young Adult",
    "30-34": "Young Adult",
    "35-39": "Young Adult",
    "40-44": "Adult",
    "45-49": "Adult",
    "50-54": "Adult",
    "55-59": "Adult",
    "60-64": "Adult",
    "65-69": "Senior",
    "70-74": "Senior",
    "75-79": "Senior",
    "80+": "Senior",
}


ORDINAL_ORDER: dict[str, list[str]] = {
    "General_Health": ["Poor", "Fair", "Good", "Very Good", "Excellent"],
    "Checkup": [
        "Never",
        "5 or more years ago",
        "Within the past 5 years",
        "Within the past 2 years",
        "Within the past year",
    ],
    "Age_Category": ["Young Adult", "Adult", "Senior"],
}

BINARY_COLS: list[str] = [
    "Exercise",
    "Heart_Disease",
    "Skin_Cancer",
    "Other_Cancer",
    "Depression",
    "Arthritis",
    "Sex",
    "Smoking_History",
]

CONTINUOUS_COLS: list[str] = [
    "Height_(cm)",
    "Weight_(kg)",
    "BMI",
    "Alcohol_Consumption",
    "Fruit_Consumption",
    "Green_Vegetables_Consumption",
    "FriedPotato_Consumption",
]


def consolidate_diabetes(series: pd.Series) -> pd.Series:
    """
    Consolidate 4 Diabetes values into binary labels using the chosen policy.

    Mapping:
    - No -> No
    - No, pre-diabetes or borderline diabetes -> No
    - Yes, but female told only during pregnancy -> Yes
    - Yes -> Yes
    """
    mapping = {
        "No": "No",
        "No, pre-diabetes or borderline diabetes": "No",
        "Yes, but female told only during pregnancy": "Yes",
        "Yes": "Yes",
    }
    return series.map(mapping)


def group_age_category(series: pd.Series) -> pd.Series:
    """Group detailed age bins into three ordered groups."""
    return series.map(AGE_CATEGORY_GROUPS)


def discretize_bmi(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Normal"
    if value < 30:
        return "Overweight"
    if value < 35:
        return "Obesity I"
    return "Obesity II+"


def discretize_height(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    if value < 160:
        return "Short"
    if value < 170:
        return "Medium"
    if value < 180:
        return "Tall"
    return "Very Tall"


def discretize_weight(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    if value < 60:
        return "Light"
    if value < 75:
        return "Moderate"
    if value < 90:
        return "Heavy"
    return "Very Heavy"


def discretize_consumption(value: float, consumption_type: str = "generic") -> str | float:
    if pd.isna(value):
        return np.nan

    if consumption_type == "alcohol":
        if value == 0:
            return "Never"
        if value <= 5:
            return "Low"
        if value <= 10:
            return "Moderate"
        return "High"
    
    if consumption_type == "fried_potato":
        if value == 0:
            return "Never"
        if value <= 2:
            return "Low"
        if value <= 5:
            return "Moderate"
        return "High"

    if value == 0:
        return "Never"
    if value < 10:
        return "Low"
    if value < 20:
        return "Moderate"
    return "High"


def encode_binary(series: pd.Series) -> pd.Series:
    """Encode binary text labels to 0/1."""
    values = series.dropna().unique().tolist()
    if len(values) != 2:
        return pd.to_numeric(series, errors="coerce")

    normalized = {str(v).strip().lower(): v for v in values}

    negative_tokens = {"no", "n", "false", "0", "female", "f"}
    positive_tokens = {"yes", "y", "true", "1", "male", "m"}

    zero_candidate = None
    one_candidate = None

    for key, raw in normalized.items():
        if key in negative_tokens:
            zero_candidate = raw
        if key in positive_tokens:
            one_candidate = raw

    if zero_candidate is None or one_candidate is None:
        sorted_values = sorted(values, key=lambda x: str(x))
        zero_candidate, one_candidate = sorted_values[0], sorted_values[1]

    return series.map({zero_candidate: 0, one_candidate: 1})


def encode_ordinal(series: pd.Series, col_name: str) -> pd.Series:
    if col_name not in ORDINAL_ORDER:
        raise ValueError(f"Column {col_name} has no defined order.")

    ordered = pd.Categorical(series, categories=ORDINAL_ORDER[col_name], ordered=True)
    codes = pd.Series(ordered.codes, index=series.index)
    codes = codes.replace(-1, pd.NA)
    return pd.to_numeric(codes, errors="coerce")


def apply_minmax_scaling(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if columns is None:
        columns = out.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        series = pd.to_numeric(out[col], errors="coerce")
        min_v = series.min()
        max_v = series.max()

        if pd.isna(min_v) or pd.isna(max_v):
            continue

        if max_v == min_v:
            out[col] = 0.0
        else:
            out[col] = (series - min_v) / (max_v - min_v)

    return out


def apply_zscore_scaling(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if columns is None:
        columns = out.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        series = pd.to_numeric(out[col], errors="coerce")
        mean_v = series.mean()
        std_v = series.std(ddof=0)

        if pd.isna(mean_v) or pd.isna(std_v):
            continue

        if std_v == 0:
            out[col] = 0.0
        else:
            out[col] = (series - mean_v) / std_v

    return out


def apply_decimal_scaling(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Apply decimal scaling: x' = x / 10^j,
    where j is the smallest integer such that max(|x'|) < 1.
    """
    out = df.copy()
    if columns is None:
        columns = out.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        series = pd.to_numeric(out[col], errors="coerce")
        max_abs = series.abs().max()

        if pd.isna(max_abs) or max_abs == 0:
            out[col] = 0.0 if max_abs == 0 else series
            continue

        j = int(np.ceil(np.log10(max_abs + 1e-12)))
        scale = 10 ** j
        out[col] = series / scale

    return out


def validate_categorical(df: pd.DataFrame) -> bool:
    # Accept object, pandas string dtype and category as categorical representations.
    valid_dtypes = {"object", "string", "str", "category"}
    return all(str(dtype) in valid_dtypes for dtype in df.dtypes)


def validate_numeric(df: pd.DataFrame) -> bool:
    return all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)


def validate_no_missing(df: pd.DataFrame) -> bool:
    return int(df.isna().sum().sum()) == 0
