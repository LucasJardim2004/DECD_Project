"""
Prepare multiple dataset versions for the CVD project.

Outputs:
- CVD_categorical.csv
- CVD_numeric.csv
- CVD_numeric_minmax.csv
- CVD_numeric_decimal_scaling.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from transformadores import (
    BINARY_COLS,
    CONTINUOUS_COLS,
    ORDINAL_ORDER,
    apply_decimal_scaling,
    apply_minmax_scaling,
    apply_zscore_scaling,
    consolidate_diabetes,
    discretize_bmi,
    discretize_consumption,
    discretize_height,
    discretize_weight,
    encode_binary,
    encode_ordinal,
    group_age_category,
    validate_categorical,
    validate_no_missing,
    validate_numeric,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Diabetes"] = consolidate_diabetes(out["Diabetes"])
    if "Age_Category" in out.columns:
        out["Age_Category"] = group_age_category(out["Age_Category"])

    for col in ORDINAL_ORDER:
        if col in out.columns:
            out[col] = pd.Categorical(out[col], categories=ORDINAL_ORDER[col], ordered=True)

    out["BMI"] = out["BMI"].apply(discretize_bmi)
    out["Height_(cm)"] = out["Height_(cm)"].apply(discretize_height)
    out["Weight_(kg)"] = out["Weight_(kg)"].apply(discretize_weight)

    out["Alcohol_Consumption"] = out["Alcohol_Consumption"].apply(
        lambda x: discretize_consumption(x, "alcohol")
    )
    out["Fruit_Consumption"] = out["Fruit_Consumption"].apply(discretize_consumption)
    out["Green_Vegetables_Consumption"] = out["Green_Vegetables_Consumption"].apply(discretize_consumption)
    out["FriedPotato_Consumption"] = out["FriedPotato_Consumption"].apply(discretize_consumption)

    return out


def prepare_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Diabetes"] = consolidate_diabetes(out["Diabetes"])
    if "Age_Category" in out.columns:
        out["Age_Category"] = group_age_category(out["Age_Category"])

    for col in ORDINAL_ORDER:
        if col in out.columns:
            out[col] = encode_ordinal(out[col], col)

    for col in BINARY_COLS:
        if col in out.columns:
            out[col] = encode_binary(out[col])

    out["Diabetes"] = encode_binary(out["Diabetes"])

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CVD dataset versions.")
    parser.add_argument(
        "--normalizations",
        default="minmax_decimal",
        choices=["minmax_decimal", "minmax_zscore", "all"],
        help=(
            "Normalization strategy: "
            "minmax_decimal (default), minmax_zscore, or all."
        ),
    )
    parser.add_argument(
        "--feature-set",
        default="all",
        choices=["all", "bmi_only", "weight_height"],
        help=(
            "Feature subset to address Weight_(kg) vs BMI redundancy: "
            "all (default), bmi_only (drops Weight_(kg)), "
            "weight_height (drops BMI)."
        ),
    )
    return parser.parse_args()


def apply_feature_set(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    out = df.copy()

    if feature_set == "bmi_only" and "Weight_(kg)" in out.columns:
        out = out.drop(columns=["Weight_(kg)"])

    if feature_set == "weight_height" and "BMI" in out.columns:
        out = out.drop(columns=["BMI"])

    return out


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir / "CVD_cleaned.csv"
    output_dir = base_dir / "output_preparacao"
    ensure_dir(output_dir)

    print_section("DATA PREPARATION - CVD")
    df = pd.read_csv(input_csv)
    print(f"Input: {input_csv}")
    print(f"Rows: {len(df)} | Cols: {df.shape[1]}")
    print(f"Feature set mode: {args.feature_set}")

    print_section("1) FULLY CATEGORICAL VERSION")
    df_cat = prepare_categorical(df)
    df_cat = apply_feature_set(df_cat, args.feature_set)
    if not validate_categorical(df_cat):
        raise ValueError("Categorical dataset validation failed: non-categorical dtypes found.")
    if not validate_no_missing(df_cat):
        raise ValueError("Categorical dataset validation failed: missing values found.")

    print_section("2) FULLY NUMERIC VERSION")
    df_num = prepare_numeric(df)
    df_num = apply_feature_set(df_num, args.feature_set)
    if not validate_numeric(df_num):
        raise ValueError("Numeric dataset validation failed: non-numeric dtypes found.")
    if not validate_no_missing(df_num):
        raise ValueError("Numeric dataset validation failed: missing values found.")

    print_section("3) TWO NORMALIZATION METHODS")
    print(f"Normalization mode: {args.normalizations}")
    scaling_cols = [col for col in CONTINUOUS_COLS if col in df_num.columns]
    print(f"Continuous columns for scaling: {', '.join(scaling_cols)}")

    df_num_minmax = apply_minmax_scaling(df_num, scaling_cols)
    df_num_decimal = None
    df_num_zscore = None

    if args.normalizations in {"minmax_decimal", "all"}:
        df_num_decimal = apply_decimal_scaling(df_num, scaling_cols)

    if args.normalizations in {"minmax_zscore", "all"}:
        df_num_zscore = apply_zscore_scaling(df_num, scaling_cols)

    if not validate_no_missing(df_num_minmax):
        raise ValueError("Normalized datasets contain missing values.")

    if df_num_decimal is not None and not validate_no_missing(df_num_decimal):
        raise ValueError("Decimal-scaled dataset contains missing values.")

    if df_num_zscore is not None and not validate_no_missing(df_num_zscore):
        raise ValueError("Z-score dataset contains missing values.")

    out_cat = output_dir / "CVD_categorical.csv"
    out_num = output_dir / "CVD_numeric.csv"
    out_minmax = output_dir / "CVD_numeric_minmax.csv"
    out_decimal = output_dir / "CVD_numeric_decimal_scaling.csv"
    out_zscore = output_dir / "CVD_numeric_zscore.csv"

    df_cat.to_csv(out_cat, index=False)
    df_num.to_csv(out_num, index=False)
    df_num_minmax.to_csv(out_minmax, index=False)

    if df_num_decimal is not None:
        df_num_decimal.to_csv(out_decimal, index=False)

    if df_num_zscore is not None:
        df_num_zscore.to_csv(out_zscore, index=False)

    print_section("OUTPUTS")
    print(f"Saved: {out_cat}")
    print(f"Saved: {out_num}")
    print(f"Saved: {out_minmax}")
    if df_num_decimal is not None:
        print(f"Saved: {out_decimal}")
    if df_num_zscore is not None:
        print(f"Saved: {out_zscore}")


if __name__ == "__main__":
    main()
