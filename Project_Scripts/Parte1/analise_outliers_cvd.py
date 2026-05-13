from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTINUOUS_COLS = [
    "Height_(cm)",
    "Weight_(kg)",
    "BMI",
    "Alcohol_Consumption",
    "Fruit_Consumption",
    "Green_Vegetables_Consumption",
    "FriedPotato_Consumption",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    flags = pd.DataFrame(index=df.index)

    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (series < lower) | (series > upper)
        flags[f"{col}_outlier_iqr"] = mask.fillna(False)

        rows.append(
            {
                "column": col,
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_limit": lower,
                "upper_limit": upper,
                "outlier_count": int(mask.sum()),
                "outlier_pct": round(mask.mean() * 100, 4),
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "median": series.median(),
            }
        )

    summary = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).reset_index(drop=True)
    return summary, flags


def save_hist_and_boxplots(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    for _, row in summary.iterrows():
        col = row["column"]
        lower = row["lower_limit"]
        upper = row["upper_limit"]

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        fig = plt.figure(figsize=(10, 8))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.hist(series, bins=30, edgecolor="black")
        ax1.axvline(lower, color="red", linestyle="--", linewidth=1.5, label="IQR lower")
        ax1.axvline(upper, color="red", linestyle="--", linewidth=1.5, label="IQR upper")
        ax1.set_title(f"Histogram - {col}")
        ax1.set_xlabel(col)
        ax1.set_ylabel("Frequency")
        ax1.legend()

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.boxplot(series, vert=False)
        ax2.axvline(lower, color="red", linestyle="--", linewidth=1.5)
        ax2.axvline(upper, color="red", linestyle="--", linewidth=1.5)
        ax2.set_title(f"Boxplot - {col}")
        ax2.set_xlabel(col)

        fig.tight_layout()
        fig.savefig(output_dir / f"{col}_hist_box.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Outlier analysis for CVD_cleaned.csv")
    parser.add_argument(
        "--csv",
        default="CVD_cleaned.csv",
        help="CSV file name/path. Default: CVD_cleaned.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="output_outliers",
        help="Output folder name. Default: output_outliers",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    output_dir = base_dir / args.output_dir
    plots_dir = output_dir / "plots"

    for path in [output_dir, plots_dir]:
        ensure_dir(path)

    df = pd.read_csv(csv_path)

    available_cols = [c for c in CONTINUOUS_COLS if c in df.columns]
    if not available_cols:
        raise ValueError("No expected continuous columns found in CSV.")

    summary, _ = iqr_outlier_summary(df, available_cols)

    save_hist_and_boxplots(df, summary, plots_dir)

    print("Outlier analysis completed.")
    print(f"CSV read: {csv_path}")
    print(f"Columns analyzed: {', '.join(available_cols)}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
