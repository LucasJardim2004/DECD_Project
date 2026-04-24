"""
Compare the impact of dataset preparation and normalization strategies.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

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


def compare_distributions(df_base: pd.DataFrame, df_mm: pd.DataFrame, df_ds: pd.DataFrame, col: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df_base[col].dropna(), bins=30, edgecolor="black")
    axes[0].set_title(f"{col} - Numeric")

    axes[1].hist(df_mm[col].dropna(), bins=30, edgecolor="black")
    axes[1].set_title(f"{col} - MinMax")

    axes[2].hist(df_ds[col].dropna(), bins=30, edgecolor="black")
    axes[2].set_title(f"{col} - Decimal Scaling")

    for ax in axes:
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    safe = col.replace("_", "-").replace("(", "").replace(")", "").lower()
    plt.savefig(output_dir / f"dist_{safe}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def compare_boxplots(df_base: pd.DataFrame, df_mm: pd.DataFrame, df_ds: pd.DataFrame, output_dir: Path) -> None:
    cols = [c for c in CONTINUOUS_COLS if c in df_base.columns and c in df_mm.columns and c in df_ds.columns]
    if not cols:
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    df_base[cols].boxplot(ax=axes[0], rot=45)
    axes[0].set_title("Numeric")

    df_mm[cols].boxplot(ax=axes[1], rot=45)
    axes[1].set_title("MinMax")

    df_ds[cols].boxplot(ax=axes[2], rot=45)
    axes[2].set_title("Decimal Scaling")

    plt.tight_layout()
    plt.savefig(output_dir / "boxplots_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_summary_stats(df_base: pd.DataFrame, df_mm: pd.DataFrame, df_ds: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for col in CONTINUOUS_COLS:
        if col not in df_base.columns or col not in df_mm.columns or col not in df_ds.columns:
            continue
        for label, frame in [("numeric", df_base), ("minmax", df_mm), ("decimal_scaling", df_ds)]:
            rows.append(
                {
                    "column": col,
                    "version": label,
                    "min": frame[col].min(),
                    "max": frame[col].max(),
                    "mean": frame[col].mean(),
                    "std": frame[col].std(),
                    "median": frame[col].median(),
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "stats_normalization.csv", index=False)


def compare_bmi_original_vs_categorical(df_original: pd.DataFrame, df_cat: pd.DataFrame, output_dir: Path) -> None:
    if "BMI" not in df_original.columns or "BMI" not in df_cat.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_original["BMI"].dropna(), bins=40, edgecolor="black")
    axes[0].set_title("BMI Original (continuous)")

    counts = df_cat["BMI"].value_counts(dropna=False)
    axes[1].bar(counts.index.astype(str), counts.values)
    axes[1].set_title("BMI Categorical (clinical bins)")
    axes[1].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(output_dir / "bmi_original_vs_categorical.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    prep_dir = base_dir / "output_preparacao"
    out_dir = base_dir / "output_analise_impacto"
    ensure_dir(out_dir)

    df_original = pd.read_csv(base_dir / "CVD_cleaned.csv")
    df_cat = pd.read_csv(prep_dir / "CVD_categorical.csv")
    df_num = pd.read_csv(prep_dir / "CVD_numeric.csv")
    df_mm = pd.read_csv(prep_dir / "CVD_numeric_minmax.csv")
    #df_ds = pd.read_csv(prep_dir / "CVD_numeric_decimal_scaling.csv")
    df_ds = pd.read_csv(prep_dir / "CVD_numeric_zscore.csv")

    common_cols = [c for c in CONTINUOUS_COLS if c in df_num.columns and c in df_mm.columns and c in df_ds.columns]

    for col in common_cols:
        compare_distributions(df_num, df_mm, df_ds, col, out_dir)

    compare_boxplots(df_num, df_mm, df_ds, out_dir)
    save_summary_stats(df_num, df_mm, df_ds, out_dir)
    compare_bmi_original_vs_categorical(df_original, df_cat, out_dir)

    print("Impact analysis finished.")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
