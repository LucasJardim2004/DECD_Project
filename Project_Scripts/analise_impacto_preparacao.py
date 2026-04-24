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


def compare_distributions(
    df_mm: pd.DataFrame,
    df_zs: pd.DataFrame,
    col: str,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df_mm[col].dropna(), bins=30, edgecolor="black")
    axes[0].set_title(f"{col} - MinMax")

    axes[1].hist(df_zs[col].dropna(), bins=30, edgecolor="black")
    axes[1].set_title(f"{col} - Z-Score")

    for ax in axes:
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    safe = col.replace("_", "-").replace("(", "").replace(")", "").lower()
    plt.savefig(output_dir / f"dist_{safe}_minmax_vs_zscore.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def compare_boxplots(
    df_mm: pd.DataFrame,
    df_zs: pd.DataFrame,
    output_dir: Path,
) -> None:
    cols = [c for c in CONTINUOUS_COLS if c in df_mm.columns and c in df_zs.columns]
    if not cols:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df_mm[cols].boxplot(ax=axes[0], rot=45)
    axes[0].set_title("MinMax")

    df_zs[cols].boxplot(ax=axes[1], rot=45)
    axes[1].set_title("Z-Score")

    plt.tight_layout()
    plt.savefig(output_dir / "boxplots_compare_minmax_vs_zscore.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_summary_stats(
    df_mm: pd.DataFrame,
    df_zs: pd.DataFrame,
    output_dir: Path,
) -> None:
    rows = []
    for col in CONTINUOUS_COLS:
        if col not in df_mm.columns or col not in df_zs.columns:
            continue
        for label, frame in [("minmax", df_mm), ("zscore", df_zs)]:
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

    pd.DataFrame(rows).to_csv(output_dir / "stats_normalization_minmax_vs_zscore.csv", index=False)


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
    df_mm = pd.read_csv(prep_dir / "CVD_numeric_minmax.csv")
    zscore_path = prep_dir / "CVD_numeric_zscore.csv"
    if not zscore_path.exists():
        raise FileNotFoundError("CVD_numeric_zscore.csv not found in output_preparacao.")
    df_zs = pd.read_csv(zscore_path)

    common_cols = [c for c in CONTINUOUS_COLS if c in df_mm.columns and c in df_zs.columns]
    for col in common_cols:
        compare_distributions(df_mm, df_zs, col, out_dir)

    compare_boxplots(df_mm, df_zs, out_dir)
    save_summary_stats(df_mm, df_zs, out_dir)

    compare_bmi_original_vs_categorical(df_original, df_cat, out_dir)

    print("Impact analysis finished.")
    print("Normalizations analyzed: MinMax, Z-Score")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
