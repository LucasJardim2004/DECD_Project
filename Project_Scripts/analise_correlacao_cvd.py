from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ORDINAL_ORDER = {
    "General_Health": ["Poor", "Fair", "Good", "Very Good", "Excellent"],
    "Checkup": [
        "Never",
        "5 or more years ago",
        "Within the past 5 years",
        "Within the past 2 years",
        "Within the past year",
    ],
    "Age_Category": [
        "18-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80+",
    ],
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def encode_binary(series: pd.Series) -> pd.Series:
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


def build_correlation_frame(df: pd.DataFrame) -> pd.DataFrame:
    corr_df = pd.DataFrame(index=df.index)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        corr_df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, order in ORDINAL_ORDER.items():
        if col in df.columns:
            categorical = pd.Categorical(df[col], categories=order, ordered=True)
            codes = pd.Series(categorical.codes, index=df.index)
            codes = codes.replace(-1, pd.NA)
            corr_df[f"{col}__ord"] = pd.to_numeric(codes, errors="coerce")

    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in ORDINAL_ORDER]
    for col in categorical_cols:
        encoded = encode_binary(df[col])
        if encoded.notna().sum() > 0 and encoded.nunique(dropna=True) == 2:
            corr_df[f"{col}__bin"] = encoded

    return corr_df


def correlation_pairs(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    cols = corr_matrix.columns.tolist()
    rows = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            corr_value = corr_matrix.loc[c1, c2]
            rows.append(
                {
                    "var1": c1,
                    "var2": c2,
                    "correlation": corr_value,
                    "abs_correlation": abs(corr_value),
                }
            )
    pairs_df = pd.DataFrame(rows)
    if pairs_df.empty:
        return pairs_df
    return pairs_df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def plot_heatmap(corr_matrix: pd.DataFrame, title: str, output_file: Path, min_annot: float = 0.30) -> None:
    annot_labels = corr_matrix.round(2).astype(str)
    annot_labels = annot_labels.where(corr_matrix.abs() > min_annot, "")

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        corr_matrix,
        annot=annot_labels,
        fmt="",
        annot_kws={"fontsize": 9},
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation analysis for CVD_cleaned.csv")
    parser.add_argument("--csv", default="CVD_cleaned.csv", help="CSV file name/path")
    parser.add_argument("--output-dir", default="output_correlacao", help="Output folder name")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K correlated pairs to save")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    output_dir = base_dir / args.output_dir
    plots_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    df = pd.read_csv(csv_path)
    corr_frame = build_correlation_frame(df)

    if corr_frame.shape[1] < 2:
        raise ValueError("Not enough encoded/numeric columns to compute correlations.")

    pearson = corr_frame.corr(method="pearson", numeric_only=True)
    spearman = corr_frame.corr(method="spearman", numeric_only=True)

    pearson_pairs = correlation_pairs(pearson)
    spearman_pairs = correlation_pairs(spearman)

    plot_heatmap(pearson, "Correlation Heatmap (Pearson)", plots_dir / "heatmap_pearson.png")
    plot_heatmap(spearman, "Correlation Heatmap (Spearman)", plots_dir / "heatmap_spearman.png")

    print("Correlation analysis completed.")
    print(f"CSV read: {csv_path}")
    print(f"Columns used for correlation: {corr_frame.shape[1]}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
