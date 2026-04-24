"""
Heart disease oriented analysis for CVD_cleaned.csv.

The script compares categorical and numeric variables against Heart_Disease
and produces a single text report with:
- a ranking by Cramer's V for categorical variables
- normalized contingency tables for each categorical variable
- a ranking by Pearson correlation for numeric variables
- class-wise summaries for numeric variables

No plots and no CSV tables are generated.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COLUMN = "Heart_Disease"
TARGET_LABELS = {0: "No", 1: "Yes"}
REPORT_NAME = "resumo_texto_heart_disease.txt"
PLOTS_DIR_NAME = "plots_categoricas_relevantes"

CONTINUOUS_COLUMNS = {
    "Height_(cm)",
    "Weight_(kg)",
    "BMI",
    "Alcohol_Consumption",
    "Fruit_Consumption",
    "Green_Vegetables_Consumption",
    "FriedPotato_Consumption",
}

NUMERIC_ASSOCIATION_COLUMNS = [
    "Height_(cm)",
    "Weight_(kg)",
    "BMI",
    "Alcohol_Consumption",
    "Fruit_Consumption",
    "Green_Vegetables_Consumption",
    "FriedPotato_Consumption",
]

CATEGORY_ORDERS = {
    "General_Health": ["Excellent", "Fair", "Good", "Poor", "Very Good"],
    "Checkup": [
        "5 or more years ago",
        "Never",
        "Within the past 2 years",
        "Within the past 5 years",
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
    "Diabetes": [
        "No",
        "No, pre-diabetes or borderline diabetes",
        "Yes",
        "Yes, but female told only during pregnancy",
    ],
    "Sex": ["Female", "Male"],
    "Exercise": ["No", "Yes"],
    "Smoking_History": ["No", "Yes"],
    "Arthritis": ["No", "Yes"],
    "Other_Cancer": ["No", "Yes"],
    "Skin_Cancer": ["No", "Yes"],
    "Depression": ["No", "Yes"],
    TARGET_COLUMN: ["No", "Yes"],
}


def clean_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def normalize_binary_series(series: pd.Series) -> pd.Series:
    values = series.dropna().unique().tolist()
    normalized = {str(value).strip().lower(): value for value in values}
    negative_tokens = {"no", "n", "false", "0"}
    positive_tokens = {"yes", "y", "true", "1"}

    zero_value = None
    one_value = None
    for key, raw_value in normalized.items():
        if key in negative_tokens:
            zero_value = raw_value
        elif key in positive_tokens:
            one_value = raw_value

    if zero_value is None or one_value is None:
        if len(values) != 2:
            raise ValueError("Target column must contain exactly two classes.")
        ordered = sorted(values, key=lambda value: str(value))
        zero_value, one_value = ordered[0], ordered[1]

    return series.map({zero_value: 0, one_value: 1})


def to_text_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: np.nan if pd.isna(value) else str(value))


def safe_name(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "variavel"


def ordered_categories(column: str, series: pd.Series) -> list[str]:
    observed = []
    for value in series.dropna().tolist():
        text_value = str(value)
        if text_value not in observed:
            observed.append(text_value)

    if column in CATEGORY_ORDERS:
        ordered = [value for value in CATEGORY_ORDERS[column] if value in observed]
        extras = [value for value in observed if value not in ordered]
        return ordered + extras

    return sorted(observed)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x.fillna("Missing"), y.fillna("Missing"))
    if table.empty:
        return float("nan")

    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total <= 0:
        return float("nan")

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)

    phi2 = chi2 / total
    rows, cols = observed.shape
    if rows <= 1 or cols <= 1:
        return float("nan")

    phi2_corr = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / max(total - 1, 1))
    rows_corr = rows - ((rows - 1) ** 2) / max(total - 1, 1)
    cols_corr = cols - ((cols - 1) ** 2) / max(total - 1, 1)
    denom = min(rows_corr - 1, cols_corr - 1)
    if denom <= 0:
        return float("nan")

    return float(np.sqrt(phi2_corr / denom))


def association_strength(value: float) -> str:
    if pd.isna(value):
        return "indeterminada"

    magnitude = abs(value)
    if magnitude < 0.30:
        return "muito fraca"
    if magnitude < 0.50:
        return "fraca"
    if magnitude < 0.70:
        return "moderada"
    if magnitude < 0.90:
        return "forte"
    return "muito forte"


def build_numeric_association_ranking(df: pd.DataFrame, numeric_columns: list[str], target_binary: pd.Series) -> pd.DataFrame:
    rows = []
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        valid_mask = series.notna() & target_binary.notna()
        values = series[valid_mask]
        target_values = target_binary[valid_mask]
        if values.empty or values.nunique(dropna=True) < 2:
            continue

        correlation = values.corr(target_values)
        group0 = values[target_values == 0]
        group1 = values[target_values == 1]

        rows.append(
            {
                "variavel": column,
                "target": TARGET_COLUMN,
                "pearson_r": correlation,
                "forca": association_strength(correlation),
                "media_no": group0.mean(),
                "media_yes": group1.mean(),
                "diferenca_media": group1.mean() - group0.mean(),
            }
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return pd.DataFrame(columns=["variavel", "target", "pearson_r", "forca", "media_no", "media_yes", "diferenca_media"])

    return ranking.sort_values(["pearson_r", "variavel"], ascending=[False, True]).reset_index(drop=True)


def build_association_ranking(df: pd.DataFrame, categorical_columns: list[str], target_binary: pd.Series) -> pd.DataFrame:
    rows = []
    for column in categorical_columns:
        score = cramers_v(to_text_series(df[column]), target_binary.map(TARGET_LABELS))
        rows.append(
            {
                "variavel": column,
                "target": TARGET_COLUMN,
                "cramers_v": score,
                "forca": association_strength(score),
            }
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return pd.DataFrame(columns=["variavel", "target", "cramers_v", "forca"])

    return ranking.sort_values(["cramers_v", "variavel"], ascending=[False, True]).reset_index(drop=True)


def build_normalized_table(df: pd.DataFrame, column: str, target_display: pd.Series) -> pd.DataFrame:
    source = to_text_series(df[column])
    table = pd.crosstab(source, target_display, normalize="index") * 100
    desired_index = ordered_categories(column, source)
    if source.isna().any():
        desired_index.append("Missing")

    table = table.reindex(index=desired_index, columns=["No", "Yes"], fill_value=0.0)
    table.index.name = column
    table.columns.name = TARGET_COLUMN
    return table


def build_influence_table(df: pd.DataFrame, column: str, target_binary: pd.Series) -> pd.DataFrame:
    source = to_text_series(df[column])
    frame = pd.DataFrame({"categoria": source, "target": target_binary})
    frame = frame.dropna(subset=["categoria", "target"])

    grouped = frame.groupby("categoria", dropna=False)["target"].agg(["count", "mean"]).reset_index()
    grouped = grouped.rename(columns={"count": "n", "mean": "taxa_yes"})

    global_rate = target_binary.mean()
    grouped["taxa_yes_pct"] = grouped["taxa_yes"] * 100
    grouped["delta_vs_global_pp"] = (grouped["taxa_yes"] - global_rate) * 100
    grouped["risco_relativo"] = grouped["taxa_yes"] / global_rate if global_rate and not pd.isna(global_rate) else np.nan

    order = ordered_categories(column, source)
    grouped["__ord__"] = grouped["categoria"].apply(lambda value: order.index(value) if value in order else len(order))
    grouped = grouped.sort_values(["__ord__", "categoria"]).drop(columns=["__ord__"])
    return grouped.reset_index(drop=True)


def plot_categorical_influence(column: str, cramers_value: float, influence_table: pd.DataFrame, output_file: Path) -> None:
    if influence_table.empty:
        return

    categories = influence_table["categoria"].astype(str).tolist()
    rates = influence_table["taxa_yes_pct"].tolist()
    counts = influence_table["n"].astype(int).tolist()
    deltas = influence_table["delta_vs_global_pp"].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(categories, rates, color="#2E7D32", alpha=0.9)
    ax.set_title(f"{column} -> taxa de Heart_Disease=Yes (Cramer's V={cramers_value:.4f})")
    ax.set_ylabel("Taxa de Heart_Disease=Yes (%)")
    ax.set_xlabel(column)
    ax.set_ylim(0, max(rates) * 1.2 if rates else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, n_value, delta in zip(bars, counts, deltas):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"n={n_value}\nΔ={delta:+.2f}pp",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_report(
    df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    target_binary: pd.Series,
    categorical_ranking: pd.DataFrame,
    selected_for_plots: list[str],
    output_path: Path,
) -> None:
    target_display = target_binary.map(TARGET_LABELS)
    numeric_ranking = build_numeric_association_ranking(df, numeric_columns, target_binary)

    lines: list[str] = []
    lines.append("=" * 110)
    lines.append("ANALISE ORIENTADA A HEART_DISEASE - CVD_CLEANED")
    lines.append("=" * 110)
    lines.append("")

    lines.append("ASSOCIACAO ENTRE VARIAVEIS CATEGORICAS E A VARIAVEL ALVO (Heart_Disease)")
    lines.append("-" * 100)
    lines.append(categorical_ranking.to_string(index=False, formatters={"cramers_v": lambda value: f"{value:.4f}"}))
    lines.append("")

    lines.append("TABELAS DE CONTINGENCIA NORMALIZADAS POR CATEGORIA (Heart_Disease)")
    lines.append("-" * 100)

    for column in categorical_columns:
        normalized = build_normalized_table(df, column, target_display)
        lines.append(f"{column} vs Heart_Disease")
        lines.append(normalized.to_string(float_format=lambda value: f"{value:,.2f}"))
        lines.append("")

    lines.append("INFLUENCIA DAS VARIAVEIS CATEGORICAS COM CRAMERS_V > 0.1")
    lines.append("-" * 100)
    if not selected_for_plots:
        lines.append("Nenhuma variavel categorica ultrapassou o limiar configurado.")
        lines.append("")
    else:
        for column in selected_for_plots:
            cramers_value = float(categorical_ranking.loc[categorical_ranking["variavel"] == column, "cramers_v"].iloc[0])
            influence = build_influence_table(df, column, target_binary)
            lines.append(f"{column} (Cramer's V={cramers_value:.4f})")
            lines.append(
                influence.to_string(
                    index=False,
                    formatters={
                        "taxa_yes": lambda value: f"{value:.4f}",
                        "taxa_yes_pct": lambda value: f"{value:.2f}",
                        "delta_vs_global_pp": lambda value: f"{value:+.2f}",
                        "risco_relativo": lambda value: f"{value:.3f}",
                    },
                )
            )
            lines.append("")

    lines.append("ASSOCIACAO ENTRE VARIAVEIS NUMERICAS E A VARIAVEL ALVO (Heart_Disease)")
    lines.append("-" * 100)
    lines.append(
        numeric_ranking.to_string(
            index=False,
            formatters={
                "pearson_r": lambda value: f"{value:.4f}",
                "media_no": lambda value: f"{value:.2f}",
                "media_yes": lambda value: f"{value:.2f}",
                "diferenca_media": lambda value: f"{value:.2f}",
            },
        )
    )
    lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart disease categorical association analysis")
    parser.add_argument("--csv", default="CVD_cleaned.csv", help="CSV file name or path")
    parser.add_argument("--output-dir", default="output_heart_disease", help="Output directory")
    parser.add_argument("--target", default=TARGET_COLUMN, help="Target column name")
    parser.add_argument(
        "--plot-threshold",
        type=float,
        default=0.1,
        help="Cramer's V minimum threshold for generating categorical plots",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    output_dir = base_dir / args.output_dir
    clean_output_directory(output_dir)

    df = pd.read_csv(csv_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV.")

    target_binary = normalize_binary_series(df[args.target])
    if target_binary.nunique(dropna=True) != 2:
        raise ValueError("Target column must contain exactly two classes after normalization.")

    categorical_columns = [
        column
        for column in df.columns
        if column != args.target and column not in CONTINUOUS_COLUMNS
    ]
    numeric_columns = [column for column in NUMERIC_ASSOCIATION_COLUMNS if column in df.columns]

    categorical_ranking = build_association_ranking(df, categorical_columns, target_binary)
    selected_for_plots = (
        categorical_ranking[categorical_ranking["cramers_v"] > args.plot_threshold]["variavel"].tolist()
        if not categorical_ranking.empty
        else []
    )

    plots_dir = output_dir / PLOTS_DIR_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)
    for column in selected_for_plots:
        influence = build_influence_table(df, column, target_binary)
        cramers_value = float(categorical_ranking.loc[categorical_ranking["variavel"] == column, "cramers_v"].iloc[0])
        plot_path = plots_dir / f"{safe_name(column)}_influencia_heart_disease.png"
        plot_categorical_influence(column, cramers_value, influence, plot_path)

    report_path = output_dir / REPORT_NAME
    render_report(
        df,
        categorical_columns,
        numeric_columns,
        target_binary,
        categorical_ranking,
        selected_for_plots,
        report_path,
    )

    print("Heart disease categorical analysis completed.")
    print(f"CSV read: {csv_path}")
    print(f"Output folder: {output_dir}")
    print(f"Report file: {report_path}")
    print(f"Categorical plots (Cramer's V > {args.plot_threshold}): {plots_dir}")


if __name__ == "__main__":
    main()