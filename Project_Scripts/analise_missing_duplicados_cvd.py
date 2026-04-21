from pathlib import Path
import argparse

import pandas as pd


MISSING_TOKENS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "unknown",
    "?",
    "-",
    "--",
    "not available",
    "not applicable",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def real_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    rows = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": n_missing,
                "missing_pct": round((n_missing / total_rows) * 100, 4),
                "dtype": str(df[col].dtype),
                "unique_non_null": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def disguised_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    rows = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        normalized = df[col].map(normalize_text)
        mask = normalized.isin(MISSING_TOKENS)
        n_flagged = int(mask.sum())
        examples = normalized[mask].value_counts().head(5)
        example_text = "; ".join([f"{idx}: {val}" for idx, val in examples.items()]) if not examples.empty else ""

        rows.append(
            {
                "column": col,
                "disguised_missing_count": n_flagged,
                "disguised_missing_pct": round((n_flagged / total_rows) * 100, 4),
                "examples": example_text,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["column", "disguised_missing_count", "disguised_missing_pct", "examples"])

    return pd.DataFrame(rows).sort_values(["disguised_missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def missing_by_row_summary(df: pd.DataFrame) -> pd.DataFrame:
    real_missing = df.isna().sum(axis=1)

    disguised_missing = pd.Series(0, index=df.index, dtype=int)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            disguised_missing = disguised_missing + df[col].map(normalize_text).isin(MISSING_TOKENS).astype(int)

    total_missing = real_missing + disguised_missing
    row_df = pd.DataFrame(
        {
            "row_index": df.index,
            "real_missing": real_missing,
            "disguised_missing": disguised_missing,
            "total_missing": total_missing,
        }
    )

    return row_df.sort_values("total_missing", ascending=False).head(20).reset_index(drop=True)


def exact_duplicates_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dup_mask = df.duplicated(keep=False)
    dup_rows = df.loc[dup_mask].copy()

    if dup_rows.empty:
        summary = pd.DataFrame(
            [
                {
                    "duplicated_rows": 0,
                    "duplicated_rows_pct": 0.0,
                    "duplicate_groups": 0,
                    "largest_group": 0,
                }
            ]
        )
        return summary, pd.DataFrame(columns=["group_size"])

    group_sizes = dup_rows.value_counts(dropna=False).reset_index(name="group_size")
    summary = pd.DataFrame(
        [
            {
                "duplicated_rows": int(dup_mask.sum()),
                "duplicated_rows_pct": round(dup_mask.mean() * 100, 4),
                "duplicate_groups": int((group_sizes["group_size"] > 1).sum()),
                "largest_group": int(group_sizes["group_size"].max()),
            }
        ]
    )

    return summary, group_sizes.sort_values("group_size", ascending=False).head(20).reset_index(drop=True)


def duplicate_subsets_summary(df: pd.DataFrame) -> pd.DataFrame:
    subsets = [
        ["Sex", "Age_Category", "Height_(cm)", "Weight_(kg)", "BMI"],
        ["General_Health", "Checkup", "Exercise", "Heart_Disease", "Diabetes", "Age_Category", "Sex"],
        ["Sex", "Age_Category", "BMI"],
    ]

    rows = []
    for subset in subsets:
        if not set(subset).issubset(df.columns):
            continue
        mask = df.duplicated(subset=subset, keep=False)
        rows.append(
            {
                "subset_columns": ", ".join(subset),
                "duplicated_rows": int(mask.sum()),
                "duplicated_rows_pct": round(mask.mean() * 100, 4),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["subset_columns", "duplicated_rows", "duplicated_rows_pct"])

    return pd.DataFrame(rows).sort_values("duplicated_rows", ascending=False).reset_index(drop=True)


def write_txt_report(
    report_path: Path,
    df: pd.DataFrame,
    real_missing: pd.DataFrame,
    disguised_missing: pd.DataFrame,
    top_missing_rows: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
    duplicate_subsets: pd.DataFrame,
) -> None:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write("ANALISE DE DADOS EM FALTA E DUPLICADOS - CVD_CLEANED\n")
        f.write("=" * 110 + "\n\n")

        f.write("1) VISAO GERAL\n")
        f.write("-" * 110 + "\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {df.shape[1]}\n")
        f.write(f"Total missing (real): {int(df.isna().sum().sum())}\n")
        f.write(f"Exact duplicate rows: {int(df.duplicated().sum())}\n\n")

        f.write("2) MISSING VALUES REAIS POR COLUNA\n")
        f.write("-" * 110 + "\n")
        f.write(real_missing.to_string(index=False))
        f.write("\n\n")

        f.write("3) MISSING VALUES DISFARCADOS (COLUNAS NAO NUMERICAS)\n")
        f.write("-" * 110 + "\n")
        f.write(disguised_missing.to_string(index=False))
        f.write("\n\n")

        f.write("4) TOP 20 LINHAS COM MAIS CAMPOS EM FALTA\n")
        f.write("-" * 110 + "\n")
        f.write(top_missing_rows.to_string(index=False))
        f.write("\n\n")

        f.write("5) DUPLICADOS EXATOS - RESUMO\n")
        f.write("-" * 110 + "\n")
        f.write(duplicate_summary.to_string(index=False))
        f.write("\n\n")

        f.write("6) TOP 20 TAMANHOS DE GRUPOS DUPLICADOS\n")
        f.write("-" * 110 + "\n")
        if duplicate_groups.empty:
            f.write("Nao foram encontrados grupos duplicados.\n\n")
        else:
            f.write(duplicate_groups.to_string(index=False))
            f.write("\n\n")

        f.write("7) DUPLICADOS POR SUBCONJUNTOS DE COLUNAS\n")
        f.write("-" * 110 + "\n")
        if duplicate_subsets.empty:
            f.write("Nao foi possivel aplicar verificacao por subconjuntos configurados.\n")
        else:
            f.write(duplicate_subsets.to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser(description="Missing and duplicates analysis for CVD_cleaned.csv")
    parser.add_argument("--csv", default="CVD_cleaned.csv", help="CSV file name/path")
    parser.add_argument("--output-dir", default="output_missing_duplicados", help="Output folder")
    parser.add_argument("--report-name", default="relatorio_missing_duplicados.txt", help="TXT report name")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    output_dir = base_dir / args.output_dir
    ensure_dir(output_dir)

    df = pd.read_csv(csv_path)

    real_missing = real_missing_summary(df)
    disguised_missing = disguised_missing_summary(df)
    top_missing_rows = missing_by_row_summary(df)
    duplicate_summary, duplicate_groups = exact_duplicates_summary(df)
    duplicate_subsets = duplicate_subsets_summary(df)

    report_path = output_dir / args.report_name
    write_txt_report(
        report_path=report_path,
        df=df,
        real_missing=real_missing,
        disguised_missing=disguised_missing,
        top_missing_rows=top_missing_rows,
        duplicate_summary=duplicate_summary,
        duplicate_groups=duplicate_groups,
        duplicate_subsets=duplicate_subsets,
    )

    print("Missing and duplicates analysis completed.")
    print(f"CSV read: {csv_path}")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    main()
