import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


# ============================================================
# ANÁLISE DE VALORES EM FALTA E DUPLICADOS
# Projeto DECD / CRISP-DM - Data Understanding
# ============================================================
# O script foca-se exclusivamente em:
# 1) missing values reais e disfarçados
# 2) linhas duplicadas exatas
# 3) duplicados por subconjuntos de colunas
# 4) linhas com maior concentração de campos vazios
# 5) impacto no target (opcional)
# 6) exportação de relatórios CSV + TXT
#
# Exemplo:
# python analise_missing_duplicados.py CVD_cleaned.csv
# python analise_missing_duplicados.py CVD_cleaned.csv --target heart_disease
# python analise_missing_duplicados.py CVD_cleaned.csv --output-dir output_missing_duplicados
# ============================================================


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


# ------------------------------------------------------------
# UTILITÁRIOS
# ------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    separators_to_try = [",", ";"]
    last_error = None

    for enc in encodings_to_try:
        for sep in separators_to_try:
            try:
                df = pd.read_csv(csv_path, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            except Exception as exc:
                last_error = exc

    raise RuntimeError(f"Não foi possível ler o ficheiro CSV: {last_error}")



def save_dataframe(df: Optional[pd.DataFrame], path: Path) -> None:
    if df is not None:
        df.to_csv(path, index=False, encoding="utf-8-sig")



def normalize_text_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_rows = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "coluna": col,
                "missing_reais": n_missing,
                "missing_reais_pct": round((n_missing / total_rows) * 100, 4),
                "dtype": str(df[col].dtype),
                "n_unicos_sem_na": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_reais", "coluna"], ascending=[False, True]).reset_index(drop=True)



def disguised_missing_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    flagged_examples = []
    total_rows = len(df)

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue

        normalized = s.map(normalize_text_value)
        mask = normalized.isin(MISSING_TOKENS)
        n_flagged = int(mask.sum())

        examples = normalized[mask].value_counts().head(10)
        example_text = "; ".join([f"{idx!r}: {val}" for idx, val in examples.items()]) if not examples.empty else ""

        rows.append(
            {
                "coluna": col,
                "missing_disfarcados": n_flagged,
                "missing_disfarcados_pct": round((n_flagged / total_rows) * 100, 4),
                "exemplos": example_text,
            }
        )

        if n_flagged > 0:
            tmp = df.loc[mask, [col]].copy()
            tmp.insert(0, "indice_original", tmp.index)
            tmp.columns = ["indice_original", "valor_original"]
            tmp.insert(0, "coluna", col)
            flagged_examples.append(tmp.head(50))

    summary_df = pd.DataFrame(rows).sort_values(["missing_disfarcados", "coluna"], ascending=[False, True]).reset_index(drop=True)
    examples_df = pd.concat(flagged_examples, ignore_index=True) if flagged_examples else pd.DataFrame(columns=["coluna", "indice_original", "valor_original"])
    return summary_df, examples_df



def missing_by_row(df: pd.DataFrame) -> pd.DataFrame:
    real_missing = df.isna().sum(axis=1)

    disguised_counts = pd.Series(0, index=df.index, dtype=int)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            disguised_counts = disguised_counts + df[col].map(normalize_text_value).isin(MISSING_TOKENS).astype(int)

    total_missing = real_missing + disguised_counts

    out = pd.DataFrame(
        {
            "indice_original": df.index,
            "missing_reais_na_linha": real_missing,
            "missing_disfarcados_na_linha": disguised_counts,
            "missing_total_na_linha": total_missing,
        }
    )
    return out.sort_values("missing_total_na_linha", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------
# DUPLICADOS
# ------------------------------------------------------------

def exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask_dup = df.duplicated(keep=False)
    dup_rows = df.loc[mask_dup].copy()

    if dup_rows.empty:
        summary = pd.DataFrame(
            [{
                "n_linhas_duplicadas_exatas": 0,
                "pct_linhas_duplicadas_exatas": 0.0,
                "n_grupos_duplicados_exatos": 0,
                "maior_grupo_duplicado_exato": 0,
            }]
        )
        dup_rows.insert(0, "indice_original", [])
        return summary, dup_rows

    dup_rows.insert(0, "indice_original", dup_rows.index)
    grouped = dup_rows.drop(columns=["indice_original"]).value_counts(dropna=False).reset_index(name="tamanho_grupo")

    summary = pd.DataFrame(
        [{
            "n_linhas_duplicadas_exatas": int(mask_dup.sum()),
            "pct_linhas_duplicadas_exatas": round(mask_dup.mean() * 100, 4),
            "n_grupos_duplicados_exatos": int(len(grouped[grouped["tamanho_grupo"] > 1])),
            "maior_grupo_duplicado_exato": int(grouped["tamanho_grupo"].max()),
        }]
    )

    return summary, dup_rows



def subset_duplicate_checks(df: pd.DataFrame) -> pd.DataFrame:
    candidate_subsets = []

    if {"sex", "age_category", "height_cm", "weight_kg", "bmi"}.issubset(df.columns):
        candidate_subsets.append(["sex", "age_category", "height_cm", "weight_kg", "bmi"])

    if {"sex", "age_category", "height_cm", "weight_kg"}.issubset(df.columns):
        candidate_subsets.append(["sex", "age_category", "height_cm", "weight_kg"])

    if {"general_health", "checkup", "exercise", "heart_disease", "diabetes", "age_category", "sex"}.issubset(df.columns):
        candidate_subsets.append(["general_health", "checkup", "exercise", "heart_disease", "diabetes", "age_category", "sex"])

    # fallback genérico: combinações simples de colunas de baixa cardinalidade
    low_card_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 20]
    if len(low_card_cols) >= 3:
        candidate_subsets.append(low_card_cols[: min(5, len(low_card_cols))])

    rows = []
    seen = set()
    for subset in candidate_subsets:
        key = tuple(subset)
        if key in seen:
            continue
        seen.add(key)

        mask = df.duplicated(subset=subset, keep=False)
        rows.append(
            {
                "subset_colunas": ", ".join(subset),
                "n_linhas_duplicadas": int(mask.sum()),
                "pct_linhas_duplicadas": round(mask.mean() * 100, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("n_linhas_duplicadas", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["subset_colunas", "n_linhas_duplicadas", "pct_linhas_duplicadas"])



def top_duplicate_groups(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    grouped = df.value_counts(dropna=False).reset_index(name="tamanho_grupo")
    grouped = grouped[grouped["tamanho_grupo"] > 1].sort_values("tamanho_grupo", ascending=False)
    return grouped.head(top_n).reset_index(drop=True)


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

def target_impact_missing_duplicates(df: pd.DataFrame, target: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not target or target not in df.columns:
        return None, None

    # Missing por classe
    missing_rows = []
    for col in df.columns:
        for cls, subset in df.groupby(target):
            n_missing = int(subset[col].isna().sum())
            if not pd.api.types.is_numeric_dtype(subset[col]):
                n_disguised = int(subset[col].map(normalize_text_value).isin(MISSING_TOKENS).sum())
            else:
                n_disguised = 0
            missing_rows.append(
                {
                    "classe_target": cls,
                    "coluna": col,
                    "missing_reais": n_missing,
                    "missing_disfarcados": n_disguised,
                    "missing_total": n_missing + n_disguised,
                }
            )
    missing_target_df = pd.DataFrame(missing_rows)

    # Duplicados por classe
    dup_rows = []
    for cls, subset in df.groupby(target):
        mask = subset.duplicated(keep=False)
        dup_rows.append(
            {
                "classe_target": cls,
                "n_linhas": int(len(subset)),
                "n_linhas_duplicadas_exatas": int(mask.sum()),
                "pct_linhas_duplicadas_exatas": round(mask.mean() * 100, 4),
            }
        )
    dup_target_df = pd.DataFrame(dup_rows)

    return missing_target_df, dup_target_df


# ------------------------------------------------------------
# RELATÓRIO TXT
# ------------------------------------------------------------

def write_report(
    output_path: Path,
    df: pd.DataFrame,
    missing_df: pd.DataFrame,
    disguised_missing_df: pd.DataFrame,
    row_missing_df: pd.DataFrame,
    dup_summary_df: pd.DataFrame,
    subset_dup_df: pd.DataFrame,
    top_dup_groups_df: pd.DataFrame,
    missing_target_df: Optional[pd.DataFrame],
    dup_target_df: Optional[pd.DataFrame],
    target: Optional[str],
) -> None:
    total_real_missing = int(df.isna().sum().sum())
    total_disguised_missing = 0
    if not disguised_missing_df.empty:
        total_disguised_missing = int(disguised_missing_df["missing_disfarcados"].sum())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("ANÁLISE DE VALORES EM FALTA E DUPLICADOS\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. VISÃO GERAL DO DATASET\n")
        f.write("-" * 100 + "\n")
        f.write(f"Número de linhas: {len(df)}\n")
        f.write(f"Número de colunas: {df.shape[1]}\n")
        f.write(f"Valores em falta reais totais: {total_real_missing}\n")
        f.write(f"Possíveis missing disfarçados totais: {total_disguised_missing}\n")
        f.write(f"Linhas duplicadas exatas: {int(dup_summary_df.iloc[0]['n_linhas_duplicadas_exatas']) if not dup_summary_df.empty else 0}\n\n")

        f.write("2. VALORES EM FALTA REAIS POR COLUNA\n")
        f.write("-" * 100 + "\n")
        f.write(missing_df.to_string(index=False))
        f.write("\n\n")

        f.write("3. POSSÍVEIS MISSING VALUES DISFARÇADOS\n")
        f.write("-" * 100 + "\n")
        f.write(disguised_missing_df.to_string(index=False))
        f.write("\n\n")

        f.write("4. LINHAS COM MAIS CAMPOS EM FALTA\n")
        f.write("-" * 100 + "\n")
        f.write(row_missing_df.head(20).to_string(index=False))
        f.write("\n\n")

        f.write("5. RESUMO DE DUPLICADOS EXATOS\n")
        f.write("-" * 100 + "\n")
        f.write(dup_summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("6. TESTES DE DUPLICADOS POR SUBCONJUNTO DE COLUNAS\n")
        f.write("-" * 100 + "\n")
        if subset_dup_df.empty:
            f.write("Nenhum teste por subconjunto foi executado.\n\n")
        else:
            f.write(subset_dup_df.to_string(index=False))
            f.write("\n\n")

        f.write("7. PRINCIPAIS GRUPOS DE DUPLICADOS EXATOS\n")
        f.write("-" * 100 + "\n")
        if top_dup_groups_df.empty:
            f.write("Não foram encontrados grupos de duplicados exatos.\n\n")
        else:
            f.write(top_dup_groups_df.to_string(index=False))
            f.write("\n\n")

        if missing_target_df is not None and dup_target_df is not None:
            f.write(f"8. IMPACTO NA VARIÁVEL ALVO ({target})\n")
            f.write("-" * 100 + "\n")
            f.write("Duplicados por classe da variável alvo\n")
            f.write(dup_target_df.to_string(index=False))
            f.write("\n\n")
            f.write("Resumo de missing por classe da variável alvo (top 30 por missing_total)\n")
            f.write(missing_target_df.sort_values("missing_total", ascending=False).head(30).to_string(index=False))
            f.write("\n\n")

        f.write("9. CONCLUSÕES AUTOMÁTICAS\n")
        f.write("-" * 100 + "\n")
        if total_real_missing == 0:
            f.write("- Não foram encontrados valores em falta reais no dataset.\n")
        else:
            f.write(f"- Foram encontrados {total_real_missing} valores em falta reais.\n")

        if total_disguised_missing == 0:
            f.write("- Não foram encontrados missing values disfarçados com base nos tokens monitorizados.\n")
        else:
            top_disg = disguised_missing_df.iloc[0]
            f.write(
                f"- Foram encontrados possíveis missing disfarçados; a coluna mais afetada é {top_disg['coluna']} "
                f"com {int(top_disg['missing_disfarcados'])} ocorrências.\n"
            )

        if not dup_summary_df.empty and int(dup_summary_df.iloc[0]["n_linhas_duplicadas_exatas"]) == 0:
            f.write("- Não foram encontradas linhas duplicadas exatas.\n")
        else:
            n_dup = int(dup_summary_df.iloc[0]["n_linhas_duplicadas_exatas"])
            pct_dup = float(dup_summary_df.iloc[0]["pct_linhas_duplicadas_exatas"])
            f.write(f"- Foram encontradas {n_dup} linhas duplicadas exatas ({pct_dup:.4f}%).\n")

        f.write("- A decisão sobre remover duplicados deve ser justificada no contexto analítico e do domínio.\n")
        f.write("- Mesmo sem missing reais, convém verificar se existem valores especiais, placeholders ou inconsistências semânticas.\n")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Análise focada em missing values e duplicados.")
    parser.add_argument("csv_path", type=str, help="Caminho para o ficheiro CSV")
    parser.add_argument("--output-dir", type=str, default="output_missing_duplicados", help="Diretório de saída")
    parser.add_argument("--target", type=str, default="heart_disease", help="Variável alvo opcional")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = safe_read_csv(csv_path)
    target = args.target.strip() if args.target and args.target.strip() in df.columns else None

    # Missing values
    missing_df = missing_summary(df)
    disguised_missing_df, disguised_examples_df = disguised_missing_summary(df)
    row_missing_df = missing_by_row(df)

    # Duplicados
    dup_summary_df, exact_dup_rows_df = exact_duplicates(df)
    subset_dup_df = subset_duplicate_checks(df)
    top_dup_groups_df = top_duplicate_groups(df, top_n=20)

    # Target
    missing_target_df, dup_target_df = target_impact_missing_duplicates(df, target)

    # Guardar CSVs
    save_dataframe(missing_df, output_dir / "01_missing_reais_por_coluna.csv")
    save_dataframe(disguised_missing_df, output_dir / "02_missing_disfarcados_por_coluna.csv")
    save_dataframe(disguised_examples_df, output_dir / "03_exemplos_missing_disfarcados.csv")
    save_dataframe(row_missing_df, output_dir / "04_missing_por_linha.csv")
    save_dataframe(dup_summary_df, output_dir / "05_resumo_duplicados_exatos.csv")
    save_dataframe(exact_dup_rows_df, output_dir / "06_linhas_duplicadas_exatas.csv")
    save_dataframe(subset_dup_df, output_dir / "07_testes_duplicados_subsets.csv")
    save_dataframe(top_dup_groups_df, output_dir / "08_top_grupos_duplicados_exatos.csv")
    save_dataframe(missing_target_df, output_dir / "09_missing_por_target.csv")
    save_dataframe(dup_target_df, output_dir / "10_duplicados_por_target.csv")

    # Relatório TXT
    write_report(
        output_path=output_dir / "relatorio_missing_duplicados.txt",
        df=df,
        missing_df=missing_df,
        disguised_missing_df=disguised_missing_df,
        row_missing_df=row_missing_df,
        dup_summary_df=dup_summary_df,
        subset_dup_df=subset_dup_df,
        top_dup_groups_df=top_dup_groups_df,
        missing_target_df=missing_target_df,
        dup_target_df=dup_target_df,
        target=target,
    )

    print("Análise de missing values e duplicados concluída com sucesso.")
    print(f"Diretório de saída: {output_dir.resolve()}")
    print(f"Relatório principal: {(output_dir / 'relatorio_missing_duplicados.txt').resolve()}")


if __name__ == "__main__":
    main()
