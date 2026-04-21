import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ANÁLISE COMPLETA DE CORRELAÇÃO
# Projeto DECD / CRISP-DM - Data Understanding
# ============================================================
# O script produz uma análise de correlação completa com foco em:
# 1) identificação automática de tipos de variáveis
# 2) correlação entre variáveis numéricas (Pearson e Spearman)
# 3) associação entre categóricas (Cramer's V)
# 4) associação entre categóricas/binárias e alvo categórico
# 5) comparação de variáveis numéricas por classes da variável alvo
# 6) deteção de pares redundantes / muito correlacionados
# 7) análise de missing values e duplicados
# 8) gráficos: heatmaps, histogramas, boxplots e scatterplots
# 9) relatório textual consolidado
#
# Exemplo:
# python analise_correlacao_completa.py CVD_cleaned.csv
# python analise_correlacao_completa.py CVD_cleaned.csv --target heart_disease
# python analise_correlacao_completa.py CVD_cleaned.csv --output-dir output_correlacao
# ============================================================


ORDINAL_HINTS: Dict[str, List[str]] = {
    "general_health": ["Poor", "Fair", "Good", "Very Good", "Excellent"],
    "checkup": [
        "Never",
        "5 or more years ago",
        "Within the past 5 years",
        "Within the past 2 years",
        "Within the past year",
    ],
    "age_category": [
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



def classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = [c for c in df.columns if c not in numeric_cols]

    binary_cols = []
    categorical_cols = []
    ordinal_cols = []

    for col in text_cols:
        nunique = df[col].dropna().nunique()
        if col in ORDINAL_HINTS:
            ordinal_cols.append(col)
        elif nunique == 2:
            binary_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, binary_cols, categorical_cols, ordinal_cols



def encode_ordinals(df: pd.DataFrame, ordinal_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in ordinal_cols:
        if col not in out.columns:
            continue
        ordered_values = ORDINAL_HINTS.get(col)
        if not ordered_values:
            continue
        mapping = {value: idx for idx, value in enumerate(ordered_values, start=1)}
        out[f"{col}__ordinal_code"] = out[col].map(mapping)
    return out



def save_dataframe(df: Optional[pd.DataFrame], path: Path) -> None:
    if df is not None:
        df.to_csv(path, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------
# ESTATÍSTICA BASE
# ------------------------------------------------------------

def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "coluna": col,
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 4),
                "n_unicos": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)



def numeric_summary(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "coluna": col,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                "min": float(s.min()),
                "q1": float(s.quantile(0.25)),
                "q3": float(s.quantile(0.75)),
                "max": float(s.max()),
                "skewness": float(s.skew()),
                "kurtosis": float(s.kurt()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["coluna", "mean", "median", "std", "min", "q1", "q3", "max", "skewness", "kurtosis"])
    return pd.DataFrame(rows).sort_values("coluna").reset_index(drop=True)


# ------------------------------------------------------------
# CORRELAÇÕES NUMÉRICAS
# ------------------------------------------------------------

def numeric_correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    cols = corr.columns.tolist()
    rows = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            value = corr.loc[c1, c2]
            rows.append(
                {
                    "var1": c1,
                    "var2": c2,
                    "correlacao": float(value),
                    "abs_correlacao": abs(float(value)),
                    "direcao": "positiva" if float(value) > 0 else "negativa" if float(value) < 0 else "nula",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["var1", "var2", "correlacao", "abs_correlacao", "direcao"])
    return pd.DataFrame(rows).sort_values("abs_correlacao", ascending=False).reset_index(drop=True)



def strength_label(value: float) -> str:
    a = abs(value)
    if a >= 0.90:
        return "muito forte"
    if a >= 0.70:
        return "forte"
    if a >= 0.50:
        return "moderada"
    if a >= 0.30:
        return "fraca"
    return "muito fraca"



def detect_redundant_pairs(pairs_df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    if pairs_df.empty:
        return pairs_df.copy()
    out = pairs_df[pairs_df["abs_correlacao"] >= threshold].copy()
    if out.empty:
        return pd.DataFrame(columns=["var1", "var2", "correlacao", "abs_correlacao", "direcao", "forca"])
    out["forca"] = out["correlacao"].apply(strength_label)
    return out.reset_index(drop=True)


# ------------------------------------------------------------
# ASSOCIAÇÕES ENTRE CATEGÓRICAS
# ------------------------------------------------------------

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    if confusion.empty:
        return np.nan

    observed = confusion.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0:
        return np.nan

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)

    r, k = observed.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)

    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))



def categorical_associations(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    rows = []
    for i, c1 in enumerate(cat_cols):
        for c2 in cat_cols[i + 1 :]:
            value = cramers_v(df[c1], df[c2])
            rows.append(
                {
                    "var1": c1,
                    "var2": c2,
                    "cramers_v": value,
                    "forca": strength_label(value) if pd.notna(value) else "indefinida",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["var1", "var2", "cramers_v", "forca"])
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------
# ALVO CATEGÓRICO / RELAÇÃO COM TARGET
# ------------------------------------------------------------

def numeric_by_target(df: pd.DataFrame, numeric_cols: List[str], target: Optional[str]) -> Optional[pd.DataFrame]:
    if not target or target not in df.columns or target in numeric_cols:
        return None

    grouped = df.groupby(target)[numeric_cols].agg(["mean", "median", "std", "min", "max"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.to_flat_index()]
    return grouped.reset_index()



def target_mean_diff(df: pd.DataFrame, numeric_cols: List[str], target: Optional[str]) -> Optional[pd.DataFrame]:
    if not target or target not in df.columns or target in numeric_cols:
        return None

    classes = df[target].dropna().unique().tolist()
    if len(classes) != 2:
        rows = []
        overall = df[numeric_cols].mean(numeric_only=True)
        for col in numeric_cols:
            class_means = df.groupby(target)[col].mean()
            for cls, val in class_means.items():
                rows.append(
                    {
                        "variavel": col,
                        "classe": cls,
                        "mean_classe": float(val),
                        "mean_global": float(overall[col]),
                        "desvio_vs_global": float(val - overall[col]),
                    }
                )
        return pd.DataFrame(rows)

    cls_a, cls_b = classes[0], classes[1]
    rows = []
    for col in numeric_cols:
        mean_a = df.loc[df[target] == cls_a, col].mean()
        mean_b = df.loc[df[target] == cls_b, col].mean()
        rows.append(
            {
                "variavel": col,
                f"mean_{cls_a}": float(mean_a),
                f"mean_{cls_b}": float(mean_b),
                "diferenca_absoluta": float(mean_b - mean_a),
            }
        )
    return pd.DataFrame(rows).sort_values("diferenca_absoluta", key=np.abs, ascending=False).reset_index(drop=True)



def categorical_vs_target(df: pd.DataFrame, cat_cols: List[str], target: Optional[str]) -> Optional[pd.DataFrame]:
    if not target or target not in df.columns or target in cat_cols:
        return None

    rows = []
    for col in cat_cols:
        val = cramers_v(df[col], df[target])
        rows.append(
            {
                "variavel": col,
                "target": target,
                "cramers_v": val,
                "forca": strength_label(val) if pd.notna(val) else "indefinida",
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)



def target_contingencies(df: pd.DataFrame, cat_cols: List[str], target: Optional[str]) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    if not target or target not in df.columns:
        print(f"[AVISO] Variável alvo não encontrada para contingências: {target}")
        return result

    for col in cat_cols:
        try:
            ct = pd.crosstab(df[col], df[target], normalize="index") * 100
            if not ct.empty:
                result[col] = ct.round(2)
        except Exception as exc:
            print(f"[AVISO] Não foi possível gerar contingência para {col}: {exc}")
    return result


# ------------------------------------------------------------
# GRÁFICOS
# ------------------------------------------------------------

def plot_heatmap(corr_df: pd.DataFrame, title: str, output_path: Path) -> None:
    if corr_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = corr_df.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def plot_histograms(df: pd.DataFrame, numeric_cols: List[str], out_dir: Path) -> None:
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(s, bins=30)
        ax.set_title(f"Histograma - {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequência")
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{col}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)



def plot_boxplots_by_target(df: pd.DataFrame, numeric_cols: List[str], target: Optional[str], out_dir: Path) -> None:
    if not target or target not in df.columns:
        return

    classes = df[target].dropna().unique().tolist()
    for col in numeric_cols:
        data = [df.loc[df[target] == cls, col].dropna() for cls in classes]
        if not any(len(arr) > 0 for arr in data):
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(data, labels=[str(c) for c in classes])
        ax.set_title(f"Boxplot de {col} por {target}")
        ax.set_xlabel(target)
        ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(out_dir / f"boxplot_{col}_by_{target}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)



def plot_scatter_top_pairs(df: pd.DataFrame, pairs_df: pd.DataFrame, out_dir: Path, top_n: int = 6, target: Optional[str] = None) -> None:
    if pairs_df.empty:
        return

    top_pairs = pairs_df.head(top_n)
    has_target = target is not None and target in df.columns and df[target].nunique(dropna=True) <= 10

    for _, row in top_pairs.iterrows():
        x = row["var1"]
        y = row["var2"]

        fig, ax = plt.subplots(figsize=(7, 5))
        if has_target:
            classes = df[target].dropna().unique().tolist()
            for cls in classes:
                subset = df[df[target] == cls]
                ax.scatter(subset[x], subset[y], s=8, alpha=0.35, label=str(cls))
            ax.legend(title=target)
        else:
            ax.scatter(df[x], df[y], s=8, alpha=0.35)

        ax.set_title(f"Scatterplot - {x} vs {y}\ncorrelação = {row['correlacao']:.4f}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{x}_vs_{y}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------
# RELATÓRIO
# ------------------------------------------------------------

def write_report(
    path: Path,
    df: pd.DataFrame,
    overview: pd.DataFrame,
    numeric_cols: List[str],
    binary_cols: List[str],
    categorical_cols: List[str],
    ordinal_cols: List[str],
    num_summary: pd.DataFrame,
    pearson_corr: pd.DataFrame,
    spearman_corr: pd.DataFrame,
    pearson_pairs: pd.DataFrame,
    spearman_pairs: pd.DataFrame,
    redundant_pairs: pd.DataFrame,
    cat_assoc: pd.DataFrame,
    num_by_target: Optional[pd.DataFrame],
    diff_by_target: Optional[pd.DataFrame],
    cat_vs_target: Optional[pd.DataFrame],
    contingencies: Dict[str, pd.DataFrame],
    target: Optional[str],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("ANÁLISE COMPLETA DE CORRELAÇÃO\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. VISÃO GERAL DO DATASET\n")
        f.write("-" * 100 + "\n")
        f.write(f"Dimensão: {df.shape[0]} linhas x {df.shape[1]} colunas\n")
        f.write(f"Duplicados exatos: {int(df.duplicated().sum())}\n")
        f.write(f"Valores em falta totais: {int(df.isna().sum().sum())}\n")
        f.write(f"Variáveis numéricas: {numeric_cols}\n")
        f.write(f"Variáveis binárias: {binary_cols}\n")
        f.write(f"Variáveis categóricas: {categorical_cols}\n")
        f.write(f"Variáveis ordinais: {ordinal_cols}\n\n")

        f.write("2. RESUMO DAS VARIÁVEIS\n")
        f.write("-" * 100 + "\n")
        f.write(overview.to_string(index=False))
        f.write("\n\n")

        f.write("3. RESUMO EXPLORATÓRIO DAS VARIÁVEIS NUMÉRICAS\n")
        f.write("-" * 100 + "\n")
        f.write(num_summary.round(4).to_string(index=False))
        f.write("\n\n")

        f.write("4. CORRELAÇÃO ENTRE VARIÁVEIS NUMÉRICAS - PEARSON\n")
        f.write("-" * 100 + "\n")
        f.write(pearson_corr.round(4).to_string())
        f.write("\n\n")
        f.write("Pares mais correlacionados (Pearson)\n")
        f.write(pearson_pairs[["var1", "var2", "correlacao", "direcao"]].head(20).round(4).to_string(index=False))
        f.write("\n\n")

        f.write("5. CORRELAÇÃO ENTRE VARIÁVEIS NUMÉRICAS - SPEARMAN\n")
        f.write("-" * 100 + "\n")
        f.write(spearman_corr.round(4).to_string())
        f.write("\n\n")
        f.write("Pares mais correlacionados (Spearman)\n")
        f.write(spearman_pairs[["var1", "var2", "correlacao", "direcao"]].head(20).round(4).to_string(index=False))
        f.write("\n\n")

        f.write("6. PARES NUMÉRICOS POTENCIALMENTE REDUNDANTES\n")
        f.write("-" * 100 + "\n")
        if redundant_pairs.empty:
            f.write("Não foram detetados pares com correlação absoluta >= 0.80.\n\n")
        else:
            f.write(redundant_pairs[["var1", "var2", "correlacao", "forca"]].round(4).to_string(index=False))
            f.write("\n\n")

        f.write("7. ASSOCIAÇÃO ENTRE VARIÁVEIS CATEGÓRICAS (CRAMER'S V)\n")
        f.write("-" * 100 + "\n")
        if cat_assoc.empty:
            f.write("Não há associações categóricas suficientes para calcular.\n\n")
        else:
            f.write(cat_assoc.head(20).round(4).to_string(index=False))
            f.write("\n\n")

        if num_by_target is not None:
            f.write(f"8. VARIÁVEIS NUMÉRICAS POR CLASSE DA VARIÁVEL ALVO ({target})\n")
            f.write("-" * 100 + "\n")
            f.write(num_by_target.round(4).to_string(index=False))
            f.write("\n\n")

        if diff_by_target is not None:
            f.write(f"9. DIFERENÇAS DE MÉDIA POR CLASSE DA VARIÁVEL ALVO ({target})\n")
            f.write("-" * 100 + "\n")
            f.write(diff_by_target.round(4).to_string(index=False))
            f.write("\n\n")

        if cat_vs_target is not None:
            f.write(f"10. ASSOCIAÇÃO ENTRE VARIÁVEIS CATEGÓRICAS E A VARIÁVEL ALVO ({target})\n")
            f.write("-" * 100 + "\n")
            f.write(cat_vs_target.round(4).to_string(index=False))
            f.write("\n\n")

        if contingencies:
            f.write(f"11. TABELAS DE CONTINGÊNCIA NORMALIZADAS POR CATEGORIA ({target})\n")
            f.write("-" * 100 + "\n")
            for name, table in contingencies.items():
                f.write(f"{name} vs {target}\n")
                f.write(table.to_string())
                f.write("\n\n")

        f.write("12. OBSERVAÇÕES AUTOMÁTICAS\n")
        f.write("-" * 100 + "\n")
        if not pearson_pairs.empty:
            top_p = pearson_pairs.iloc[0]
            f.write(
                f"- Par numérico com maior correlação de Pearson: {top_p['var1']} ~ {top_p['var2']} "
                f"({top_p['correlacao']:.4f}, {strength_label(top_p['correlacao'])}).\n"
            )
        if not spearman_pairs.empty:
            top_s = spearman_pairs.iloc[0]
            f.write(
                f"- Par numérico com maior correlação de Spearman: {top_s['var1']} ~ {top_s['var2']} "
                f"({top_s['correlacao']:.4f}, {strength_label(top_s['correlacao'])}).\n"
            )
        if not redundant_pairs.empty:
            names = ", ".join([f"{r.var1}~{r.var2}" for r in redundant_pairs.itertuples(index=False)])
            f.write(f"- Pairs possivelmente redundantes: {names}.\n")
        if cat_vs_target is not None and not cat_vs_target.empty:
            top_c = cat_vs_target.iloc[0]
            f.write(
                f"- Variável categórica mais associada ao target {target}: {top_c['variavel']} "
                f"(Cramer's V = {top_c['cramers_v']:.4f}).\n"
            )
        f.write("- Correlação elevada sugere associação ou redundância, mas não prova causalidade.\n")
        f.write("- Correlação de Pearson mede relação linear; Spearman é mais robusta para ordinais, monotonicidade e outliers.\n")
        f.write("- A interpretação final deve ser apoiada por gráficos, contexto do domínio e impacto na modelação.\n")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Análise completa de correlação para datasets em CSV.")
    parser.add_argument("csv_path", type=str, help="Caminho para o ficheiro CSV")
    parser.add_argument("--output-dir", type=str, default="output_correlacao", help="Diretório de saída")
    parser.add_argument("--target", type=str, default="heart_disease", help="Variável alvo categórica opcional")
    parser.add_argument("--no-plots", action="store_true", help="Não gera gráficos")
    parser.add_argument("--redundancy-threshold", type=float, default=0.80, help="Threshold de correlação absoluta para redundância")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "graficos"

    ensure_dir(output_dir)
    if not args.no_plots:
        ensure_dir(plots_dir)

    df = safe_read_csv(csv_path)
    overview = dataset_overview(df)

    numeric_cols, binary_cols, categorical_cols, ordinal_cols = classify_columns(df)
    df_encoded = encode_ordinals(df, ordinal_cols)

    ordinal_code_cols = [c for c in df_encoded.columns if c.endswith("__ordinal_code")]
    numeric_analysis_cols = numeric_cols + ordinal_code_cols
    categorical_analysis_cols = binary_cols + categorical_cols + ordinal_cols

    num_summary = numeric_summary(df_encoded, numeric_analysis_cols)

    # Correlações numéricas
    if numeric_analysis_cols:
        pearson_corr = df_encoded[numeric_analysis_cols].corr(method="pearson", numeric_only=True)
        spearman_corr = df_encoded[numeric_analysis_cols].corr(method="spearman", numeric_only=True)
        pearson_pairs = numeric_correlation_pairs(pearson_corr)
        spearman_pairs = numeric_correlation_pairs(spearman_corr)
        redundant_pairs = detect_redundant_pairs(pearson_pairs, threshold=args.redundancy_threshold)
    else:
        pearson_corr = pd.DataFrame()
        spearman_corr = pd.DataFrame()
        pearson_pairs = pd.DataFrame(columns=["var1", "var2", "correlacao", "abs_correlacao", "direcao"])
        spearman_pairs = pearson_pairs.copy()
        redundant_pairs = pd.DataFrame(columns=["var1", "var2", "correlacao", "abs_correlacao", "direcao", "forca"])

    # Associações categóricas
    cat_assoc = categorical_associations(df, categorical_analysis_cols)

    # Target
    target = args.target.strip() if args.target and args.target.strip() in df.columns else None
    if target is None:
        print(f"[AVISO] A variável alvo '{args.target}' não foi encontrada nas colunas do ficheiro.")
        print(f"[AVISO] Colunas disponíveis: {list(df.columns)}")

    target_cat_cols = [c for c in categorical_analysis_cols if c != target and c in df.columns]

    num_by_target = numeric_by_target(df_encoded, numeric_cols, target)
    diff_by_target = target_mean_diff(df_encoded, numeric_cols, target)
    cat_vs_target = categorical_vs_target(df, target_cat_cols, target)
    contingencies = target_contingencies(df, target_cat_cols, target)

    # Exportações
    save_dataframe(overview, output_dir / "01_overview_variaveis.csv")
    save_dataframe(num_summary, output_dir / "02_resumo_numericas.csv")
    save_dataframe(pearson_corr.reset_index().rename(columns={"index": "coluna"}), output_dir / "03_matriz_pearson.csv")
    save_dataframe(spearman_corr.reset_index().rename(columns={"index": "coluna"}), output_dir / "04_matriz_spearman.csv")
    save_dataframe(pearson_pairs, output_dir / "05_pares_pearson.csv")
    save_dataframe(spearman_pairs, output_dir / "06_pares_spearman.csv")
    save_dataframe(redundant_pairs, output_dir / "07_pares_redundantes.csv")
    save_dataframe(cat_assoc, output_dir / "08_associacoes_categoricas_cramers_v.csv")
    save_dataframe(num_by_target, output_dir / "09_numericas_por_target.csv")
    save_dataframe(diff_by_target, output_dir / "10_diferencas_media_target.csv")
    save_dataframe(cat_vs_target, output_dir / "11_categoricas_vs_target.csv")

    # Guardar contingências
    contingencies_dir = output_dir / "contingencias"
    ensure_dir(contingencies_dir)
    for name, table in contingencies.items():
        safe_name = str(name).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_target = str(target).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
        table.to_csv(contingencies_dir / f"contingencia_{safe_name}_vs_{safe_target}.csv", encoding="utf-8-sig")

    if not contingencies:
        pd.DataFrame({"aviso": ["Nenhuma tabela de contingência foi gerada. Verifique o nome da variável alvo e os avisos impressos no terminal."]}).to_csv(
            contingencies_dir / "README_contingencias.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Relatório
    write_report(
        path=output_dir / "relatorio_correlacao.txt",
        df=df,
        overview=overview,
        numeric_cols=numeric_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols,
        ordinal_cols=ordinal_cols,
        num_summary=num_summary,
        pearson_corr=pearson_corr,
        spearman_corr=spearman_corr,
        pearson_pairs=pearson_pairs,
        spearman_pairs=spearman_pairs,
        redundant_pairs=redundant_pairs,
        cat_assoc=cat_assoc,
        num_by_target=num_by_target,
        diff_by_target=diff_by_target,
        cat_vs_target=cat_vs_target,
        contingencies=contingencies,
        target=target,
    )

    # Gráficos
    if not args.no_plots:
        plot_histograms(df_encoded, numeric_cols, plots_dir)
        plot_boxplots_by_target(df_encoded, numeric_cols, target, plots_dir)
        plot_scatter_top_pairs(df_encoded, pearson_pairs, plots_dir, top_n=min(6, len(pearson_pairs)), target=target)
        plot_heatmap(pearson_corr, "Heatmap - Correlação de Pearson", plots_dir / "heatmap_pearson.png")
        plot_heatmap(spearman_corr, "Heatmap - Correlação de Spearman", plots_dir / "heatmap_spearman.png")

    print("Análise de correlação concluída com sucesso.")
    print(f"Diretório de saída: {output_dir.resolve()}")
    print(f"Relatório principal: {(output_dir / 'relatorio_correlacao.txt').resolve()}")


if __name__ == "__main__":
    main()
