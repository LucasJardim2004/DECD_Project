import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ANÁLISE INICIAL DE OUTLIERS
# Projeto DECD / CRISP-DM - Data Understanding
# ============================================================
# O script produz uma análise inicial de outliers com foco em:
# 1) estatística descritiva
# 2) deteção por IQR
# 3) deteção por z-score e modified z-score (MAD)
# 4) plausibilidade por regras de domínio
# 5) valores extremos observados
# 6) outliers multivariados simples (contagem por linha)
# 7) relação dos outliers com a variável alvo
# 8) gráficos de apoio (histograma, boxplot, scatterplots)
#
# Exemplo de utilização:
# python analise_outliers_inicial.py CVD_cleaned.csv --target heart_disease
# python analise_outliers_inicial.py CVD_cleaned.csv --output-dir resultados_outliers
# ============================================================


DEFAULT_PLAUSIBILITY_RULES: Dict[str, Tuple[float, float]] = {
    "height_cm": (120, 230),
    "weight_kg": (30, 250),
    "bmi": (10, 80),
    "alcohol_consumption": (0, 30),
    "fruit_consumption": (0, 180),
    "green_vegetables_consumption": (0, 180),
    "friedpotato_consumption": (0, 180),
}


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
                    return df
            except Exception as exc:
                last_error = exc

    raise RuntimeError(f"Não foi possível ler o ficheiro CSV: {last_error}")


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def describe_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "coluna": col,
                "count": int(s.count()),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(df[col].isna().mean() * 100, 4),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                "min": float(s.min()),
                "q1": float(s.quantile(0.25)),
                "q3": float(s.quantile(0.75)),
                "max": float(s.max()),
                "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
                "skewness": float(s.skew()),
                "kurtosis": float(s.kurt()),
                "unique_values": int(s.nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("coluna").reset_index(drop=True)


def iqr_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    flags = pd.DataFrame(index=df.index)

    for col in numeric_cols:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        flags[f"{col}_iqr_outlier"] = mask.fillna(False)

        summary_rows.append(
            {
                "coluna": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "limite_inferior": lower,
                "limite_superior": upper,
                "n_outliers": int(mask.sum()),
                "pct_outliers": round(mask.mean() * 100, 4),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True)
    return summary, flags


def zscore_outliers(df: pd.DataFrame, numeric_cols: List[str], threshold: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    flags = pd.DataFrame(index=df.index)

    for col in numeric_cols:
        s = df[col]
        mean = s.mean()
        std = s.std(ddof=1)
        if pd.isna(std) or std == 0:
            z = pd.Series([0] * len(s), index=s.index, dtype=float)
        else:
            z = (s - mean) / std
        mask = z.abs() > threshold
        flags[f"{col}_zscore_outlier"] = mask.fillna(False)
        rows.append(
            {
                "coluna": col,
                "zscore_threshold": threshold,
                "n_outliers": int(mask.sum()),
                "pct_outliers": round(mask.mean() * 100, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True), flags


def modified_zscore_outliers(df: pd.DataFrame, numeric_cols: List[str], threshold: float = 3.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    flags = pd.DataFrame(index=df.index)

    for col in numeric_cols:
        s = df[col]
        median = s.median()
        mad = np.median(np.abs(s.dropna() - median)) if not s.dropna().empty else np.nan

        if pd.isna(mad) or mad == 0:
            mz = pd.Series([0] * len(s), index=s.index, dtype=float)
        else:
            mz = 0.6745 * (s - median) / mad

        mask = mz.abs() > threshold
        flags[f"{col}_mad_outlier"] = mask.fillna(False)
        rows.append(
            {
                "coluna": col,
                "mad_threshold": threshold,
                "median": median,
                "mad": mad,
                "n_outliers": int(mask.sum()),
                "pct_outliers": round(mask.mean() * 100, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True), flags


def plausible_range_check(df: pd.DataFrame, numeric_cols: List[str], rules: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        if col not in rules:
            continue
        min_expected, max_expected = rules[col]
        s = df[col]
        below = (s < min_expected).sum()
        above = (s > max_expected).sum()
        total = below + above
        rows.append(
            {
                "coluna": col,
                "min_esperado": min_expected,
                "max_esperado": max_expected,
                "observado_min": float(s.min()),
                "observado_max": float(s.max()),
                "abaixo_do_esperado": int(below),
                "acima_do_esperado": int(above),
                "total_fora": int(total),
                "pct_fora": round(total / len(df) * 100, 4),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "coluna",
                "min_esperado",
                "max_esperado",
                "observado_min",
                "observado_max",
                "abaixo_do_esperado",
                "acima_do_esperado",
                "total_fora",
                "pct_fora",
            ]
        )

    return pd.DataFrame(rows).sort_values("pct_fora", ascending=False).reset_index(drop=True)


def get_extreme_values(df: pd.DataFrame, numeric_cols: List[str], n: int = 5) -> Dict[str, Dict[str, List[float]]]:
    result: Dict[str, Dict[str, List[float]]] = {}
    for col in numeric_cols:
        s = df[col].dropna().sort_values()
        result[col] = {
            "menores": [float(x) for x in s.head(n).tolist()],
            "maiores": [float(x) for x in s.tail(n).tolist()],
        }
    return result


def correlation_analysis(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corr = df[numeric_cols].corr(numeric_only=True)
    pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1 :]:
            val = corr.loc[c1, c2]
            pairs.append({"var1": c1, "var2": c2, "correlacao": float(val), "abs_correlacao": abs(float(val))})
    pairs_df = pd.DataFrame(pairs).sort_values("abs_correlacao", ascending=False).reset_index(drop=True)
    return corr, pairs_df


def row_level_outlier_summary(iqr_flags: pd.DataFrame) -> pd.DataFrame:
    counts = iqr_flags.sum(axis=1)
    summary = pd.DataFrame({"n_outliers_iqr_na_linha": counts})
    return summary


def target_outlier_analysis(
    df: pd.DataFrame,
    iqr_flags: pd.DataFrame,
    numeric_cols: List[str],
    target: Optional[str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not target or target not in df.columns:
        return None, None

    numeric_by_target = (
        df.groupby(target)[numeric_cols]
        .agg(["mean", "median", "std", "min", "max"])
        .round(4)
    )
    numeric_by_target.columns = ["_".join(col).strip() for col in numeric_by_target.columns.to_flat_index()]
    numeric_by_target = numeric_by_target.reset_index()

    tmp = df[[target]].copy()
    for col in numeric_cols:
        tmp[f"{col}_iqr_outlier"] = iqr_flags[f"{col}_iqr_outlier"]

    cat_rows = []
    for col in numeric_cols:
        tab = pd.crosstab(tmp[target], tmp[f"{col}_iqr_outlier"], normalize="index") * 100
        for cls in tab.index:
            pct_false = float(tab.loc[cls].get(False, 0.0))
            pct_true = float(tab.loc[cls].get(True, 0.0))
            cat_rows.append(
                {
                    "variavel": col,
                    "classe_target": cls,
                    "pct_nao_outlier": round(pct_false, 4),
                    "pct_outlier_iqr": round(pct_true, 4),
                }
            )
    outlier_by_target = pd.DataFrame(cat_rows)
    return numeric_by_target, outlier_by_target


def save_dataframe(df: Optional[pd.DataFrame], path: Path) -> None:
    if df is not None:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_hist_and_box(df: pd.DataFrame, col: str, output_dir: Path) -> None:
    s = df[col].dropna()
    if s.empty:
        return

    fig = plt.figure(figsize=(10, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.hist(s, bins=30)
    ax1.set_title(f"Histograma - {col}")
    ax1.set_xlabel(col)
    ax1.set_ylabel("Frequência")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.boxplot(s, vert=False)
    ax2.set_title(f"Boxplot - {col}")
    ax2.set_xlabel(col)

    fig.tight_layout()
    fig.savefig(output_dir / f"{col}_hist_box.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_pairs(df: pd.DataFrame, pairs_df: pd.DataFrame, output_dir: Path, top_n: int = 5) -> None:
    top_pairs = pairs_df.head(top_n)
    for _, row in top_pairs.iterrows():
        x = row["var1"]
        y = row["var2"]

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df[x], df[y], s=8, alpha=0.25)
        ax.set_title(f"Scatterplot - {x} vs {y}\ncorrelação = {row['correlacao']:.4f}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()
        fig.savefig(output_dir / f"scatter_{x}_vs_{y}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def write_text_report(
    output_path: Path,
    df: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    corr_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    iqr_summary: pd.DataFrame,
    z_summary: pd.DataFrame,
    mad_summary: pd.DataFrame,
    plausibility_df: pd.DataFrame,
    extremes: Dict[str, Dict[str, List[float]]],
    row_outlier_summary: pd.DataFrame,
    target_numeric: Optional[pd.DataFrame],
    target_outliers: Optional[pd.DataFrame],
    target: Optional[str],
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("ANÁLISE INICIAL DE OUTLIERS\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. VISÃO GERAL DO DATASET\n")
        f.write("-" * 100 + "\n")
        f.write(f"Número de linhas: {len(df)}\n")
        f.write(f"Número de colunas: {df.shape[1]}\n")
        f.write(f"Linhas duplicadas exatas: {int(df.duplicated().sum())}\n")
        f.write(f"Valores em falta totais: {int(df.isna().sum().sum())}\n\n")

        f.write("2. RESUMO EXPLORATÓRIO DAS VARIÁVEIS NUMÉRICAS\n")
        f.write("-" * 100 + "\n")
        f.write(numeric_summary.to_string(index=False))
        f.write("\n\n")

        f.write("3. CORRELAÇÕES ENTRE VARIÁVEIS NUMÉRICAS\n")
        f.write("-" * 100 + "\n")
        f.write("Matriz de correlação\n")
        f.write(corr_df.round(4).to_string())
        f.write("\n\n")
        f.write("Pares com maior correlação absoluta\n")
        f.write(pairs_df[["var1", "var2", "correlacao"]].head(15).round(4).to_string(index=False))
        f.write("\n\n")

        f.write("4. DETEÇÃO DE OUTLIERS POR IQR\n")
        f.write("-" * 100 + "\n")
        f.write(iqr_summary.round(4).to_string(index=False))
        f.write("\n\n")

        f.write("5. DETEÇÃO DE OUTLIERS POR Z-SCORE\n")
        f.write("-" * 100 + "\n")
        f.write(z_summary.round(4).to_string(index=False))
        f.write("\n\n")

        f.write("6. DETEÇÃO DE OUTLIERS POR MODIFIED Z-SCORE (MAD)\n")
        f.write("-" * 100 + "\n")
        f.write(mad_summary.round(4).to_string(index=False))
        f.write("\n\n")

        f.write("7. VERIFICAÇÃO DE PLAUSIBILIDADE\n")
        f.write("-" * 100 + "\n")
        if plausibility_df.empty:
            f.write("Nenhuma regra de plausibilidade aplicada.\n\n")
        else:
            f.write(plausibility_df.round(4).to_string(index=False))
            f.write("\n\n")

        f.write("8. VALORES EXTREMOS OBSERVADOS\n")
        f.write("-" * 100 + "\n")
        for col, info in extremes.items():
            f.write(f"{col}:\n")
            f.write(f" - 5 menores valores: {info['menores']}\n")
            f.write(f" - 5 maiores valores: {info['maiores']}\n\n")

        f.write("9. OUTLIERS POR LINHA (VISÃO MULTIVARIADA SIMPLES)\n")
        f.write("-" * 100 + "\n")
        freq = row_outlier_summary["n_outliers_iqr_na_linha"].value_counts().sort_index()
        f.write(freq.to_string())
        f.write("\n\n")
        f.write(
            f"Linhas com pelo menos 1 outlier por IQR: "
            f"{int((row_outlier_summary['n_outliers_iqr_na_linha'] > 0).sum())}"
            f" ({(row_outlier_summary['n_outliers_iqr_na_linha'] > 0).mean() * 100:.2f}%)\n\n"
        )

        if target_numeric is not None and target_outliers is not None:
            f.write(f"10. RELAÇÃO DOS OUTLIERS COM A VARIÁVEL ALVO: {target}\n")
            f.write("-" * 100 + "\n")
            f.write("Resumo das variáveis numéricas por classe da variável alvo\n")
            f.write(target_numeric.to_string(index=False))
            f.write("\n\n")
            f.write("Percentagem de outliers IQR por classe da variável alvo\n")
            f.write(target_outliers.to_string(index=False))
            f.write("\n\n")

        f.write("11. NOTAS AUTOMÁTICAS\n")
        f.write("-" * 100 + "\n")
        if not iqr_summary.empty:
            top_iqr = iqr_summary.iloc[0]
            f.write(
                f"- Variável com maior percentagem de outliers por IQR: {top_iqr['coluna']} "
                f"({top_iqr['pct_outliers']:.2f}%).\n"
            )
        if not pairs_df.empty:
            top_corr = pairs_df.iloc[0]
            f.write(
                f"- Par com maior correlação absoluta: {top_corr['var1']} ~ {top_corr['var2']} "
                f"({top_corr['correlacao']:.4f}).\n"
            )
        if not plausibility_df.empty and (plausibility_df["total_fora"] > 0).any():
            flagged = plausibility_df[plausibility_df["total_fora"] > 0][["coluna", "total_fora"]]
            flagged_text = ", ".join([f"{r.coluna} ({r.total_fora})" for r in flagged.itertuples(index=False)])
            f.write(f"- Foram encontrados valores fora do intervalo plausível em: {flagged_text}.\n")
        else:
            f.write("- Não foram encontrados valores fora dos intervalos plausíveis definidos.\n")
        f.write(
            "- Nem todo o outlier deve ser removido: alguns podem ser casos raros mas reais e clinicamente relevantes.\n"
        )
        f.write(
            "- Antes da modelação, convém decidir por variável se os extremos devem ser mantidos, truncados, winsorizados, corrigidos ou removidos.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise inicial de outliers para datasets tabulares em CSV.")
    parser.add_argument("csv_path", type=str, help="Caminho para o ficheiro CSV")
    parser.add_argument("--output-dir", type=str, default="output_outliers", help="Diretório de saída")
    parser.add_argument("--target", type=str, default="heart_disease", help="Nome da variável alvo (opcional)")
    parser.add_argument("--no-plots", action="store_true", help="Se usado, não gera gráficos")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "graficos"

    ensure_dir(output_dir)
    if not args.no_plots:
        ensure_dir(plots_dir)

    df = safe_read_csv(csv_path)
    numeric_cols = get_numeric_columns(df)

    if not numeric_cols:
        raise ValueError("O dataset não contém colunas numéricas para análise de outliers.")

    numeric_summary = describe_numeric(df, numeric_cols)
    corr_df, pairs_df = correlation_analysis(df, numeric_cols)
    iqr_summary, iqr_flags = iqr_outliers(df, numeric_cols)
    z_summary, z_flags = zscore_outliers(df, numeric_cols, threshold=3.0)
    mad_summary, mad_flags = modified_zscore_outliers(df, numeric_cols, threshold=3.5)
    plausibility_df = plausible_range_check(df, numeric_cols, DEFAULT_PLAUSIBILITY_RULES)
    extremes = get_extreme_values(df, numeric_cols, n=5)
    row_outlier_summary = row_level_outlier_summary(iqr_flags)
    target_numeric, target_outliers = target_outlier_analysis(df, iqr_flags, numeric_cols, args.target)

    # Guardar outputs tabulares
    save_dataframe(numeric_summary, output_dir / "01_resumo_numericas.csv")
    save_dataframe(corr_df.reset_index().rename(columns={"index": "coluna"}), output_dir / "02_matriz_correlacao.csv")
    save_dataframe(pairs_df, output_dir / "03_pares_correlacao.csv")
    save_dataframe(iqr_summary, output_dir / "04_outliers_iqr.csv")
    save_dataframe(z_summary, output_dir / "05_outliers_zscore.csv")
    save_dataframe(mad_summary, output_dir / "06_outliers_mad.csv")
    save_dataframe(plausibility_df, output_dir / "07_plausibilidade.csv")
    save_dataframe(row_outlier_summary, output_dir / "08_outliers_por_linha.csv")
    save_dataframe(target_numeric, output_dir / "09_target_resumo_numericas.csv")
    save_dataframe(target_outliers, output_dir / "10_target_outliers_iqr.csv")

    # Guardar flags para auditoria
    flags_all = pd.concat([iqr_flags, z_flags, mad_flags], axis=1)
    flags_all.to_csv(output_dir / "11_flags_outliers_por_registo.csv", index=False, encoding="utf-8-sig")

    # Top linhas com mais outliers IQR
    top_rows = df.copy()
    top_rows["n_outliers_iqr_na_linha"] = row_outlier_summary["n_outliers_iqr_na_linha"]
    top_rows = top_rows.sort_values("n_outliers_iqr_na_linha", ascending=False).head(100)
    top_rows.to_csv(output_dir / "12_top_100_linhas_mais_extremas.csv", index=False, encoding="utf-8-sig")

    # Relatório em txt
    write_text_report(
        output_path=output_dir / "relatorio_outliers.txt",
        df=df,
        numeric_summary=numeric_summary,
        corr_df=corr_df,
        pairs_df=pairs_df,
        iqr_summary=iqr_summary,
        z_summary=z_summary,
        mad_summary=mad_summary,
        plausibility_df=plausibility_df,
        extremes=extremes,
        row_outlier_summary=row_outlier_summary,
        target_numeric=target_numeric,
        target_outliers=target_outliers,
        target=args.target if args.target in df.columns else None,
    )

    # Gráficos
    if not args.no_plots:
        for col in numeric_cols:
            plot_hist_and_box(df, col, plots_dir)
        plot_scatter_pairs(df, pairs_df, plots_dir, top_n=min(5, len(pairs_df)))

    print("Análise concluída com sucesso.")
    print(f"Diretório de saída: {output_dir.resolve()}")
    print(f"Relatório principal: {(output_dir / 'relatorio_outliers.txt').resolve()}")


if __name__ == "__main__":
    main()
