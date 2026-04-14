import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import sys
import csv


def limpar_texto(valor):
    if isinstance(valor, str):
        return " ".join(valor.strip().split())
    return valor


def normalizar_colunas(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def detetar_separador(input_file, encoding="utf-8", fallback=","):
    try:
        with open(input_file, "r", encoding=encoding, newline="") as f:
            sample = f.read(4096)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
    except Exception:
        return fallback


def ler_csv_seguro(input_file, encoding=None, sep=None):
    encodings_teste = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    ultimo_erro = None
    for enc in encodings_teste:
        try:
            separador = sep if sep else detetar_separador(input_file, encoding=enc)
            df = pd.read_csv(
                input_file,
                sep=separador,
                encoding=enc,
                low_memory=False
            )
            return df, enc, separador, None
        except Exception as e:
            ultimo_erro = e

    return None, None, None, ultimo_erro


def formatar_valor(valor, max_len=80):
    if pd.isna(valor):
        return "VALOR_EM_FALTA"
    valor_str = str(valor)
    valor_str = " ".join(valor_str.split())
    if len(valor_str) > max_len:
        valor_str = valor_str[:max_len - 3] + "..."
    return valor_str


def bytes_humanos(num_bytes):
    unidades = ["B", "KB", "MB", "GB", "TB"]
    valor = float(num_bytes)
    for unidade in unidades:
        if valor < 1024 or unidade == unidades[-1]:
            return f"{valor:.2f} {unidade}"
        valor /= 1024


def escrever_secao(linhas_saida, titulo):
    linhas_saida.append("=" * 100)
    linhas_saida.append(titulo)
    linhas_saida.append("=" * 100)


def dataframe_para_texto(df, titulo=None, max_linhas=50):
    linhas = []
    if titulo:
        linhas.append(titulo)
        linhas.append("-" * len(titulo))

    if df is None or len(df) == 0:
        linhas.append("Sem dados para apresentar.\n")
        return "\n".join(linhas)

    if len(df) > max_linhas:
        linhas.append(df.head(max_linhas).to_string(index=False))
        linhas.append(f"\n... (mostradas apenas {max_linhas} linhas de {len(df)})")
    else:
        linhas.append(df.to_string(index=False))

    linhas.append("")
    return "\n".join(linhas)


def colunas_numericas(df):
    return list(df.select_dtypes(include=["number"]).columns)


def colunas_categoricas(df, max_unicos=20):
    resultado = []
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "string":
            resultado.append(col)
        elif df[col].nunique(dropna=True) <= max_unicos:
            resultado.append(col)
    return resultado


def resumo_assimetria_numericas(df):
    cols = colunas_numericas(df)
    linhas = []
    for col in cols:
        serie = df[col].dropna()
        if len(serie) == 0:
            continue
        linhas.append({
            "coluna": col,
            "mean": round(float(serie.mean()), 4),
            "median": round(float(serie.median()), 4),
            "std": round(float(serie.std()), 4),
            "skewness": round(float(serie.skew()), 4),
            "kurtosis": round(float(serie.kurt()), 4),
            "min": round(float(serie.min()), 4),
            "max": round(float(serie.max()), 4),
        })
    return pd.DataFrame(linhas)


def detetar_outliers_iqr(df):
    cols = colunas_numericas(df)
    linhas = []
    for col in cols:
        serie = df[col].dropna()
        if len(serie) == 0:
            continue

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr

        mask = (serie < lim_inf) | (serie > lim_sup)
        n_outliers = int(mask.sum())
        pct_outliers = (n_outliers / len(serie)) * 100 if len(serie) > 0 else 0

        linhas.append({
            "coluna": col,
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "limite_inferior": round(float(lim_inf), 4),
            "limite_superior": round(float(lim_sup), 4),
            "n_outliers": n_outliers,
            "pct_outliers": round(pct_outliers, 4)
        })
    return pd.DataFrame(linhas)


def top_valores_extremos(df, top_n=5):
    cols = colunas_numericas(df)
    saida = {}
    for col in cols:
        serie = df[col].dropna().sort_values()
        if len(serie) == 0:
            continue

        menores = serie.head(top_n).tolist()
        maiores = serie.tail(top_n).tolist()

        saida[col] = {
            "menores": [round(float(v), 4) for v in menores],
            "maiores": [round(float(v), 4) for v in maiores]
        }
    return saida


def correlacoes_numericas(df):
    cols = colunas_numericas(df)
    if len(cols) < 2:
        return None, None

    corr = df[cols].corr(numeric_only=True)
    pares = []

    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            valor = corr.loc[c1, c2]
            pares.append({
                "var1": c1,
                "var2": c2,
                "correlacao": round(float(valor), 4)
            })

    pares_df = pd.DataFrame(pares)
    pares_df["abs_corr"] = pares_df["correlacao"].abs()
    pares_df = pares_df.sort_values(by="abs_corr", ascending=False).drop(columns=["abs_corr"])

    return corr, pares_df


def distribuicao_por_target_binario(df, target):
    if target not in df.columns:
        return None

    if df[target].nunique(dropna=True) != 2:
        return None

    linhas = []
    cols_num = [c for c in colunas_numericas(df) if c != target]

    for col in cols_num:
        agrupado = df.groupby(target)[col].agg(["mean", "median", "std", "min", "max"]).reset_index()
        agrupado.insert(0, "variavel", col)
        linhas.append(agrupado)

    if not linhas:
        return None

    return pd.concat(linhas, ignore_index=True)


def taxas_por_target_categoricas(df, target, max_categorias=20):
    if target not in df.columns:
        return {}

    if df[target].nunique(dropna=True) != 2:
        return {}

    resultado = {}
    cols_cat = colunas_categoricas(df, max_unicos=max_categorias)
    cols_cat = [c for c in cols_cat if c != target]

    for col in cols_cat:
        tab = pd.crosstab(df[col], df[target], normalize="index") * 100
        tab = tab.round(2)
        resultado[col] = tab

    return resultado


def tabelas_contingencia_relevantes(df, max_categorias=20):
    candidatas = [
        "general_health",
        "checkup",
        "exercise",
        "heart_disease",
        "diabetes",
        "arthritis",
        "sex",
        "age_category",
        "smoking_history",
        "depression"
    ]
    presentes = [c for c in candidatas if c in df.columns]
    resultado = {}

    base = "heart_disease"
    if base not in presentes:
        return resultado

    for col in presentes:
        if col == base:
            continue
        if df[col].nunique(dropna=True) <= max_categorias:
            tab = pd.crosstab(df[col], df[base], margins=True)
            resultado[f"{col}_vs_{base}"] = tab

    return resultado


def frequencias_relativas_categoricas(df, max_categorias=20):
    resultado = {}
    for col in colunas_categoricas(df, max_unicos=max_categorias):
        vc = df[col].value_counts(dropna=False, normalize=True).mul(100).round(2)
        resultado[col] = vc
    return resultado


def verificar_plausibilidade_saude(df):
    regras = {
        "height_cm": (120, 230),
        "weight_kg": (30, 250),
        "bmi": (10, 80),
        "alcohol_consumption": (0, 30),
        "fruit_consumption": (0, 180),
        "green_vegetables_consumption": (0, 180),
        "friedpotato_consumption": (0, 180),
    }

    linhas = []
    for col, (min_esp, max_esp) in regras.items():
        if col not in df.columns:
            continue

        serie = df[col].dropna()
        if len(serie) == 0:
            continue

        abaixo = int((serie < min_esp).sum())
        acima = int((serie > max_esp).sum())
        total_fora = abaixo + acima
        pct_fora = (total_fora / len(serie)) * 100 if len(serie) > 0 else 0

        linhas.append({
            "coluna": col,
            "min_esperado": min_esp,
            "max_esperado": max_esp,
            "observado_min": round(float(serie.min()), 4),
            "observado_max": round(float(serie.max()), 4),
            "abaixo_do_esperado": abaixo,
            "acima_do_esperado": acima,
            "total_fora": total_fora,
            "pct_fora": round(pct_fora, 4)
        })

    return pd.DataFrame(linhas)


def gerar_observacoes_exploratorias(df, outliers_df, corr_pares_df, plaus_df, target="heart_disease"):
    obs = []

    if target in df.columns and df[target].nunique(dropna=True) == 2:
        freq = df[target].value_counts(normalize=True).mul(100).round(2)
        classes = ", ".join([f"{idx}: {val}%" for idx, val in freq.items()])
        obs.append(f"A variável alvo '{target}' apresenta distribuição: {classes}.")

    if corr_pares_df is not None and len(corr_pares_df) > 0:
        top_corr = corr_pares_df.head(5)
        pares = ", ".join([f"{r.var1}~{r.var2} ({r.correlacao})" for _, r in top_corr.iterrows()])
        obs.append(f"As correlações numéricas mais fortes observadas foram: {pares}.")

    if outliers_df is not None and len(outliers_df) > 0:
        top_out = outliers_df.sort_values(by="pct_outliers", ascending=False).head(5)
        pares = ", ".join([f"{r.coluna} ({r.pct_outliers:.2f}%)" for _, r in top_out.iterrows()])
        obs.append(f"As variáveis numéricas com maior percentagem de outliers por IQR foram: {pares}.")

    if plaus_df is not None and len(plaus_df) > 0:
        suspeitas = plaus_df[plaus_df["total_fora"] > 0]
        if len(suspeitas) > 0:
            pares = ", ".join([f"{r.coluna} ({r.total_fora} fora do intervalo)" for _, r in suspeitas.iterrows()])
            obs.append(f"Foram encontrados valores fora dos intervalos plausíveis definidos em: {pares}.")
        else:
            obs.append("Não foram encontrados valores fora dos intervalos plausíveis definidos.")

    if "diabetes" in df.columns:
        obs.append("A variável 'diabetes' mantém categorias clinicamente distintas e deve ser analisada com cuidado na preparação dos dados.")

    if "age_category" in df.columns:
        obs.append("A variável 'age_category' deve ser explorada respeitando a sua ordem natural.")

    return obs


def main():
    parser = argparse.ArgumentParser(
        description="Relatório Explore Data + Verify Data Quality."
    )
    parser.add_argument("input_file", help="Caminho para o CSV")
    parser.add_argument(
        "-o",
        "--output",
        default="explore_verify_data_report.txt",
        help="Ficheiro TXT de output"
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Encoding do ficheiro CSV (opcional)"
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Separador do CSV (opcional)"
    )
    parser.add_argument(
        "--target",
        default="heart_disease",
        help="Variável alvo binária para exploração orientada"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Erro: o ficheiro '{input_path}' não existe.")
        sys.exit(1)

    df, encoding_usado, separador_usado, erro = ler_csv_seguro(
        input_file=input_path,
        encoding=args.encoding,
        sep=args.sep
    )

    if df is None:
        print(f"Erro ao ler o ficheiro: {erro}")
        sys.exit(1)

    df = normalizar_colunas(df)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(limpar_texto)

    resumo_num = resumo_assimetria_numericas(df)
    outliers_df = detetar_outliers_iqr(df)
    extremos = top_valores_extremos(df, top_n=5)
    corr_matrix, corr_pares_df = correlacoes_numericas(df)
    target_num = distribuicao_por_target_binario(df, args.target)
    target_cat = taxas_por_target_categoricas(df, args.target, max_categorias=20)
    contingencias = tabelas_contingencia_relevantes(df, max_categorias=20)
    plaus_df = verificar_plausibilidade_saude(df)
    observacoes = gerar_observacoes_exploratorias(df, outliers_df, corr_pares_df, plaus_df, target=args.target)

    linhas_saida = []

    escrever_secao(linhas_saida, "FASE 1 - DATA UNDERSTANDING | EXPLORE DATA + VERIFY DATA QUALITY")
    linhas_saida.append(f"Data/hora de execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    linhas_saida.append(f"Ficheiro analisado: {input_path.name}")
    linhas_saida.append(f"Caminho: {input_path.resolve()}")
    linhas_saida.append(f"Encoding usado: {encoding_usado}")
    linhas_saida.append(f"Separador usado: {repr(separador_usado)}")
    linhas_saida.append(f"Dimensão do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
    linhas_saida.append(f"Memória ocupada: {bytes_humanos(df.memory_usage(deep=True).sum())}")
    linhas_saida.append(f"Target usada para análise orientada: {args.target}")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "1. RESUMO EXPLORATÓRIO DAS VARIÁVEIS NUMÉRICAS")
    linhas_saida.append(dataframe_para_texto(resumo_num, max_linhas=100))

    escrever_secao(linhas_saida, "2. CORRELAÇÕES ENTRE VARIÁVEIS NUMÉRICAS")
    if corr_matrix is not None:
        linhas_saida.append(dataframe_para_texto(corr_matrix.round(4), "Matriz de correlação", max_linhas=50))
        linhas_saida.append(dataframe_para_texto(corr_pares_df, "Pares com maior correlação absoluta", max_linhas=30))
    else:
        linhas_saida.append("Não existem variáveis numéricas suficientes para calcular correlações.\n")

    escrever_secao(linhas_saida, "3. DETEÇÃO DE OUTLIERS POR IQR")
    linhas_saida.append(dataframe_para_texto(outliers_df.sort_values(by="pct_outliers", ascending=False), max_linhas=100))

    escrever_secao(linhas_saida, "4. VALORES EXTREMOS OBSERVADOS")
    for col, dados in extremos.items():
        linhas_saida.append(f"{col}:")
        linhas_saida.append(f" - 5 menores valores: {dados['menores']}")
        linhas_saida.append(f" - 5 maiores valores: {dados['maiores']}")
        linhas_saida.append("")

    escrever_secao(linhas_saida, "5. VERIFICAÇÃO DE PLAUSIBILIDADE EM VARIÁVEIS DE SAÚDE")
    linhas_saida.append(dataframe_para_texto(plaus_df, max_linhas=50))

    escrever_secao(linhas_saida, "6. ANÁLISE ORIENTADA À VARIÁVEL ALVO - NUMÉRICAS")
    if target_num is not None:
        linhas_saida.append(dataframe_para_texto(target_num, max_linhas=200))
    else:
        linhas_saida.append("A variável alvo indicada não existe ou não é binária.\n")

    escrever_secao(linhas_saida, "7. ANÁLISE ORIENTADA À VARIÁVEL ALVO - CATEGÓRICAS")
    if target_cat:
        for col, tab in target_cat.items():
            linhas_saida.append(dataframe_para_texto(tab.reset_index(), f"{col} vs {args.target}", max_linhas=50))
    else:
        linhas_saida.append("Não foi possível gerar tabelas relativas à variável alvo.\n")

    escrever_secao(linhas_saida, "8. TABELAS DE CONTINGÊNCIA RELEVANTES")
    if contingencias:
        for nome, tab in contingencias.items():
            linhas_saida.append(dataframe_para_texto(tab.reset_index(), nome, max_linhas=50))
    else:
        linhas_saida.append("Não foi possível gerar tabelas de contingência relevantes.\n")

    escrever_secao(linhas_saida, "9. OBSERVAÇÕES AUTOMÁTICAS")
    for i, obs in enumerate(observacoes, start=1):
        linhas_saida.append(f"{i}. {obs}")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "10. PONTOS A LEVAR PARA A PREPARAÇÃO DOS DADOS")
    linhas_saida.append("- Rever tratamento de outliers antes da modelação, sobretudo em height_cm, weight_kg, bmi e consumos.")
    linhas_saida.append("- Confirmar a interpretação e possível codificação ordinal de general_health, checkup e age_category.")
    linhas_saida.append("- Decidir como tratar diabetes: manter 4 categorias, agrupar ou criar versão binária justificada.")
    linhas_saida.append("- Considerar o desequilíbrio da variável heart_disease na fase supervisionada.")
    linhas_saida.append("- Avaliar impacto de variáveis correlacionadas e possíveis redundâncias.")
    linhas_saida.append("- Registar que a ausência de missing values não elimina a necessidade de verificar plausibilidade e consistência.")
    linhas_saida.append("")

    texto_final = "\n".join(linhas_saida)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(texto_final)

    print(f"Relatório criado com sucesso em: {output_path.resolve()}")


if __name__ == "__main__":
    main()