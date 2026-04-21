import pandas as pd
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
        valor_str = valor_str[: max_len - 3] + "..."
    return valor_str


def bytes_humanos(num_bytes):
    unidades = ["B", "KB", "MB", "GB", "TB"]
    valor = float(num_bytes)
    for unidade in unidades:
        if valor < 1024 or unidade == unidades[-1]:
            return f"{valor:.2f} {unidade}"
        valor /= 1024


def classificar_colunas(df, max_unicos_categorica=30):
    colunas_numericas = list(df.select_dtypes(include=["number"]).columns)
    colunas_texto = list(df.select_dtypes(include=["object", "string"]).columns)
    colunas_booleanas = list(df.select_dtypes(include=["bool"]).columns)

    colunas_categoricas = []
    colunas_suspeitas = []

    for col in df.columns:
        n_unicos = df[col].nunique(dropna=True)
        n_total = len(df)

        if col in colunas_texto or col in colunas_booleanas:
            colunas_categoricas.append(col)
        elif col in colunas_numericas and n_unicos <= max_unicos_categorica:
            colunas_categoricas.append(col)

        if n_total > 0 and 0 < n_unicos < n_total and df[col].dtype == "object":
            colunas_suspeitas.append(col)

    return {
        "numericas": colunas_numericas,
        "texto": colunas_texto,
        "booleanas": colunas_booleanas,
        "categoricas": sorted(set(colunas_categoricas)),
        "suspeitas_texto": colunas_suspeitas,
    }


def detetar_tipos_mistos(df, amostra=1000):
    colunas_tipos_mistos = []

    for col in df.columns:
        serie = df[col].dropna()
        if serie.empty:
            continue

        if len(serie) > amostra:
            serie = serie.sample(amostra, random_state=42)

        tipos = {type(v).__name__ for v in serie}
        if len(tipos) > 1:
            colunas_tipos_mistos.append((col, sorted(tipos)))

    return colunas_tipos_mistos


def detetar_missing_disfarcados(df):
    tokens_missing = {"", "na", "n/a", "null", "none", "unknown", "?", "-", "--"}
    resultado = {}

    for col in df.select_dtypes(include=["object", "string"]).columns:
        serie = df[col].dropna().astype(str).str.strip().str.lower()
        encontrados = serie[serie.isin(tokens_missing)]
        if not encontrados.empty:
            resultado[col] = encontrados.value_counts().to_dict()

    return resultado


def analisar_colunas_constantes(df):
    constantes = []
    quase_constantes = []

    for col in df.columns:
        contagens = df[col].value_counts(dropna=False)
        if contagens.empty:
            continue

        if len(contagens) == 1:
            constantes.append(col)
        else:
            freq_top = contagens.iloc[0] / len(df) if len(df) > 0 else 0
            if freq_top >= 0.95:
                quase_constantes.append((col, freq_top, formatar_valor(contagens.index[0])))

    return constantes, quase_constantes


def analisar_cardinalidade(df, limiar_ratio=0.9, limiar_absoluto=100):
    alta_cardinalidade = []

    for col in df.columns:
        n_unicos = df[col].nunique(dropna=True)
        ratio = (n_unicos / len(df)) if len(df) > 0 else 0

        if n_unicos >= limiar_absoluto or ratio >= limiar_ratio:
            alta_cardinalidade.append((col, n_unicos, ratio))

    return alta_cardinalidade


def obter_preview(df, n=5):
    return {
        "head": df.head(n),
        "tail": df.tail(n),
        "sample": df.sample(min(n, len(df)), random_state=42) if len(df) > 0 else df.head(0)
    }


def resumo_numerico(df):
    colunas_num = df.select_dtypes(include=["number"]).columns
    if len(colunas_num) == 0:
        return None

    resumo = df[colunas_num].describe().T
    resumo["mediana"] = df[colunas_num].median(numeric_only=True)
    resumo["missing"] = df[colunas_num].isna().sum()
    resumo["missing_pct"] = (df[colunas_num].isna().mean() * 100).round(2)
    return resumo


def resumo_categorico(df, max_unicos=30):
    linhas = []

    for col in df.columns:
        n_unicos = df[col].nunique(dropna=True)

        if df[col].dtype == "object" or str(df[col].dtype) == "string" or n_unicos <= max_unicos:
            contagens = df[col].value_counts(dropna=False)
            top_valor = contagens.index[0] if not contagens.empty else None
            top_freq = contagens.iloc[0] if not contagens.empty else 0

            linhas.append({
                "coluna": col,
                "dtype": str(df[col].dtype),
                "unicos_sem_na": n_unicos,
                "missing": int(df[col].isna().sum()),
                "top_valor": formatar_valor(top_valor),
                "top_frequencia": int(top_freq),
                "top_percentagem": round((top_freq / len(df)) * 100, 2) if len(df) > 0 else 0.0
            })

    if not linhas:
        return None

    return pd.DataFrame(linhas).sort_values(by=["unicos_sem_na", "coluna"])


def gerar_observacoes_automaticas(
    df,
    missing_por_coluna,
    duplicados,
    constantes,
    quase_constantes,
    alta_cardinalidade,
    tipos_mistos,
    missing_disfarcados
):
    obs = []

    if duplicados > 0:
        obs.append(f"Foram detetadas {duplicados} linhas duplicadas exatas.")
    else:
        obs.append("Não foram detetadas linhas duplicadas exatas.")

    colunas_com_missing = missing_por_coluna[missing_por_coluna > 0]
    if not colunas_com_missing.empty:
        top_missing = colunas_com_missing.sort_values(ascending=False).head(5)
        nomes = ", ".join(
            [f"{col} ({int(val)})" for col, val in top_missing.items()]
        )
        obs.append(f"Existem colunas com valores em falta. Maiores ocorrências: {nomes}.")
    else:
        obs.append("Não foram encontrados valores em falta.")

    if constantes:
        obs.append(f"Existem colunas constantes: {', '.join(constantes)}.")

    if quase_constantes:
        nomes = ", ".join([f"{col} ({freq*100:.2f}% no valor dominante)" for col, freq, _ in quase_constantes[:5]])
        obs.append(f"Existem colunas quase constantes: {nomes}.")

    if alta_cardinalidade:
        nomes = ", ".join([f"{col} ({n_unicos} únicos)" for col, n_unicos, _ in alta_cardinalidade[:5]])
        obs.append(f"Foram detetadas colunas com cardinalidade alta: {nomes}.")

    if tipos_mistos:
        nomes = ", ".join([f"{col} ({', '.join(tipos)})" for col, tipos in tipos_mistos[:5]])
        obs.append(f"Foram detetadas colunas com tipos mistos suspeitos: {nomes}.")

    if missing_disfarcados:
        nomes = ", ".join(list(missing_disfarcados.keys())[:5])
        obs.append(f"Foram encontrados possíveis missing values disfarçados em: {nomes}.")

    if not obs:
        obs.append("Não foram detetados problemas automáticos relevantes nesta análise inicial.")

    return obs


def dataframe_para_texto(df, titulo=None, max_linhas=20):
    linhas = []
    if titulo:
        linhas.append(titulo)
        linhas.append("-" * len(titulo))

    if df is None:
        linhas.append("Sem dados para apresentar.\n")
        return "\n".join(linhas)

    if len(df) > max_linhas:
        texto = df.head(max_linhas).to_string()
        linhas.append(texto)
        linhas.append(f"\n... (mostradas apenas {max_linhas} linhas de {len(df)})")
    else:
        linhas.append(df.to_string())

    linhas.append("")
    return "\n".join(linhas)


def escrever_secao(linhas_saida, titulo):
    linhas_saida.append("=" * 100)
    linhas_saida.append(titulo)
    linhas_saida.append("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Relatório inicial da Fase 1 de Data Understanding (Collect Initial Data)."
    )
    parser.add_argument("input_file", help="Caminho para o CSV")
    parser.add_argument(
        "-o",
        "--output",
        default="fase1_data_understanding.txt",
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
        help="Separador do CSV (opcional). Ex.: ',' ';' '\\t'"
    )
    parser.add_argument(
        "--max-unicos",
        type=int,
        default=30,
        help="Máximo de valores únicos para considerar uma coluna como categórica no resumo"
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

    # Normalização inicial
    df = normalizar_colunas(df)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(limpar_texto)

    # Métricas principais
    total_linhas, total_colunas = df.shape
    memoria_bytes = df.memory_usage(deep=True).sum()
    duplicados = int(df.duplicated().sum())
    perc_duplicados = (duplicados / total_linhas * 100) if total_linhas > 0 else 0

    classificacao = classificar_colunas(df, max_unicos_categorica=args.max_unicos)
    tipos_mistos = detetar_tipos_mistos(df)
    missing_disfarcados = detetar_missing_disfarcados(df)
    constantes, quase_constantes = analisar_colunas_constantes(df)
    alta_cardinalidade = analisar_cardinalidade(df)

    missing_por_coluna = df.isna().sum().sort_values(ascending=False)
    missing_pct_por_coluna = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    total_missing = int(df.isna().sum().sum())
    linhas_com_missing = int(df.isna().any(axis=1).sum())

    preview = obter_preview(df, n=5)
    resumo_num = resumo_numerico(df)
    resumo_cat = resumo_categorico(df, max_unicos=args.max_unicos)

    observacoes = gerar_observacoes_automaticas(
        df=df,
        missing_por_coluna=missing_por_coluna,
        duplicados=duplicados,
        constantes=constantes,
        quase_constantes=quase_constantes,
        alta_cardinalidade=alta_cardinalidade,
        tipos_mistos=tipos_mistos,
        missing_disfarcados=missing_disfarcados
    )

    linhas_saida = []

    escrever_secao(linhas_saida, "FASE 1 - DATA UNDERSTANDING | COLLECT INITIAL DATA REPORT")
    linhas_saida.append(f"Data/hora de execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "1. IDENTIFICAÇÃO DO DATASET")
    linhas_saida.append(f"Nome do ficheiro: {input_path.name}")
    linhas_saida.append(f"Caminho: {input_path.resolve()}")
    linhas_saida.append(f"Formato: {input_path.suffix.lower()}")
    linhas_saida.append(f"Encoding usado: {encoding_usado}")
    linhas_saida.append(f"Separador usado: {repr(separador_usado)}")
    linhas_saida.append("Estado do carregamento: SUCESSO")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "2. DIMENSÃO GERAL DOS DADOS")
    linhas_saida.append(f"Número de linhas: {total_linhas}")
    linhas_saida.append(f"Número de colunas: {total_colunas}")
    linhas_saida.append(f"Shape: {df.shape}")
    linhas_saida.append(f"Memória ocupada: {bytes_humanos(memoria_bytes)}")
    linhas_saida.append(f"Linhas duplicadas exatas: {duplicados} ({perc_duplicados:.2f}%)")
    linhas_saida.append(f"Linhas únicas: {total_linhas - duplicados}")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "3. ESTRUTURA DAS VARIÁVEIS")
    linhas_saida.append(f"Lista de colunas ({len(df.columns)}):")
    for col in df.columns:
        linhas_saida.append(f" - {col}")

    linhas_saida.append("")
    linhas_saida.append("Tipos pandas por coluna:")
    for col in df.columns:
        linhas_saida.append(f" - {col}: {df[col].dtype}")

    linhas_saida.append("")
    linhas_saida.append(f"Nº colunas numéricas: {len(classificacao['numericas'])}")
    linhas_saida.append(f"Nº colunas de texto: {len(classificacao['texto'])}")
    linhas_saida.append(f"Nº colunas booleanas: {len(classificacao['booleanas'])}")
    linhas_saida.append(f"Nº colunas categóricas (heurística): {len(classificacao['categoricas'])}")
    linhas_saida.append("")

    if tipos_mistos:
        linhas_saida.append("Colunas com tipos mistos suspeitos:")
        for col, tipos in tipos_mistos:
            linhas_saida.append(f" - {col}: {', '.join(tipos)}")
    else:
        linhas_saida.append("Colunas com tipos mistos suspeitos: nenhuma")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "4. PREVIEW DOS DADOS")
    linhas_saida.append(dataframe_para_texto(preview["head"], "Primeiras 5 linhas"))
    linhas_saida.append(dataframe_para_texto(preview["tail"], "Últimas 5 linhas"))
    linhas_saida.append(dataframe_para_texto(preview["sample"], "Amostra aleatória de 5 linhas"))

    escrever_secao(linhas_saida, "5. COMPLETUDE E VALORES EM FALTA")
    linhas_saida.append(f"Total de valores em falta no dataset: {total_missing}")
    linhas_saida.append(f"Número de linhas com pelo menos um missing: {linhas_com_missing}")
    linhas_saida.append(f"Número de colunas com missing: {int((missing_por_coluna > 0).sum())}")
    linhas_saida.append("")

    missing_df = pd.DataFrame({
        "missing": missing_por_coluna,
        "missing_pct": missing_pct_por_coluna
    })
    linhas_saida.append(dataframe_para_texto(missing_df, "Missing values por coluna", max_linhas=len(missing_df)))

    if missing_disfarcados:
        linhas_saida.append("Possíveis missing values disfarçados encontrados:")
        for col, valores in missing_disfarcados.items():
            linhas_saida.append(f" - {col}: {valores}")
    else:
        linhas_saida.append("Possíveis missing values disfarçados encontrados: nenhum")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "6. QUALIDADE ESTRUTURAL INICIAL")
    if constantes:
        linhas_saida.append("Colunas constantes:")
        for col in constantes:
            linhas_saida.append(f" - {col}")
    else:
        linhas_saida.append("Colunas constantes: nenhuma")

    linhas_saida.append("")

    if quase_constantes:
        linhas_saida.append("Colunas quase constantes (>=95% no valor dominante):")
        for col, freq, valor_dom in quase_constantes:
            linhas_saida.append(f" - {col}: valor dominante='{valor_dom}' ({freq*100:.2f}%)")
    else:
        linhas_saida.append("Colunas quase constantes: nenhuma")

    linhas_saida.append("")

    if alta_cardinalidade:
        linhas_saida.append("Colunas com alta cardinalidade:")
        for col, n_unicos, ratio in alta_cardinalidade:
            linhas_saida.append(f" - {col}: {n_unicos} valores únicos ({ratio*100:.2f}% das linhas)")
    else:
        linhas_saida.append("Colunas com alta cardinalidade: nenhuma")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "7. RESUMO ESTATÍSTICO INICIAL - VARIÁVEIS NUMÉRICAS")
    linhas_saida.append(dataframe_para_texto(resumo_num, max_linhas=50))

    escrever_secao(linhas_saida, "8. RESUMO DE COLUNAS CATEGÓRICAS / BAIXA CARDINALIDADE")
    linhas_saida.append(dataframe_para_texto(resumo_cat, max_linhas=100))

    escrever_secao(linhas_saida, "9. DISTRIBUIÇÃO DETALHADA DE VALORES POR COLUNA CATEGÓRICA")
    colunas_candidatas = []

    for col in df.columns:
        n_unicos = df[col].nunique(dropna=False)
        if df[col].dtype == "object" or str(df[col].dtype) == "string" or n_unicos <= args.max_unicos:
            colunas_candidatas.append(col)

    for col in colunas_candidatas:
        linhas_saida.append("-" * 100)
        linhas_saida.append(f"COLUNA: {col}")
        linhas_saida.append("-" * 100)

        contagens = df[col].value_counts(dropna=False)
        for valor, contagem in contagens.items():
            percentagem = (contagem / len(df)) * 100 if len(df) > 0 else 0
            valor_str = formatar_valor(valor)
            linhas_saida.append(f"{valor_str}: {contagem} registos ({percentagem:.2f}%)")
        linhas_saida.append("")

    escrever_secao(linhas_saida, "10. OBSERVAÇÕES AUTOMÁTICAS")
    for i, obs in enumerate(observacoes, start=1):
        linhas_saida.append(f"{i}. {obs}")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "11. PONTOS A VERIFICAR MANUALMENTE NA FASE SEGUINTE")
    linhas_saida.append("- Confirmar o tipo analítico de cada variável: nominal, ordinal, binária ou contínua.")
    linhas_saida.append("- Verificar se existem códigos especiais a representar missing values.")
    linhas_saida.append("- Confirmar se colunas com alta cardinalidade são identificadores ou variáveis úteis.")
    linhas_saida.append("- Rever colunas quase constantes para perceber se têm utilidade analítica.")
    linhas_saida.append("- Validar intervalos plausíveis das variáveis numéricas.")
    linhas_saida.append("- Preparar a tabela de descrição semântica das variáveis para a fase 'Describe Data'.")
    linhas_saida.append("")

    texto_final = "\n".join(linhas_saida)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(texto_final)

    print(f"Relatório criado com sucesso em: {output_path.resolve()}")


if __name__ == "__main__":
    main()