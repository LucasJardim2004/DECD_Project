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


def escrever_secao(linhas_saida, titulo):
    linhas_saida.append("=" * 100)
    linhas_saida.append(titulo)
    linhas_saida.append("=" * 100)


def inferir_tipo_analitico(serie, nome_coluna=None, max_unicos_categorica=15):
    nome = nome_coluna.lower() if nome_coluna else ""

    n_unicos = serie.nunique(dropna=True)
    dtype = str(serie.dtype)

    nomes_binarios = {
        "heart_disease", "skin_cancer", "other_cancer", "depression",
        "arthritis", "exercise", "smoking_history", "sex"
    }
    nomes_ordinais = {
        "general_health", "checkup", "age_category"
    }
    nomes_continuos = {
        "height_cm", "weight_kg", "bmi",
        "alcohol_consumption", "fruit_consumption",
        "green_vegetables_consumption", "friedpotato_consumption"
    }

    if nome in nomes_binarios:
        if nome == "sex":
            return "categórica binária"
        return "binária"

    if nome in nomes_ordinais:
        return "categórica ordinal"

    if nome in nomes_continuos:
        return "numérica contínua"

    if pd.api.types.is_bool_dtype(serie):
        return "binária"

    if pd.api.types.is_numeric_dtype(serie):
        if n_unicos <= 2:
            return "binária"
        elif n_unicos <= max_unicos_categorica:
            return "numérica discreta / possivelmente categórica"
        else:
            return "numérica contínua"

    if dtype in {"object", "string"}:
        if n_unicos <= 2:
            return "binária"
        elif n_unicos <= max_unicos_categorica:
            return "categórica nominal/ordinal"
        else:
            return "texto / categórica de alta cardinalidade"

    return "não determinado"


def exemplos_valores(serie, limite=10):
    valores = serie.dropna().unique().tolist()
    valores = [formatar_valor(v) for v in valores[:limite]]
    return valores


def descrever_variavel(coluna, serie, max_valores_listados=15):
    total = len(serie)
    missing = int(serie.isna().sum())
    missing_pct = (missing / total * 100) if total > 0 else 0
    n_unicos = int(serie.nunique(dropna=True))
    dtype = str(serie.dtype)
    tipo_analitico = inferir_tipo_analitico(serie, coluna)

    descricao = {
        "coluna": coluna,
        "dtype": dtype,
        "tipo_analitico_sugerido": tipo_analitico,
        "total_registos": total,
        "missing": missing,
        "missing_pct": round(missing_pct, 2),
        "unicos_sem_na": n_unicos,
    }

    if n_unicos > 0:
        moda = serie.mode(dropna=True)
        descricao["moda"] = formatar_valor(moda.iloc[0]) if not moda.empty else "N/A"
        descricao["exemplos_valores"] = exemplos_valores(serie, limite=max_valores_listados)
    else:
        descricao["moda"] = "N/A"
        descricao["exemplos_valores"] = []

    if pd.api.types.is_numeric_dtype(serie):
        descricao["min"] = round(float(serie.min()), 4) if serie.notna().any() else None
        descricao["q1"] = round(float(serie.quantile(0.25)), 4) if serie.notna().any() else None
        descricao["mediana"] = round(float(serie.median()), 4) if serie.notna().any() else None
        descricao["media"] = round(float(serie.mean()), 4) if serie.notna().any() else None
        descricao["q3"] = round(float(serie.quantile(0.75)), 4) if serie.notna().any() else None
        descricao["max"] = round(float(serie.max()), 4) if serie.notna().any() else None
        descricao["desvio_padrao"] = round(float(serie.std()), 4) if serie.notna().any() else None
    else:
        contagens = serie.value_counts(dropna=False)
        top_k = contagens.head(max_valores_listados)

        descricao["valores_e_contagens"] = [
            {
                "valor": formatar_valor(valor),
                "contagem": int(contagem),
                "percentagem": round((contagem / total) * 100, 2) if total > 0 else 0.0
            }
            for valor, contagem in top_k.items()
        ]

    descricao["observacoes"] = gerar_observacoes_variavel(coluna, serie, descricao)
    return descricao


def gerar_observacoes_variavel(coluna, serie, descricao):
    obs = []
    nome = coluna.lower()
    tipo = descricao["tipo_analitico_sugerido"]
    total = descricao["total_registos"]
    n_unicos = descricao["unicos_sem_na"]

    if descricao["missing"] == 0:
        obs.append("Sem valores em falta.")
    else:
        obs.append(f"Apresenta {descricao['missing']} valores em falta ({descricao['missing_pct']:.2f}%).")

    if tipo == "binária" or tipo == "categórica binária":
        contagens = serie.value_counts(dropna=True)
        if len(contagens) == 2:
            percentagem_top = (contagens.iloc[0] / total) * 100 if total > 0 else 0
            obs.append(f"Variável binária com classe dominante de {percentagem_top:.2f}%.")

    if tipo == "categórica ordinal":
        obs.append("Deve ser tratada respeitando a ordem natural das categorias.")

    if tipo == "numérica contínua":
        minimo = descricao.get("min")
        maximo = descricao.get("max")
        media = descricao.get("media")
        mediana = descricao.get("mediana")

        if minimo is not None and maximo is not None:
            obs.append(f"Intervalo observado: [{minimo}, {maximo}].")

        if media is not None and mediana is not None:
            if abs(media - mediana) > 0.15 * abs(media) if media != 0 else abs(mediana) > 0:
                obs.append("Pode apresentar assimetria, pois média e mediana diferem de forma visível.")

    if n_unicos == 1:
        obs.append("Variável constante.")
    elif 1 < n_unicos <= 5 and not pd.api.types.is_numeric_dtype(serie):
        obs.append("Baixa cardinalidade, fácil de resumir integralmente.")
    elif n_unicos > 50 and not pd.api.types.is_numeric_dtype(serie):
        obs.append("Cardinalidade elevada para variável textual/categórica.")

    if nome == "diabetes":
        obs.append("Inclui categorias clinicamente distintas; não deve ser reduzida sem justificação.")
    if nome == "age_category":
        obs.append("Representa grupos etários; variável ordinal.")
    if nome == "general_health":
        obs.append("Autoavaliação do estado de saúde; variável ordinal.")
    if nome == "checkup":
        obs.append("Representa tempo desde último check-up; variável ordinal.")
    if nome in {"height_cm", "weight_kg", "bmi"}:
        obs.append("Convém validar plausibilidade dos extremos na fase de qualidade dos dados.")
    if nome == "heart_disease":
        obs.append("Possível variável alvo para classificação supervisionada.")
    if nome in {"alcohol_consumption", "fruit_consumption", "green_vegetables_consumption", "friedpotato_consumption"}:
        obs.append("Convém confirmar unidade/escala usada no dicionário de dados.")

    return obs


def criar_tabela_resumo_variaveis(df):
    linhas = []

    for col in df.columns:
        serie = df[col]
        linhas.append({
            "coluna": col,
            "dtype": str(serie.dtype),
            "tipo_analitico_sugerido": inferir_tipo_analitico(serie, col),
            "missing": int(serie.isna().sum()),
            "missing_pct": round((serie.isna().mean() * 100), 2),
            "unicos_sem_na": int(serie.nunique(dropna=True)),
            "moda": formatar_valor(serie.mode(dropna=True).iloc[0]) if not serie.mode(dropna=True).empty else "N/A"
        })

    return pd.DataFrame(linhas)


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


def main():
    parser = argparse.ArgumentParser(
        description="Relatório da fase Describe Data (Data Description Report)."
    )
    parser.add_argument("input_file", help="Caminho para o CSV")
    parser.add_argument(
        "-o",
        "--output",
        default="describe_data_report.txt",
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
        "--max-valores-listados",
        type=int,
        default=15,
        help="Número máximo de valores a listar por variável categórica"
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

    linhas_saida = []

    escrever_secao(linhas_saida, "FASE 1 - DATA UNDERSTANDING | DESCRIBE DATA REPORT")
    linhas_saida.append(f"Data/hora de execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    linhas_saida.append(f"Ficheiro analisado: {input_path.name}")
    linhas_saida.append(f"Caminho: {input_path.resolve()}")
    linhas_saida.append(f"Encoding usado: {encoding_usado}")
    linhas_saida.append(f"Separador usado: {repr(separador_usado)}")
    linhas_saida.append(f"Dimensão do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
    linhas_saida.append(f"Memória ocupada: {bytes_humanos(df.memory_usage(deep=True).sum())}")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "1. RESUMO GLOBAL DAS VARIÁVEIS")
    tabela_resumo = criar_tabela_resumo_variaveis(df)
    linhas_saida.append(dataframe_para_texto(tabela_resumo, max_linhas=len(tabela_resumo)))

    escrever_secao(linhas_saida, "2. DESCRIÇÃO DETALHADA DE CADA VARIÁVEL")

    for col in df.columns:
        descricao = descrever_variavel(
            coluna=col,
            serie=df[col],
            max_valores_listados=args.max_valores_listados
        )

        linhas_saida.append("-" * 100)
        linhas_saida.append(f"VARIÁVEL: {descricao['coluna']}")
        linhas_saida.append("-" * 100)
        linhas_saida.append(f"Tipo técnico (pandas): {descricao['dtype']}")
        linhas_saida.append(f"Tipo analítico sugerido: {descricao['tipo_analitico_sugerido']}")
        linhas_saida.append(f"Total de registos: {descricao['total_registos']}")
        linhas_saida.append(f"Valores em falta: {descricao['missing']} ({descricao['missing_pct']:.2f}%)")
        linhas_saida.append(f"Número de valores únicos (sem NA): {descricao['unicos_sem_na']}")
        linhas_saida.append(f"Moda: {descricao['moda']}")

        if pd.api.types.is_numeric_dtype(df[col]):
            linhas_saida.append(f"Mínimo: {descricao.get('min')}")
            linhas_saida.append(f"1º quartil (Q1): {descricao.get('q1')}")
            linhas_saida.append(f"Mediana: {descricao.get('mediana')}")
            linhas_saida.append(f"Média: {descricao.get('media')}")
            linhas_saida.append(f"3º quartil (Q3): {descricao.get('q3')}")
            linhas_saida.append(f"Máximo: {descricao.get('max')}")
            linhas_saida.append(f"Desvio padrão: {descricao.get('desvio_padrao')}")
            linhas_saida.append(f"Exemplos de valores observados: {', '.join(descricao['exemplos_valores'])}")
        else:
            linhas_saida.append("Valores/categorias observados mais frequentes:")
            for item in descricao.get("valores_e_contagens", []):
                linhas_saida.append(
                    f" - {item['valor']}: {item['contagem']} registos ({item['percentagem']:.2f}%)"
                )
            linhas_saida.append(f"Exemplos de valores observados: {', '.join(descricao['exemplos_valores'])}")

        linhas_saida.append("Observações automáticas:")
        for obs in descricao["observacoes"]:
            linhas_saida.append(f" - {obs}")
        linhas_saida.append("")

    escrever_secao(linhas_saida, "3. CLASSIFICAÇÃO ANALÍTICA SUGERIDA DAS VARIÁVEIS")
    linhas_saida.append("Sugestão inicial para apoiar a escrita do relatório e a fase seguinte:")
    linhas_saida.append("")

    for col in df.columns:
        tipo = inferir_tipo_analitico(df[col], col)
        linhas_saida.append(f" - {col}: {tipo}")
    linhas_saida.append("")

    escrever_secao(linhas_saida, "4. NOTAS PARA O RELATÓRIO")
    linhas_saida.append("Esta secção pode ser reaproveitada para o relatório escrito:")
    linhas_saida.append("")
    linhas_saida.append(" - O dataset contém um conjunto de variáveis categóricas e numéricas relacionadas com saúde, comportamentos e características demográficas.")
    linhas_saida.append(" - Variáveis como general_health, checkup e age_category devem ser tratadas como ordinais.")
    linhas_saida.append(" - Variáveis como heart_disease, exercise, depression, arthritis e smoking_history têm natureza binária.")
    linhas_saida.append(" - Variáveis como height_cm, weight_kg, bmi e consumos alimentares são numéricas contínuas.")
    linhas_saida.append(" - A variável diabetes requer atenção especial, pois apresenta mais do que duas categorias clinicamente distintas.")
    linhas_saida.append(" - Esta descrição prepara o trabalho para a fase Explore Data e para a posterior preparação dos dados.")
    linhas_saida.append("")

    texto_final = "\n".join(linhas_saida)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(texto_final)

    print(f"Relatório criado com sucesso em: {output_path.resolve()}")


if __name__ == "__main__":
    main()