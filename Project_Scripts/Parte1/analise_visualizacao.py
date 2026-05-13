"""
Analise de Visualizacao - CVD_cleaned.csv

Gera apenas os graficos pedidos pela classificacao das variaveis:
- categorica ordinal: histogramas
- numerica continua: histogramas
- binaria: circular

A variavel diabetes fica numa secao separada porque a classificacao foi
indicada como nao determinada.
"""

from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme()
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 22,
    }
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
ORDINAL_DIR = OUTPUT_DIR / "categoricas_ordinais"
CONTINUA_DIR = OUTPUT_DIR / "numericas_continuas"
BINARIA_DIR = OUTPUT_DIR / "binarias"
INDETERMINADA_DIR = OUTPUT_DIR / "indeterminadas"

for directory in [OUTPUT_DIR, ORDINAL_DIR, CONTINUA_DIR, BINARIA_DIR, INDETERMINADA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", str(name).strip().lower())
    return cleaned.strip("_") or "sem_nome"


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


def save_histogram(df: pd.DataFrame, column: str, base_dir: Path, order: list[str] | None = None) -> None:
    col_dir = base_dir / safe_name(column)
    col_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    values = df[column].dropna()

    if order is not None:
        categorical = pd.Categorical(values, categories=order, ordered=True)
        codes = pd.Series(categorical.codes)
        codes = codes[codes >= 0]
        plt.hist(codes, bins=np.arange(-0.5, len(order) + 0.5, 1), edgecolor="black")
        plt.xticks(range(len(order)), order, rotation=45, ha="right")
    else:
        numeric_values = pd.to_numeric(values, errors="coerce").dropna()
        plt.hist(numeric_values, bins=20, edgecolor="black")

    plt.title(f"Histograma - {column}", fontsize=20)
    plt.xlabel(column, fontsize=17)
    plt.ylabel("Frequencia", fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(col_dir / "histograma.png", dpi=100, bbox_inches="tight")
    plt.close()


def save_pie_chart(df: pd.DataFrame, column: str, base_dir: Path) -> None:
    col_dir = base_dir / safe_name(column)
    col_dir.mkdir(parents=True, exist_ok=True)

    counts = df[column].fillna("Missing").value_counts(dropna=False)

    plt.figure(figsize=(8, 8))
    ax = counts.plot(kind="pie", autopct="%1.1f%%", textprops={"fontsize": 20})
    for text in ax.texts:
        text.set_fontsize(25)
    plt.title(f"Grafico circular - {column}", fontsize=20)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(col_dir / "grafico_circular.png", dpi=100, bbox_inches="tight")
    plt.close()


def save_value_counts(df: pd.DataFrame, column: str, base_dir: Path) -> None:
    col_dir = base_dir / safe_name(column)
    col_dir.mkdir(parents=True, exist_ok=True)

    counts = df[column].fillna("Missing").value_counts(dropna=False)
    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar", edgecolor="black")
    plt.title(f"Contagem - {column}", fontsize=20)
    plt.xlabel(column, fontsize=17)
    plt.ylabel("Frequencia", fontsize=17)
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(col_dir / "contagem.png", dpi=100, bbox_inches="tight")
    plt.close()


print("=" * 80)
print("ANALISE DE VISUALIZACAO")
print("=" * 80)

df = pd.read_csv(BASE_DIR / "CVD_cleaned.csv")

print(f"Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
print(f"Output: {OUTPUT_DIR}")

ordinal_cols = ["General_Health", "Checkup", "Age_Category"]
continuous_cols = [
    "Height_(cm)",
    "Weight_(kg)",
    "BMI",
    "Alcohol_Consumption",
    "Fruit_Consumption",
    "Green_Vegetables_Consumption",
    "FriedPotato_Consumption",
]
binary_cols = [
    "Exercise",
    "Heart_Disease",
    "Skin_Cancer",
    "Other_Cancer",
    "Depression",
    "Arthritis",
    "Sex",
    "Smoking_History",
]
indeterminate_cols = ["Diabetes"]

available_cols = set(df.columns)

print("\nGerando histogramas para variaveis ordinais...")
for col in ordinal_cols:
    if col in available_cols:
        save_histogram(df, col, ORDINAL_DIR, order=ORDINAL_ORDER.get(col))

print("Gerando histogramas para variaveis continuas...")
for col in continuous_cols:
    if col in available_cols:
        save_histogram(df, col, CONTINUA_DIR)

print("Gerando graficos circulares para variaveis binarias...")
for col in binary_cols:
    if col in available_cols:
        save_pie_chart(df, col, BINARIA_DIR)

print("Gerando grafico de contagem para a variavel com classificacao indeterminada...")
for col in indeterminate_cols:
    if col in available_cols:
        save_value_counts(df, col, INDETERMINADA_DIR)

print("\nResumo:")
print(f"- Ordinais: {', '.join([c for c in ordinal_cols if c in available_cols])}")
print(f"- Continuas: {', '.join([c for c in continuous_cols if c in available_cols])}")
print(f"- Binarias: {', '.join([c for c in binary_cols if c in available_cols])}")
print(f"- Indeterminadas: {', '.join([c for c in indeterminate_cols if c in available_cols])}")
print("\nConcluido. Todos os graficos foram guardados em subpastas dentro de output.")
