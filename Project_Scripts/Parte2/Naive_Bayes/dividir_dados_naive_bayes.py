"""
Divisão de dados para Naive Bayes
Divide CVD_numeric_minmax.csv em 15% teste e 85% treino.
Cria também um conjunto de treino reduzido (metade dos registos de treino)
com proporção aproximada de 1:5 (com doença : sem doença).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

TARGET_COL = "Heart_Disease"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "output_preparacao" / "CVD_numeric_minmax.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR_REDUZIDA = OUTPUT_DIR / "Analise_Reduzida"

# Ler o ficheiro de dados
dados = pd.read_csv(INPUT_FILE)
print(f"Total de registos no ficheiro original: {len(dados)}")

# Dividir os dados: 15% teste, 85% treino
dados_teste, dados_treino = train_test_split(
    dados,
    test_size=0.85,
    random_state=42
)

print(f"Registos no ficheiro de teste (15%): {len(dados_teste)}")
print(f"Registos no ficheiro de treino (85%): {len(dados_treino)}")

# Verificar se não há sobreposição
assert len(set(dados_teste.index) & set(dados_treino.index)) == 0, "Erro: há registos repetidos!"
print("✓ Verificado: Sem registos repetidos entre os dois ficheiros")

# Guardar os ficheiros originais
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
dados_teste.to_csv(OUTPUT_DIR / "CVD_test_15pct.csv", index=False)
dados_treino.to_csv(OUTPUT_DIR / "CVD_train_85pct.csv", index=False)

print(f"\n✓ Ficheiros criados em {OUTPUT_DIR}:")
print("  - CVD_test_15pct.csv")
print("  - CVD_train_85pct.csv")

# --- Conjunto de treino reduzido (proporção 1:3) ---
treino_doenca = dados_treino[dados_treino[TARGET_COL] == 1]
treino_sem_doenca = dados_treino[dados_treino[TARGET_COL] == 0]

n_doenca = len(treino_doenca)
n_sem_doenca = n_doenca * 3

amostra_doenca = treino_doenca.sample(n=n_doenca, random_state=42)
amostra_sem_doenca = treino_sem_doenca.sample(n=n_sem_doenca, random_state=42)

dados_treino_reduzido = pd.concat([amostra_doenca, amostra_sem_doenca]).sample(frac=1, random_state=42)

proporcao = n_sem_doenca / n_doenca if n_doenca > 0 else float("inf")

print(f"\n--- Conjunto de treino reduzido ---")
print(f"  Com doença (1):    {n_doenca}")
print(f"  Sem doença (0):    {n_sem_doenca}")
print(f"  Total:             {len(dados_treino_reduzido)}")
print(f"  Proporção (0:1):   1:{proporcao:.2f}")

OUTPUT_DIR_REDUZIDA.mkdir(parents=True, exist_ok=True)
dados_teste.to_csv(OUTPUT_DIR_REDUZIDA / "CVD_test_15pct.csv", index=False)
dados_treino_reduzido.to_csv(OUTPUT_DIR_REDUZIDA / "CVD_train_reduzido.csv", index=False)

print(f"\n✓ Ficheiros criados em {OUTPUT_DIR_REDUZIDA}:")
print("  - CVD_test_15pct.csv")
print("  - CVD_train_reduzido.csv")
