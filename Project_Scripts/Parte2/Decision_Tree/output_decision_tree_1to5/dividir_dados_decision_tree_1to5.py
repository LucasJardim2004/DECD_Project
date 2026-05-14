"""
Criação de um novo dataset de treino para Decision Tree com proporção 1:5
(Doentes cardíacos : Sem doença cardíaca).

Este script:
- Lê o dataset de treino 85% já existente em output_decision_tree_85pct/
- Mantém todos os registos com Heart_Disease = 1
- Seleciona aleatoriamente registos com Heart_Disease = 0 até perfazer uma proporção
  aproximada de 1:5 no conjunto final
- Guarda o novo dataset de treino na pasta output_decision_tree_1to5/

O conjunto de teste permanece o mesmo:
- Project_Scripts/Parte2/Decision_Tree/CVD_test_15pct.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
INPUT_FILE = BASE_DIR / "output_decision_tree_85pct" / "CVD_train_85pct.csv"
OUTPUT_FILE = SCRIPT_DIR / "CVD_train_1to5.csv"
TARGET_COL_CANDIDATES = ["heart_disease", "Heart_Disease"]
RANDOM_STATE = 42
TARGET_RATIO = 5  # 1 doente para cada 5 registos no total


def resolve_target_column(columns: pd.Index) -> str:
    """Resolve o nome real da coluna alvo no dataset."""
    column_map = {column.lower(): column for column in columns}
    for candidate in TARGET_COL_CANDIDATES:
        if candidate.lower() in column_map:
            return column_map[candidate.lower()]

    raise ValueError(
        "Coluna alvo não encontrada. Esperado: heart_disease ou Heart_Disease."
    )


def print_balance(df: pd.DataFrame, target_col: str, label: str) -> None:
    """Mostra a distribuição da variável alvo."""
    counts = df[target_col].value_counts().reindex([0, 1], fill_value=0)
    total = len(df)

    sem_doenca = int(counts.loc[0])
    com_doenca = int(counts.loc[1])

    pct_sem = (sem_doenca / total) * 100 if total else 0.0
    pct_com = (com_doenca / total) * 100 if total else 0.0

    print(f"\nDistribuição em {label}:")
    print(f"  - Sem doença cardíaca (0): {sem_doenca} ({pct_sem:.2f}%)")
    print(f"  - Com doença cardíaca (1): {com_doenca} ({pct_com:.2f}%)")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Ficheiro de treino não encontrado: {INPUT_FILE}")

    dados = pd.read_csv(INPUT_FILE)
    target_col = resolve_target_column(dados.columns)

    print(f"Ficheiro de entrada: {INPUT_FILE}")
    print(f"Ficheiro de saída:   {OUTPUT_FILE}")
    print(f"Coluna alvo:         {target_col}")
    print(f"Total de registos originais: {len(dados)}")

    dados_doentes = dados[dados[target_col] == 1].copy()
    dados_sem_doenca = dados[dados[target_col] == 0].copy()

    n_doentes = len(dados_doentes)
    n_sem_doenca_necessario = n_doentes * (TARGET_RATIO - 1)

    if n_sem_doenca_necessario > len(dados_sem_doenca):
        raise ValueError(
            "Não há registos suficientes sem doença cardíaca para atingir a proporção 1:5."
        )

    print_balance(dados, target_col, "dataset original de treino 85%")
    print(f"\nObjetivo do novo dataset:")
    print(f"  - Doentes cardíacos (1): {n_doentes}")
    print(f"  - Sem doença cardíaca (0) necessários: {n_sem_doenca_necessario}")
    print(f"  - Total esperado: {n_doentes + n_sem_doenca_necessario}")

    amostra_sem_doenca = dados_sem_doenca.sample(
        n=n_sem_doenca_necessario,
        random_state=RANDOM_STATE,
    )

    dados_novos = pd.concat([dados_doentes, amostra_sem_doenca], axis=0)
    dados_novos = dados_novos.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dados_novos.to_csv(OUTPUT_FILE, index=False)

    print_balance(dados_novos, target_col, "novo dataset 1:5")
    print(f"\n✓ Novo dataset de treino criado com sucesso: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
