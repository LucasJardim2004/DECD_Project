"""
Divisão de dados para Decision Tree.
Usa o dataset numérico sem normalização e cria:
- 85% para treino
- 15% para teste

A divisão é estratificada pela variável alvo heart_disease/Heart_Disease.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_CANDIDATES = ["heart_disease", "Heart_Disease"]
TEST_SIZE = 0.15
RANDOM_STATE = 42


def resolve_target_column(columns: pd.Index) -> str:
    """Resolve o nome real da coluna alvo no dataset."""
    col_map = {c.lower(): c for c in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]

    raise ValueError(
        "Coluna alvo não encontrada. Esperado: heart_disease ou Heart_Disease."
    )


def print_target_balance(df: pd.DataFrame, target_col: str, label: str) -> None:
    """Mostra o equilíbrio da variável alvo para um dataframe."""
    counts = df[target_col].value_counts().reindex([0, 1], fill_value=0)
    total = len(df)

    sem_doenca = int(counts.loc[0])
    com_doenca = int(counts.loc[1])

    pct_sem = (sem_doenca / total) * 100 if total else 0.0
    pct_com = (com_doenca / total) * 100 if total else 0.0

    print(f"\nDistribuição de {target_col} em {label}:")
    print(f"  - Sem problemas cardíacos (0): {sem_doenca} ({pct_sem:.2f}%)")
    print(f"  - Com problemas cardíacos (1): {com_doenca} ({pct_com:.2f}%)")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "CVD_numeric.csv"
    train_file = base_dir / "CVD_train_85pct.csv"
    test_file = base_dir / "CVD_test_15pct.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {input_file}")

    dados = pd.read_csv(input_file)
    target_col = resolve_target_column(dados.columns)

    print(f"Total de registos no ficheiro original: {len(dados)}")
    print(f"Coluna alvo detetada: {target_col}")

    dados_treino, dados_teste = train_test_split(
        dados,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dados[target_col],
    )

    print(f"Registos no ficheiro de treino (85%): {len(dados_treino)}")
    print(f"Registos no ficheiro de teste (15%): {len(dados_teste)}")
    print_target_balance(dados_treino, target_col, "treino")
    print_target_balance(dados_teste, target_col, "teste")

    # Verifica que não há linhas repetidas entre treino e teste.
    assert len(set(dados_treino.index) & set(dados_teste.index)) == 0, (
        "Erro: há registos repetidos entre treino e teste!"
    )
    print("✓ Verificado: sem registos repetidos entre os dois ficheiros")

    dados_treino.to_csv(train_file, index=False)
    dados_teste.to_csv(test_file, index=False)

    print("\n✓ Ficheiros criados com sucesso:")
    print(f"  - {train_file.name}")
    print(f"  - {test_file.name}")


if __name__ == "__main__":
    main()
