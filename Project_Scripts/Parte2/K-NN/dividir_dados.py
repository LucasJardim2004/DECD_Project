import pandas as pd
from sklearn.model_selection import train_test_split

# Ler o ficheiro de dados
dados = pd.read_csv('Project_Scripts/output_preparacao/CVD_numeric_zscore.csv')

print(f"Total de registos no ficheiro original: {len(dados)}")

# Dividir os dados: 15% teste, 85% treino
# random_state garante reprodutibilidade
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

# Guardar os ficheiros
dados_teste.to_csv('Project_Scripts/output_preparacao/CVD_test_15pct.csv', index=False)
dados_treino.to_csv('Project_Scripts/output_preparacao/CVD_train_85pct.csv', index=False)

print("\n✓ Ficheiros criados com sucesso:")
print("  - CVD_test_15pct.csv")
print("  - CVD_train_85pct.csv")
