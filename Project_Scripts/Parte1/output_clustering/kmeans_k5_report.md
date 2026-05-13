# Relatório de Clustering KMeans

## Metodologia
- Dataset de entrada: `/Users/lucasjardim/Desktop/Pastas progamação/DECD_Proj/Project_Scripts/output_preparacao/CVD_numeric_zscore.csv`
- Observações usadas no treino: 308854
- Variáveis usadas: 18
- Algoritmo: KMeans com `k=5`
- `n_init`: 10
- `random_state`: 42
- Inertia final: 2156442,283
- Métrica interna do KMeans: distância euclidiana aos centróides
- Verificação da distância entre clusters: este script não utiliza distância máxima entre clusters; isso seria um critério de linkage hierárquico (complete linkage), não KMeans

## Resumo Global

| cluster | count | pct |
| --- | --- | --- |
| 0 | 35375 | 11.45 |
| 1 | 41883 | 13.56 |
| 2 | 66254 | 21.45 |
| 3 | 82488 | 26.71 |
| 4 | 82854 | 26.83 |

## Cluster 0
- Observações: 35375 (11,45%)
- Principais traços acima da média:
  - consumo de álcool: consumo de álcool acima da média (2,392)
  - altura: altura acima da média (0,284)
  - saúde geral: melhor avaliação de saúde geral (2,727)
- Principais traços abaixo da média:
  - IMC: IMC abaixo da média (-0,255)
  - consumo de fruta: consumo de fruta abaixo da média (-0,117)
  - frequência de check-up: menor regularidade de check-up (3,568)
- Indicadores binários mais distintivos:
  - sexo masculino: 0,638 no cluster vs 0,481 no global
  - histórico de tabagismo: 0,548 no cluster vs 0,406 no global
  - prática de exercício: 0,836 no cluster vs 0,775 no global
- Leitura interpretativa:
  - consumo de álcool: consumo de álcool acima da média; altura: altura acima da média; IMC: IMC abaixo da média; saúde geral: melhor avaliação de saúde geral.

## Cluster 1
- Observações: 41883 (13,56%)
- Principais traços acima da média:
  - consumo de fruta: consumo de fruta acima da média (1,659)
  - consumo de vegetais verdes: consumo de vegetais verdes acima da média (1,231)
  - saúde geral: melhor avaliação de saúde geral (2,837)
- Principais traços abaixo da média:
  - IMC: IMC abaixo da média (-0,242)
  - consumo de álcool: consumo de álcool abaixo da média (-0,213)
  - consumo de batata frita: consumo de batata frita abaixo da média (-0,134)
- Indicadores binários mais distintivos:
  - prática de exercício: 0,887 no cluster vs 0,775 no global
  - sexo masculino: 0,374 no cluster vs 0,481 no global
  - histórico de tabagismo: 0,311 no cluster vs 0,406 no global
- Leitura interpretativa:
  - consumo de fruta: consumo de fruta acima da média; consumo de vegetais verdes: consumo de vegetais verdes acima da média; saúde geral: melhor avaliação de saúde geral; IMC: IMC abaixo da média.

## Cluster 2
- Observações: 66254 (21,45%)
- Principais traços acima da média:
  - IMC: IMC acima da média (0,965)
  - artrite: maior prevalência de artrite (0,562)
  - frequência de check-up: maior regularidade de check-up (3,823)
- Principais traços abaixo da média:
  - saúde geral: pior avaliação de saúde geral (1,374)
  - consumo de álcool: consumo de álcool abaixo da média (-0,431)
  - consumo de vegetais verdes: consumo de vegetais verdes abaixo da média (-0,309)
- Indicadores binários mais distintivos:
  - artrite: 0,562 no cluster vs 0,327 no global
  - prática de exercício: 0,548 no cluster vs 0,775 no global
  - depressão: 0,347 no cluster vs 0,200 no global
- Leitura interpretativa:
  - saúde geral: pior avaliação de saúde geral; IMC: IMC acima da média; consumo de álcool: consumo de álcool abaixo da média; consumo de vegetais verdes: consumo de vegetais verdes abaixo da média.

## Cluster 3
- Observações: 82488 (26,71%)
- Principais traços acima da média:
  - saúde geral: melhor avaliação de saúde geral (2,900)
  - frequência de check-up: maior regularidade de check-up (3,718)
  - faixa etária: perfil etário mais avançado (1,151)
- Principais traços abaixo da média:
  - altura: altura abaixo da média (-0,772)
  - IMC: IMC abaixo da média (-0,401)
  - sexo masculino: menor prevalência de sexo masculino (0,103)
- Indicadores binários mais distintivos:
  - sexo masculino: 0,103 no cluster vs 0,481 no global
  - histórico de tabagismo: 0,331 no cluster vs 0,406 no global
  - doença cardíaca: 0,043 no cluster vs 0,081 no global
- Leitura interpretativa:
  - altura: altura abaixo da média; IMC: IMC abaixo da média; sexo masculino: menor prevalência de sexo masculino; saúde geral: melhor avaliação de saúde geral.

## Cluster 4
- Observações: 82854 (26,83%)
- Principais traços acima da média:
  - altura: altura acima da média (0,910)
  - sexo masculino: maior prevalência de sexo masculino (0,917)
  - saúde geral: melhor avaliação de saúde geral (2,848)
- Principais traços abaixo da média:
  - consumo de fruta: consumo de fruta abaixo da média (-0,332)
  - frequência de check-up: menor regularidade de check-up (3,338)
  - faixa etária: perfil etário mais jovem (0,843)
- Indicadores binários mais distintivos:
  - sexo masculino: 0,917 no cluster vs 0,481 no global
  - artrite: 0,187 no cluster vs 0,327 no global
  - depressão: 0,129 no cluster vs 0,200 no global
- Leitura interpretativa:
  - altura: altura acima da média; sexo masculino: maior prevalência de sexo masculino; consumo de fruta: consumo de fruta abaixo da média; saúde geral: melhor avaliação de saúde geral.
