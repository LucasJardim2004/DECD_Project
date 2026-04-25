# Relatório de Clustering KMeans

## Metodologia
- Dataset de entrada: `/Users/lucasjardim/Desktop/Pastas progamação/DECD_Proj/Project_Scripts/output_preparacao/CVD_numeric_zscore.csv`
- Observações usadas no treino: 308854
- Variáveis usadas: 18
- Algoritmo: KMeans com `k=9`
- `n_init`: 10
- `random_state`: 42
- Inertia final: 1761353,254
- Métrica interna do KMeans: distância euclidiana aos centróides
- Verificação da distância entre clusters: este script não utiliza distância máxima entre clusters; isso seria um critério de linkage hierárquico (complete linkage), não KMeans

## Resumo Global

| cluster | count | pct |
| --- | --- | --- |
| 0 | 31516 | 10.2 |
| 1 | 23729 | 7.68 |
| 2 | 47221 | 15.29 |
| 3 | 40702 | 13.18 |
| 4 | 60756 | 19.67 |
| 5 | 10820 | 3.5 |
| 6 | 29269 | 9.48 |
| 7 | 9712 | 3.14 |
| 8 | 55129 | 17.85 |

## Cluster 0
- Observações: 31516 (10,20%)
- Principais traços acima da média:
  - consumo de álcool: consumo de álcool acima da média (2,465)
  - altura: altura acima da média (0,271)
  - faixa etária: perfil etário mais avançado (1,308)
- Principais traços abaixo da média:
  - IMC: IMC abaixo da média (-0,260)
  - consumo de fruta: consumo de fruta abaixo da média (-0,122)
  - consumo de batata frita: consumo de batata frita abaixo da média (-0,059)
- Indicadores binários mais distintivos:
  - sexo masculino: 0,636 no cluster vs 0,481 no global
  - histórico de tabagismo: 0,553 no cluster vs 0,406 no global
  - prática de exercício: 0,834 no cluster vs 0,775 no global
- Leitura interpretativa:
  - consumo de álcool: consumo de álcool acima da média; altura: altura acima da média; IMC: IMC abaixo da média; faixa etária: perfil etário mais avançado.

## Cluster 1
- Observações: 23729 (7,68%)
- Principais traços acima da média:
  - altura: altura acima da média (0,389)
  - saúde geral: melhor avaliação de saúde geral (2,797)
  - sexo masculino: maior prevalência de sexo masculino (0,689)
- Principais traços abaixo da média:
  - frequência de check-up: menor regularidade de check-up (1,418)
  - faixa etária: perfil etário mais jovem (0,596)
  - consumo de vegetais verdes: consumo de vegetais verdes abaixo da média (-0,289)
- Indicadores binários mais distintivos:
  - artrite: 0,115 no cluster vs 0,327 no global
  - sexo masculino: 0,689 no cluster vs 0,481 no global
  - outro cancro: 0,025 no cluster vs 0,097 no global
- Leitura interpretativa:
  - frequência de check-up: menor regularidade de check-up; faixa etária: perfil etário mais jovem; altura: altura acima da média; consumo de vegetais verdes: consumo de vegetais verdes abaixo da média.

## Cluster 2
- Observações: 47221 (15,29%)
- Principais traços acima da média:
  - faixa etária: perfil etário mais avançado (1,587)
  - frequência de check-up: maior regularidade de check-up (3,898)
  - artrite: maior prevalência de artrite (0,600)
- Principais traços abaixo da média:
  - saúde geral: pior avaliação de saúde geral (1,283)
  - consumo de álcool: consumo de álcool abaixo da média (-0,456)
  - consumo de fruta: consumo de fruta abaixo da média (-0,367)
- Indicadores binários mais distintivos:
  - artrite: 0,600 no cluster vs 0,327 no global
  - prática de exercício: 0,567 no cluster vs 0,775 no global
  - histórico de tabagismo: 0,550 no cluster vs 0,406 no global
- Leitura interpretativa:
  - saúde geral: pior avaliação de saúde geral; faixa etária: perfil etário mais avançado; consumo de álcool: consumo de álcool abaixo da média; consumo de fruta: consumo de fruta abaixo da média.

## Cluster 3
- Observações: 40702 (13,18%)
- Principais traços acima da média:
  - consumo de fruta: consumo de fruta acima da média (1,728)
  - saúde geral: melhor avaliação de saúde geral (2,866)
  - frequência de check-up: maior regularidade de check-up (3,791)
- Principais traços abaixo da média:
  - consumo de batata frita: consumo de batata frita abaixo da média (-0,292)
  - IMC: IMC abaixo da média (-0,274)
  - altura: altura abaixo da média (-0,255)
- Indicadores binários mais distintivos:
  - sexo masculino: 0,317 no cluster vs 0,481 no global
  - histórico de tabagismo: 0,291 no cluster vs 0,406 no global
  - prática de exercício: 0,884 no cluster vs 0,775 no global
- Leitura interpretativa:
  - consumo de fruta: consumo de fruta acima da média; saúde geral: melhor avaliação de saúde geral; consumo de batata frita: consumo de batata frita abaixo da média; IMC: IMC abaixo da média.

## Cluster 4
- Observações: 60756 (19,67%)
- Principais traços acima da média:
  - altura: altura acima da média (1,005)
  - sexo masculino: maior prevalência de sexo masculino (0,948)
  - saúde geral: melhor avaliação de saúde geral (2,847)
- Principais traços abaixo da média:
  - consumo de fruta: consumo de fruta abaixo da média (-0,358)
  - consumo de álcool: consumo de álcool abaixo da média (-0,227)
  - consumo de vegetais verdes: consumo de vegetais verdes abaixo da média (-0,214)
- Indicadores binários mais distintivos:
  - sexo masculino: 0,948 no cluster vs 0,481 no global
  - artrite: 0,223 no cluster vs 0,327 no global
  - prática de exercício: 0,856 no cluster vs 0,775 no global
- Leitura interpretativa:
  - altura: altura acima da média; sexo masculino: maior prevalência de sexo masculino; consumo de fruta: consumo de fruta abaixo da média; saúde geral: melhor avaliação de saúde geral.

## Cluster 5
- Observações: 10820 (3,50%)
- Principais traços acima da média:
  - consumo de vegetais verdes: consumo de vegetais verdes acima da média (3,605)
  - consumo de fruta: consumo de fruta acima da média (0,854)
  - saúde geral: melhor avaliação de saúde geral (2,789)
- Principais traços abaixo da média:
  - IMC: IMC abaixo da média (-0,198)
  - consumo de batata frita: consumo de batata frita abaixo da média (-0,192)
  - altura: altura abaixo da média (-0,132)
- Indicadores binários mais distintivos:
  - sexo masculino: 0,378 no cluster vs 0,481 no global
  - prática de exercício: 0,870 no cluster vs 0,775 no global
  - histórico de tabagismo: 0,337 no cluster vs 0,406 no global
- Leitura interpretativa:
  - consumo de vegetais verdes: consumo de vegetais verdes acima da média; consumo de fruta: consumo de fruta acima da média; saúde geral: melhor avaliação de saúde geral; IMC: IMC abaixo da média.

## Cluster 6
- Observações: 29269 (9,48%)
- Principais traços acima da média:
  - IMC: IMC acima da média (2,017)
  - frequência de check-up: maior regularidade de check-up (3,806)
  - depressão: maior prevalência de depressão (0,371)
- Principais traços abaixo da média:
  - saúde geral: pior avaliação de saúde geral (1,784)
  - consumo de álcool: consumo de álcool abaixo da média (-0,383)
  - altura: altura abaixo da média (-0,376)
- Indicadores binários mais distintivos:
  - prática de exercício: 0,584 no cluster vs 0,775 no global
  - sexo masculino: 0,307 no cluster vs 0,481 no global
  - depressão: 0,371 no cluster vs 0,200 no global
- Leitura interpretativa:
  - IMC: IMC acima da média; saúde geral: pior avaliação de saúde geral; consumo de álcool: consumo de álcool abaixo da média; altura: altura abaixo da média.

## Cluster 7
- Observações: 9712 (3,14%)
- Principais traços acima da média:
  - consumo de batata frita: consumo de batata frita acima da média (3,777)
  - altura: altura acima da média (0,155)
  - consumo de vegetais verdes: consumo de vegetais verdes acima da média (0,151)
- Principais traços abaixo da média:
  - consumo de álcool: consumo de álcool abaixo da média (-0,202)
  - faixa etária: perfil etário mais jovem (0,946)
  - saúde geral: pior avaliação de saúde geral (2,390)
- Indicadores binários mais distintivos:
  - sexo masculino: 0,598 no cluster vs 0,481 no global
  - prática de exercício: 0,705 no cluster vs 0,775 no global
  - histórico de tabagismo: 0,466 no cluster vs 0,406 no global
- Leitura interpretativa:
  - consumo de batata frita: consumo de batata frita acima da média; consumo de álcool: consumo de álcool abaixo da média; faixa etária: perfil etário mais jovem; altura: altura acima da média.

## Cluster 8
- Observações: 55129 (17,85%)
- Principais traços acima da média:
  - saúde geral: melhor avaliação de saúde geral (3,149)
  - frequência de check-up: maior regularidade de check-up (3,824)
  - prática de exercício: maior prevalência de prática de exercício (0,829)
- Principais traços abaixo da média:
  - altura: altura abaixo da média (-0,740)
  - consumo de fruta: consumo de fruta abaixo da média (-0,404)
  - sexo masculino: menor prevalência de sexo masculino (0,101)
- Indicadores binários mais distintivos:
  - sexo masculino: 0,101 no cluster vs 0,481 no global
  - histórico de tabagismo: 0,300 no cluster vs 0,406 no global
  - artrite: 0,233 no cluster vs 0,327 no global
- Leitura interpretativa:
  - altura: altura abaixo da média; saúde geral: melhor avaliação de saúde geral; consumo de fruta: consumo de fruta abaixo da média; sexo masculino: menor prevalência de sexo masculino.
