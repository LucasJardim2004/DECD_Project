# 4. Normalização de Dados

## 4.1 Métodos Selecionados

Foram selecionados dois métodos de normalização para análise comparativa: Min-Max scaling e standardização Z-Score. A escolha destes métodos justifica-se pela sua relevância teórica e prática em contextos de análise de dados e machine learning. O método Min-Max normaliza as variáveis contínuas para o intervalo [0,1] através da transformação $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$, permitindo comparações diretas entre atributos com escalas e unidades distintas, preservando a forma da distribuição original. A standardização Z-Score, por sua vez, transforma os dados em torno de uma média de zero e desvio padrão unitário, utilizando a fórmula $z = \frac{x - \mu}{\sigma}$, mantendo a informação relativa de cada observação relativamente à dispersão populacional e facilitando a identificação de valores atípicos. Ambos os métodos são amplamente utilizados em análise exploratória de dados (EDA) e servem como base para algoritmos de clustering, regressão e classificação que apresentam sensibilidade à escala das variáveis.

## 4.2 Comparação e Impacto das Normalizações

A análise comparativa entre o formato numérico bruto, Min-Max e Z-Score revela diferenças significativas nas características estatísticas e distribucionais dos dados. O método Min-Max mantém a variabilidade relativa mas força convergência para o intervalo [0,1], resultando em comparabilidade direta entre variáveis; por exemplo, a altura (Height_cm) e peso (Weight_kg), originalmente com escalas 91-241 cm e 24,95-293,02 kg, respetivamente, são reduzidas ao intervalo [0,1], permitindo análises conjuntas sem enviesamento de escala. Em contraste, o Z-Score preserva melhor a forma da distribuição original e a magnitude das diferenças entre observações, mantendo valores negativos para observações abaixo da média e positivos acima dela, o que facilita a deteção de outliers e a compreensão da variabilidade intrínseca de cada variável. Observações nos boxplots e histogramas mostram que o Min-Max reduz visualmente o espaço de representação de forma homogénea, enquanto o Z-Score amplia variáveis com baixa dispersão relativa (como Alcohol_Consumption, com σ ≈ 8,20) e comprime aquelas com elevada dispersão relativa. A escolha entre ambos depende do contexto: Min-Max é preferível para algoritmos baseados em distância (e.g., K-Means, KNN) onde a magnitude absoluta é crítica, enquanto Z-Score é recomendado para regressão linear, análise de correlação e deteção de anomalias, onde a padronização em torno da média oferece propriedades estatísticas mais adequadas.

## 4.3 Ficheiros de Saída

- `CVD_numeric_minmax.csv` — Dataset com todas as variáveis contínuas normalizadas via Min-Max
- `CVD_numeric_zscore.csv` — Dataset com todas as variáveis contínuas normalizadas via Z-Score
- `stats_normalization_z-score.csv` — Tabela descritiva comparando min, max, mean, std, median para as três versões (numérica, MinMax, Z-Score)

## 4.4 Visualizações Associadas

As seguintes visualizações documentam o impacto da normalização:
- `boxplots_compare_z-score.png` — Comparação conjunta de todas as variáveis contínuas nas três versões
- `dist_bmi_z-score.png`, `dist_height-cm_z-score.png`, `dist_weight-kg_z-score.png`, `dist_alcohol-consumption_z-score.png` — Histogramas por variável selecionada, mostrando mudanças distributivas com cada normalização
