# Relatório do Modelo de Árvores de Decisão Melhorado

- Coluna alvo: Heart_Disease
- Entropia do conjunto de treino: 0.721928
- Limiar de decisão selecionado na validação: 0.40

## Melhores atributos por information gain
- Age_Category: IG=0.095287, threshold=1.500000
- General_Health: IG=0.083825, threshold=2.500000
- Arthritis: IG=0.052277, threshold=0.500000
- Diabetes: IG=0.049587, threshold=0.500000
- Smoking_History: IG=0.028371, threshold=0.500000
- Checkup: IG=0.026556, threshold=3.500000
- Exercise: IG=0.019741, threshold=0.500000
- Alcohol_Consumption: IG=0.016930, threshold=0.500000
- Other_Cancer: IG=0.015353, threshold=0.500000
- Skin_Cancer: IG=0.014088, threshold=0.500000

## Hiperparâmetros selecionados
- max_depth: 6
- min_samples_split: 20
- min_samples_leaf: 10
- class_weight: balanced

## Melhor validação interna
- validation_f2: 0.657755
- validation_recall: 0.789399
- validation_f1: 0.526142
- validation_accuracy: 0.715618

## Métricas no treino
- accuracy: 0.656678
- precision: 0.354767
- recall: 0.875241
- f1: 0.504885
- f2: 0.676689

## Métricas no teste
- accuracy: 0.619353
- precision: 0.161393
- recall: 0.883609
- f1: 0.272933
- f2: 0.466289
- pr_auc: 0.294307

## Observação
- O modelo usa class_weight='balanced' e limiar de decisão ajustado para favorecer recall, reduzindo falsos negativos.