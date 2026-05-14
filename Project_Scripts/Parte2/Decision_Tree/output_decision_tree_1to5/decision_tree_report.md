# Relatório do Modelo de Árvores de Decisão

- Coluna alvo: Heart_Disease
- Entropia do conjunto de treino: 0.721928

## Melhores atributos por information gain
- Age_Category: IG=0.061515, threshold=1.500000
- General_Health: IG=0.055318, threshold=1.500000
- Diabetes: IG=0.034385, threshold=0.500000
- Arthritis: IG=0.034092, threshold=0.500000
- Smoking_History: IG=0.018239, threshold=0.500000
- Checkup: IG=0.015938, threshold=3.500000
- Exercise: IG=0.013077, threshold=0.500000
- Alcohol_Consumption: IG=0.010827, threshold=0.500000
- Other_Cancer: IG=0.010517, threshold=0.500000
- Skin_Cancer: IG=0.009617, threshold=0.500000

## Hiperparâmetros selecionados
- max_depth: 6
- min_samples_split: 2
- min_samples_leaf: 1

## Melhor validação interna
- validation_f1: 0.438372
- validation_recall: 0.362780
- validation_accuracy: 0.814087

## Métricas no treino
- accuracy: 0.821060
- precision: 0.605117
- recall: 0.303086
- f1: 0.403880

## Métricas no teste
- accuracy: 0.896976
- precision: 0.342726
- recall: 0.298719
- f1: 0.319213

## Observação
- O critério 'entropy' no DecisionTreeClassifier corresponde à lógica de information gain usada para escolher os cortes.