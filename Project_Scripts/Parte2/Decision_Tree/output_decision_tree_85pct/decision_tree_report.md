# Relatório do Modelo de Árvores de Decisão

- Coluna alvo: Heart_Disease
- Entropia do conjunto de treino: 0.405165

## Melhores atributos por information gain
- Age_Category: IG=0.028836, threshold=1.500000
- General_Health: IG=0.026411, threshold=1.500000
- Diabetes: IG=0.016460, threshold=0.500000
- Arthritis: IG=0.015860, threshold=0.500000
- Smoking_History: IG=0.008374, threshold=0.500000
- Checkup: IG=0.007299, threshold=3.500000
- Exercise: IG=0.006126, threshold=0.500000
- Other_Cancer: IG=0.004960, threshold=0.500000
- Alcohol_Consumption: IG=0.004960, threshold=0.500000
- Skin_Cancer: IG=0.004716, threshold=0.500000

## Hiperparâmetros selecionados
- max_depth: None
- min_samples_split: 20
- min_samples_leaf: 20

## Melhor validação interna
- validation_f1: 0.148287
- validation_recall: 0.093286
- validation_accuracy: 0.913361

## Métricas no treino
- accuracy: 0.923531
- precision: 0.605699
- recall: 0.155241
- f1: 0.247140

## Métricas no teste
- accuracy: 0.913834
- precision: 0.371069
- recall: 0.094501
- f1: 0.150638

## Observação
- O critério 'entropy' no DecisionTreeClassifier corresponde à lógica de information gain usada para escolher os cortes.