# Relatório do Modelo Random Forest

- Coluna alvo: Heart_Disease
- Limiar de decisão selecionado: 0.38

## Hiperparâmetros selecionados
- n_estimators: 200
- max_depth: 8
- min_samples_split: 20
- min_samples_leaf: 10
- class_weight: balanced_subsample

## Melhor validação interna
- validation_f2: 0.665446
- validation_recall: 0.795995
- validation_f1: 0.534060
- validation_accuracy: 0.722214

## Métricas no treino
- accuracy: 0.661164
- precision: 0.359842
- recall: 0.891119
- f1: 0.512665
- f2: 0.687972

## Métricas no teste
- accuracy: 0.622051
- precision: 0.163735
- recall: 0.894554
- f1: 0.276805
- f2: 0.472638
- pr_auc: 0.297305

## Observação
- Random Forest combina várias árvores e usa balanced_subsample para ajustar o peso das classes em cada amostra bootstrap.