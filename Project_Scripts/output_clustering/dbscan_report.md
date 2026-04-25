# Relatorio DBSCAN - Analise Completa

## Metodologia
- Dataset de entrada: `/Users/lucasjardim/Desktop/Pastas progamação/DECD_Proj/Project_Scripts/output_preparacao/CVD_numeric_zscore.csv`
- Observacoes totais: 308854
- Observacoes usadas na grelha DBSCAN: 50000
- Variaveis usadas: 18
- Algoritmo: DBSCAN (distance-based density clustering)
- Distancia padrao: euclidiana
- Avaliacao principal: silhouette_non_noise, ruido (%), numero de clusters

## Melhor Configuracao Encontrada
- eps: 1.200
- min_samples: 15
- clusters (sem ruido): 7
- pontos de ruido: 29314 (58.63%)
- silhouette_non_noise: 0.0512

## Top 5 Configuracoes por Silhouette (sem ruido)

| eps | min_samples | n_clusters | noise_pct | silhouette_non_noise | silhouette_all |
| --- | --- | --- | --- | --- | --- |
| 1.2 | 15 | 7 | 58.628 | 0.05122745078692552 | -0.17061931822264292 |
| 1.5 | 10 | 3 | 22.024 | 0.036410466405379815 | -0.08929892352716168 |
| 1.2 | 10 | 10 | 52.266 | 0.011704968544219787 | -0.15794012688376396 |
| 1.5 | 5 | 27 | 17.291999999999998 | -0.06889503132472999 | -0.1563622092324632 |
| 1.2 | 5 | 70 | 43.065999999999995 | -0.21784424327589444 | -0.35717617696152737 |

## Leituras e Conclusoes
- DBSCAN e sensivel aos parametros `eps` e `min_samples`; por isso a grelha foi necessaria.
- Configuracoes com muito ruido (>60%) tendem a reduzir interpretabilidade dos clusters.
- O k-distance plot ajuda a definir uma faixa inicial de `eps` para tentativas futuras.
- A interpretacao final deve combinar qualidade numerica (silhouette) e utilidade de negocio.

## Artefactos Gerados
- `dbscan_grid_metrics.csv`
- `dbscan_k_distance_plot.png`
- `dbscan_silhouette_heatmap.png`
- `dbscan_noise_heatmap.png`
- `dbscan_best_cluster_summary.csv`
- `dbscan_best_pca_scatter.png`