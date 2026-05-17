"""Gerar gráfico de evaluation_metrics a partir de decision_tree_metrics_summary.csv

Este script gera um gráfico parecido com os outros modelos (barras das métricas)
quando só existe o CSV resumo com 'train' e 'test'.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "output_decision_tree_85pct" / "decision_tree_metrics_summary.csv"
OUTPUT_PNG = BASE_DIR / "output_decision_tree_85pct" / "decision_tree_evaluation_metrics.png"


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Ficheiro não encontrado: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # Espera colunas: split, accuracy, precision, recall, f1
    metric_cols = [c for c in ['accuracy', 'precision', 'recall', 'f1'] if c in df.columns]
    if df.empty or not metric_cols:
        raise SystemExit("CSV vazio ou sem colunas de métricas reconhecíveis.")

    # Transformar para formato longo para plot com seaborn
    df_long = df.melt(id_vars=['split'], value_vars=metric_cols,
                      var_name='metric', value_name='value')

    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))

    # Plot horizontal grouped bars: metrics no eixo y, valor no x, hue=split
    ax = sns.barplot(data=df_long, x='value', y='metric', hue='split', orient='h', ci=None)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Score')
    ax.set_ylabel('Métrica')
    ax.set_title('Decision Tree - Evaluation Metrics')

    # Anotações com valores
    for p in ax.patches:
        width = p.get_width()
        if pd.notna(width):
            ax.text(width + 0.01, p.get_y() + p.get_height() / 2,
                    f"{width:.3f}", va='center')

    plt.legend(title='Split')
    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico salvo: {OUTPUT_PNG}")


if __name__ == '__main__':
    main()
