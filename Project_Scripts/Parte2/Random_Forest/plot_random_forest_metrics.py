"""Gerar gráfico de evaluation metrics para Random_Forest a partir do CSV resumo."""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / 'output_random_forest' / 'random_forest_metrics_summary.csv'
OUT_PNG = BASE_DIR / 'output_random_forest' / 'random_forest_evaluation_metrics.png'


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Ficheiro não encontrado: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    metric_cols = [c for c in ['accuracy', 'precision', 'recall', 'f1'] if c in df.columns]
    if df.empty or not metric_cols:
        raise SystemExit("CSV vazio ou sem colunas de métricas reconhecíveis.")

    df_long = df.melt(id_vars=['split'], value_vars=metric_cols,
                      var_name='metric', value_name='value')

    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_long, x='value', y='metric', hue='split', orient='h', ci=None)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Score')
    ax.set_ylabel('Métrica')
    ax.set_title('Random Forest - Evaluation Metrics')
    for p in ax.patches:
        width = p.get_width()
        if pd.notna(width):
            ax.text(width + 0.01, p.get_y() + p.get_height() / 2, f"{width:.3f}", va='center')
    plt.legend(title='Split')
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico salvo: {OUT_PNG}")


if __name__ == '__main__':
    main()
