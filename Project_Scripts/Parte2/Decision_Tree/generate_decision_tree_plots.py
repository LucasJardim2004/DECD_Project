"""Gerar evaluation metrics para todos os outputs do Decision_Tree.

Este script procura por pastas cujo nome começa por 'output_decision_tree' dentro
da pasta `Decision_Tree` e gera um PNG `*_evaluation_metrics.png` para cada
`*_metrics_summary.csv` encontrado, usando o mesmo formato dos outros modelos.
"""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent


def plot_from_csv(csv_path: Path, out_png: Path) -> None:
    df = pd.read_csv(csv_path)
    metric_cols = [c for c in ['accuracy', 'precision', 'recall', 'f1'] if c in df.columns]
    if df.empty or not metric_cols:
        print(f"Ignorado (sem métricas): {csv_path}")
        return

    df_long = df.melt(id_vars=['split'], value_vars=metric_cols,
                      var_name='metric', value_name='value')

    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_long, x='value', y='metric', hue='split', orient='h', ci=None)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Score')
    ax.set_ylabel('Métrica')
    ax.set_title(out_png.stem.replace('_', ' ').title())
    for p in ax.patches:
        width = p.get_width()
        if pd.notna(width):
            ax.text(width + 0.01, p.get_y() + p.get_height() / 2, f"{width:.3f}", va='center')
    plt.legend(title='Split')
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico salvo: {out_png}")


def main():
    # Procurar por diretórios output_decision_tree*
    for d in sorted(BASE_DIR.iterdir()):
        if d.is_dir() and d.name.startswith('output_decision_tree'):
            csv = d / 'decision_tree_metrics_summary.csv'
            if csv.exists():
                out_png = d / f"{d.name}_evaluation_metrics.png"
                plot_from_csv(csv, out_png)
            else:
                print(f"Sem ficheiro metrics_summary em: {d}")


if __name__ == '__main__':
    main()
