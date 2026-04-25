"""
DBSCAN full analysis for the CVD project.

Default input:
- output_preparacao/CVD_numeric_zscore.csv

Outputs:
- output_clustering/dbscan_grid_metrics.csv
- output_clustering/dbscan_k_distance_plot.png
- output_clustering/dbscan_silhouette_heatmap.png
- output_clustering/dbscan_noise_heatmap.png
- output_clustering/dbscan_best_cluster_summary.csv
- output_clustering/dbscan_best_pca_scatter.png
- output_clustering/dbscan_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DBSCAN complete analysis")
    parser.add_argument(
        "--csv",
        default="output_preparacao/CVD_numeric_zscore.csv",
        help="Input CSV path (relative to script folder or absolute).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_clustering",
        help="Output directory for DBSCAN artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Max rows used for DBSCAN grid analysis.",
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=10000,
        help="Max rows for silhouette computation inside each DBSCAN run.",
    )
    parser.add_argument(
        "--eps-values",
        default="0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,1.20,1.50",
        help="Comma-separated eps values for DBSCAN.",
    )
    parser.add_argument(
        "--min-samples-values",
        default="5,10,15,20,30,40",
        help="Comma-separated min_samples values for DBSCAN.",
    )
    parser.add_argument(
        "--k-distance-k",
        type=int,
        default=5,
        help="k used in k-distance plot (NearestNeighbors).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_numeric_data(csv_path: Path, sample_size: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include="number").dropna().reset_index(drop=True)

    if numeric_df.empty:
        raise ValueError("Input CSV has no numeric rows after dropna().")

    if sample_size > 0 and len(numeric_df) > sample_size:
        sample_df = numeric_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        sample_df = numeric_df

    return numeric_df, sample_df


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")

    counts = np.bincount(pd.Categorical(labels).codes)
    if np.any(counts < 2):
        return float("nan")

    return float(silhouette_score(X, labels))


def evaluate_dbscan_grid(
    sample_df: pd.DataFrame,
    eps_values: list[float],
    min_samples_values: list[int],
    silhouette_sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    if silhouette_sample_size > 0 and len(sample_df) > silhouette_sample_size:
        silhouette_df = sample_df.sample(n=silhouette_sample_size, random_state=random_state).reset_index(drop=True)
    else:
        silhouette_df = sample_df

    X = sample_df.to_numpy()
    X_sil = silhouette_df.to_numpy()

    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)

            labels_series = pd.Series(labels)
            n_noise = int((labels_series == -1).sum())
            n_clusters = int(labels_series[labels_series != -1].nunique())
            noise_pct = float(n_noise / len(labels_series) * 100)
            core_points = int(len(model.core_sample_indices_))

            labels_sil = model.fit_predict(X_sil)
            sil_all = safe_silhouette(X_sil, labels_sil)

            non_noise_mask = labels_sil != -1
            if non_noise_mask.sum() > 1:
                sil_non_noise = safe_silhouette(X_sil[non_noise_mask], labels_sil[non_noise_mask])
            else:
                sil_non_noise = float("nan")

            cluster_sizes = labels_series[labels_series != -1].value_counts(normalize=True)
            largest_cluster_pct = float(cluster_sizes.iloc[0] * 100) if not cluster_sizes.empty else float("nan")

            rows.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_pct": noise_pct,
                    "core_points": core_points,
                    "largest_cluster_pct": largest_cluster_pct,
                    "silhouette_all": sil_all,
                    "silhouette_non_noise": sil_non_noise,
                }
            )

    return pd.DataFrame(rows)


def choose_best_configuration(metrics_df: pd.DataFrame) -> pd.Series:
    candidates = metrics_df.copy()

    # Favor practical clusterings first: >=2 clusters, non-trivial non-noise silhouette, and controlled noise.
    candidates = candidates[(candidates["n_clusters"] >= 2) & (candidates["noise_pct"] <= 60)]
    candidates = candidates.dropna(subset=["silhouette_non_noise"])

    if candidates.empty:
        fallback = metrics_df.copy()
        fallback = fallback.sort_values(
            by=["n_clusters", "noise_pct", "core_points"],
            ascending=[False, True, False],
        )
        return fallback.iloc[0]

    ranked = candidates.sort_values(
        by=["silhouette_non_noise", "noise_pct", "n_clusters", "core_points"],
        ascending=[False, True, False, False],
    )
    return ranked.iloc[0]


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def plot_k_distance(sample_df: pd.DataFrame, k: int, output_file: Path) -> None:
    X = sample_df.to_numpy()
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    k_dist = np.sort(distances[:, -1])

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(k_dist)), k_dist, color="tab:blue", linewidth=1)
    plt.title(f"DBSCAN k-distance plot (k={k})")
    plt.xlabel("Sorted points")
    plt.ylabel(f"Distance to {k}th nearest neighbor")
    plt.grid(alpha=0.3)

    q90 = float(np.quantile(k_dist, 0.90))
    q95 = float(np.quantile(k_dist, 0.95))
    plt.axhline(q90, linestyle="--", color="orange", label=f"q90={q90:.3f}")
    plt.axhline(q95, linestyle="--", color="red", label=f"q95={q95:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def plot_heatmap(metrics_df: pd.DataFrame, value_col: str, title: str, output_file: Path) -> None:
    pivot = metrics_df.pivot_table(index="min_samples", columns="eps", values=value_col, aggfunc="mean")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".3f")
    plt.title(title)
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def build_best_cluster_artifacts(
    sample_df: pd.DataFrame,
    eps: float,
    min_samples: int,
    random_state: int,
    summary_path: Path,
    plot_path: Path,
) -> tuple[int, int, float]:
    X = sample_df.to_numpy()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    counts = pd.Series(labels, name="cluster").value_counts().sort_index().rename_axis("cluster").reset_index(name="count")
    counts["pct"] = counts["count"] / len(sample_df) * 100
    counts.to_csv(summary_path, index=False)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(sample_df)

    unique_labels = sorted(pd.Series(labels).unique().tolist())
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_id[lbl] for lbl in labels])

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=numeric_labels,
        cmap="tab20",
        s=12,
        alpha=0.75,
    )
    plt.title(f"DBSCAN best config (eps={eps}, min_samples={min_samples}) - PCA 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)

    legend_labels = [f"noise" if lbl == -1 else f"cluster {lbl}" for lbl in unique_labels]
    handles, _ = scatter.legend_elements(num=len(unique_labels))
    plt.legend(handles, legend_labels, title="Label", loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    n_clusters = int(pd.Series(labels)[pd.Series(labels) != -1].nunique())
    n_noise = int((pd.Series(labels) == -1).sum())
    noise_pct = float(n_noise / len(sample_df) * 100)

    return n_clusters, n_noise, noise_pct


def build_report(
    csv_path: Path,
    full_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    best_row: pd.Series,
    best_n_clusters: int,
    best_n_noise: int,
    best_noise_pct: float,
) -> str:
    best_eps = float(best_row["eps"])
    best_min_samples = int(best_row["min_samples"])
    best_sil_non_noise = best_row.get("silhouette_non_noise", float("nan"))

    practical = metrics_df[
        (metrics_df["n_clusters"] >= 2)
        & (metrics_df["noise_pct"] <= 60)
    ].dropna(subset=["silhouette_non_noise"])

    if practical.empty:
        top_by_sil = (
            metrics_df.dropna(subset=["silhouette_non_noise"])
            .sort_values(by=["silhouette_non_noise", "noise_pct"], ascending=[False, True])
            .head(5)
            .copy()
        )
    else:
        top_by_sil = (
            practical.sort_values(by=["silhouette_non_noise", "noise_pct"], ascending=[False, True])
            .head(5)
            .copy()
        )

    lines = [
        "# Relatorio DBSCAN - Analise Completa",
        "",
        "## Metodologia",
        f"- Dataset de entrada: `{csv_path}`",
        f"- Observacoes totais: {len(full_df)}",
        f"- Observacoes usadas na grelha DBSCAN: {len(sample_df)}",
        f"- Variaveis usadas: {len(full_df.columns)}",
        "- Algoritmo: DBSCAN (distance-based density clustering)",
        "- Distancia padrao: euclidiana",
        "- Avaliacao principal: silhouette_non_noise, ruido (%), numero de clusters",
        "",
        "## Melhor Configuracao Encontrada",
        f"- eps: {best_eps:.3f}",
        f"- min_samples: {best_min_samples}",
        f"- clusters (sem ruido): {best_n_clusters}",
        f"- pontos de ruido: {best_n_noise} ({best_noise_pct:.2f}%)",
        f"- silhouette_non_noise: {best_sil_non_noise:.4f}" if pd.notna(best_sil_non_noise) else "- silhouette_non_noise: N/A",
        "",
        "## Top 5 Configuracoes por Silhouette (sem ruido)",
        "",
    ]

    if top_by_sil.empty:
        lines.append("Nenhuma configuracao com silhouette valido foi encontrada.")
    else:
        table = top_by_sil[
            [
                "eps",
                "min_samples",
                "n_clusters",
                "noise_pct",
                "silhouette_non_noise",
                "silhouette_all",
            ]
        ].copy()
        lines.append(dataframe_to_markdown_table(table))

    lines.extend(
        [
            "",
            "## Leituras e Conclusoes",
            "- DBSCAN e sensivel aos parametros `eps` e `min_samples`; por isso a grelha foi necessaria.",
            "- Configuracoes com muito ruido (>60%) tendem a reduzir interpretabilidade dos clusters.",
            "- O k-distance plot ajuda a definir uma faixa inicial de `eps` para tentativas futuras.",
            "- A interpretacao final deve combinar qualidade numerica (silhouette) e utilidade de negocio.",
            "",
            "## Artefactos Gerados",
            "- `dbscan_grid_metrics.csv`",
            "- `dbscan_k_distance_plot.png`",
            "- `dbscan_silhouette_heatmap.png`",
            "- `dbscan_noise_heatmap.png`",
            "- `dbscan_best_cluster_summary.csv`",
            "- `dbscan_best_pca_scatter.png`",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    output_dir = base_dir / args.output_dir
    ensure_dir(output_dir)

    eps_values = parse_float_list(args.eps_values)
    min_samples_values = parse_int_list(args.min_samples_values)

    if not eps_values:
        raise ValueError("eps-values cannot be empty")
    if not min_samples_values:
        raise ValueError("min-samples-values cannot be empty")

    full_df, sample_df = load_numeric_data(csv_path, args.sample_size, args.random_state)

    metrics_df = evaluate_dbscan_grid(
        sample_df=sample_df,
        eps_values=eps_values,
        min_samples_values=min_samples_values,
        silhouette_sample_size=args.silhouette_sample_size,
        random_state=args.random_state,
    )

    best_row = choose_best_configuration(metrics_df)

    best_eps = float(best_row["eps"])
    best_min_samples = int(best_row["min_samples"])

    metrics_path = output_dir / "dbscan_grid_metrics.csv"
    kdist_plot_path = output_dir / "dbscan_k_distance_plot.png"
    sil_heatmap_path = output_dir / "dbscan_silhouette_heatmap.png"
    noise_heatmap_path = output_dir / "dbscan_noise_heatmap.png"
    best_summary_path = output_dir / "dbscan_best_cluster_summary.csv"
    best_plot_path = output_dir / "dbscan_best_pca_scatter.png"
    report_path = output_dir / "dbscan_report.md"

    metrics_df.to_csv(metrics_path, index=False)

    plot_k_distance(sample_df=sample_df, k=args.k_distance_k, output_file=kdist_plot_path)
    plot_heatmap(
        metrics_df=metrics_df,
        value_col="silhouette_non_noise",
        title="DBSCAN silhouette (sem ruido)",
        output_file=sil_heatmap_path,
    )
    plot_heatmap(
        metrics_df=metrics_df,
        value_col="noise_pct",
        title="DBSCAN ruido (%)",
        output_file=noise_heatmap_path,
    )

    best_n_clusters, best_n_noise, best_noise_pct = build_best_cluster_artifacts(
        sample_df=sample_df,
        eps=best_eps,
        min_samples=best_min_samples,
        random_state=args.random_state,
        summary_path=best_summary_path,
        plot_path=best_plot_path,
    )

    report_text = build_report(
        csv_path=csv_path,
        full_df=full_df,
        sample_df=sample_df,
        metrics_df=metrics_df,
        best_row=best_row,
        best_n_clusters=best_n_clusters,
        best_n_noise=best_n_noise,
        best_noise_pct=best_noise_pct,
    )
    report_path.write_text(report_text, encoding="utf-8")

    print("DBSCAN analysis finished.")
    print(f"Input CSV: {csv_path}")
    print(f"Rows total: {len(full_df)}")
    print(f"Rows used: {len(sample_df)}")
    print(f"Columns used: {', '.join(full_df.columns)}")
    print(f"Best eps: {best_eps:.3f}")
    print(f"Best min_samples: {best_min_samples}")
    print(f"Best clusters (without noise): {best_n_clusters}")
    print(f"Noise pct (best): {best_noise_pct:.2f}%")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved k-distance plot: {kdist_plot_path}")
    print(f"Saved silhouette heatmap: {sil_heatmap_path}")
    print(f"Saved noise heatmap: {noise_heatmap_path}")
    print(f"Saved best cluster summary: {best_summary_path}")
    print(f"Saved best scatter plot: {best_plot_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
