"""
Hierarchical clustering analysis on z-score normalized CVD dataset.

Based on the methods used in Source_Material/06-unsupervised.ipynb:
- AgglomerativeClustering (scikit-learn)
- Dendrogram plotting helper from sklearn docs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model: AgglomerativeClustering, **kwargs) -> None:
    """Create linkage matrix from fitted AgglomerativeClustering and plot dendrogram."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)


def suggest_distance_cut(distances: np.ndarray) -> tuple[float, int, float]:
    """
    Suggest dendrogram cut using the largest jump in merge distances.

    Returns:
        threshold, merge_distance_before_jump, jump_size
    """
    if len(distances) < 2:
        return float(distances[0]) if len(distances) == 1 else 0.0, 0, 0.0

    jumps = np.diff(distances)
    idx = int(np.argmax(jumps))
    threshold = float((distances[idx] + distances[idx + 1]) / 2.0)
    return threshold, int(idx), float(jumps[idx])


def evaluate_silhouette_range(
    X: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": score})
    return pd.DataFrame(rows)


def evaluate_partition_metrics(X: pd.DataFrame, labels: np.ndarray) -> dict[str, float | int]:
    """Evaluate common internal clustering metrics for one partition."""
    n_clusters = int(len(np.unique(labels)))
    if n_clusters < 2:
        return {
            "n_clusters": n_clusters,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    return {
        "n_clusters": n_clusters,
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def plot_pca_clusters(
    X: pd.DataFrame,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Plot 2D PCA projection colored by cluster labels."""
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(X)
    expl = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        projected[:, 0],
        projected[:, 1],
        c=labels,
        cmap="tab10",
        s=20,
        alpha=0.75,
    )
    plt.title(title)
    plt.xlabel(f"PC1 ({expl[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({expl[1] * 100:.1f}% var)")
    plt.grid(alpha=0.2)
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def characterize_clusters(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Build cluster-level summary statistics and top distinguishing features."""
    work = X.copy()
    work["cluster"] = labels

    rows = []
    global_means = X.mean()
    n_total = len(work)

    for cluster_id, group in work.groupby("cluster"):
        cluster_means = group.drop(columns=["cluster"]).mean()
        mean_diff = (cluster_means - global_means).abs().sort_values(ascending=False)
        top_features = mean_diff.head(5).index.tolist()

        row = {
            "cluster": int(cluster_id),
            "n_samples": int(len(group)),
            "percentage": float((len(group) / n_total) * 100),
            "top_features": ", ".join(top_features),
        }

        for feat in top_features:
            row[f"mean_{feat}"] = float(cluster_means[feat])
            row[f"delta_{feat}"] = float(cluster_means[feat] - global_means[feat])

        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def representative_examples(
    X: pd.DataFrame,
    labels: np.ndarray,
    n_examples: int = 3,
) -> pd.DataFrame:
    """Return representative examples per cluster (nearest to cluster centroid)."""
    arr = X.to_numpy()
    examples = []

    for cluster_id in sorted(np.unique(labels)):
        idx = np.where(labels == cluster_id)[0]
        cluster_arr = arr[idx]
        centroid = cluster_arr.mean(axis=0)
        dists = np.linalg.norm(cluster_arr - centroid, axis=1)

        chosen_local = np.argsort(dists)[:n_examples]
        chosen_idx = idx[chosen_local]

        for rank, original_idx in enumerate(chosen_idx, start=1):
            row = X.iloc[original_idx].to_dict()
            row["cluster"] = int(cluster_id)
            row["example_rank"] = rank
            row["distance_to_centroid"] = float(np.linalg.norm(arr[original_idx] - centroid))
            examples.append(row)

    cols_front = ["cluster", "example_rank", "distance_to_centroid"]
    out = pd.DataFrame(examples)
    other_cols = [c for c in out.columns if c not in cols_front]
    return out[cols_front + other_cols]


def plot_silhouette_curve(sil_df: pd.DataFrame, output_path: Path) -> None:
    """Plot silhouette score as a function of k."""
    plt.figure(figsize=(8, 5))
    plt.plot(sil_df["k"], sil_df["silhouette"], marker="o")
    best_i = sil_df["silhouette"].idxmax()
    best_k = int(sil_df.loc[best_i, "k"])
    best_s = float(sil_df.loc[best_i, "silhouette"])
    plt.scatter([best_k], [best_s], color="red", zorder=3, label=f"Best k={best_k}")
    plt.title("Silhouette score for hierarchical clustering")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical clustering on z-score normalized CVD data.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output_preparacao/CVD_numeric_zscore.csv",
        help="Path to z-score normalized dataset (relative to Project_Scripts).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2500,
        help="Sample size used to build full hierarchical tree (for performance).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_clustering_hierarquico",
        help="Output folder (relative to Project_Scripts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    input_path = base_dir / args.input
    output_dir = base_dir / args.output_dir
    ensure_dir(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric columns found in input dataset.")

    X = df[numeric_cols].dropna().copy()
    if len(X) < 2:
        raise ValueError("Not enough rows for hierarchical clustering.")

    sample_size = min(args.sample_size, len(X))
    X_sample = X.sample(n=sample_size, random_state=args.random_state)

    print("=" * 90)
    print("HIERARCHICAL CLUSTERING ANALYSIS (Z-SCORE DATA)")
    print("=" * 90)
    print(f"Input: {input_path}")
    print(f"Total rows available: {len(X)}")
    print(f"Rows used for dendrogram/tree: {len(X_sample)}")
    print(f"Features used: {', '.join(numeric_cols)}")

    hierarchical_full = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        linkage="ward",
        compute_distances=True,
    ).fit(X_sample)

    threshold, jump_idx, jump_size = suggest_distance_cut(hierarchical_full.distances_)

    cut_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="ward",
    ).fit(X_sample)

    labels_cut = cut_model.labels_
    n_clusters_cut = int(len(np.unique(labels_cut)))

    sil_df = evaluate_silhouette_range(X_sample, k_min=2, k_max=10)
    k_best = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    sil_best = float(sil_df["silhouette"].max())

    bestk_model = AgglomerativeClustering(n_clusters=k_best, linkage="ward").fit(X_sample)
    labels_bestk = bestk_model.labels_

    metrics_cut = evaluate_partition_metrics(X_sample, labels_cut)
    metrics_bestk = evaluate_partition_metrics(X_sample, labels_bestk)

    # Dendrogram (truncated: last clusters)
    plt.figure(figsize=(14, 7))
    plot_dendrogram(hierarchical_full, truncate_mode="lastp", p=20)
    plt.axhline(y=threshold, color="red", linestyle="--", linewidth=2, label=f"Suggested cut = {threshold:.3f}")
    plt.title("Dendrogram (ward, truncate_mode='lastp', p=20)")
    plt.xlabel("Cluster index / merged groups")
    plt.ylabel("Merge distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "dendrograma_lastp.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Dendrogram (truncated by level)
    plt.figure(figsize=(14, 7))
    plot_dendrogram(hierarchical_full, truncate_mode="level", p=5)
    plt.axhline(y=threshold, color="red", linestyle="--", linewidth=2, label=f"Suggested cut = {threshold:.3f}")
    plt.title("Dendrogram (ward, truncate_mode='level', p=5)")
    plt.xlabel("Cluster index / merged groups")
    plt.ylabel("Merge distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "dendrograma_level.png", dpi=150, bbox_inches="tight")
    plt.close()

    labeled_sample = X_sample.copy()
    labeled_sample["cluster_cut"] = labels_cut
    labeled_sample["cluster_bestk"] = labels_bestk
    labeled_sample.to_csv(output_dir / "amostra_clusters_corte_sugerido.csv", index=False)

    pd.DataFrame(
        {
            "merge_distance": hierarchical_full.distances_,
            "distance_jump": np.append(np.nan, np.diff(hierarchical_full.distances_)),
        }
    ).to_csv(output_dir / "distancias_fusoes_dendrograma.csv", index=False)

    sil_df.to_csv(output_dir / "silhouette_hierarquico_k2_k10.csv", index=False)
    plot_silhouette_curve(sil_df, output_dir / "silhouette_hierarquico_k2_k10.png")

    plot_pca_clusters(
        X_sample,
        labels_cut,
        output_dir / "pca_clusters_corte_sugerido.png",
        f"PCA - Clusters pelo corte do dendrograma (n={n_clusters_cut})",
    )
    plot_pca_clusters(
        X_sample,
        labels_bestk,
        output_dir / "pca_clusters_bestk_silhouette.png",
        f"PCA - Clusters para melhor k por silhouette (k={k_best})",
    )

    char_cut = characterize_clusters(X_sample, labels_cut)
    char_bestk = characterize_clusters(X_sample, labels_bestk)
    char_cut.to_csv(output_dir / "caracterizacao_clusters_corte_sugerido.csv", index=False)
    char_bestk.to_csv(output_dir / "caracterizacao_clusters_bestk_silhouette.csv", index=False)

    reps_cut = representative_examples(X_sample, labels_cut, n_examples=3)
    reps_bestk = representative_examples(X_sample, labels_bestk, n_examples=3)
    reps_cut.to_csv(output_dir / "exemplos_representativos_corte_sugerido.csv", index=False)
    reps_bestk.to_csv(output_dir / "exemplos_representativos_bestk_silhouette.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "partition": "corte_sugerido_dendrograma",
                **metrics_cut,
            },
            {
                "partition": "best_k_silhouette",
                **metrics_bestk,
            },
        ]
    )
    metrics_df.to_csv(output_dir / "metricas_internas_clusters.csv", index=False)

    summary_lines = [
        "ANALISE DE CLUSTERING HIERARQUICO (WARD) - DADOS Z-SCORE",
        "",
        f"Ficheiro de entrada: {input_path}",
        f"N linhas totais (sem NA): {len(X)}",
        f"N linhas usadas na analise hierarquica: {len(X_sample)}",
        f"N atributos numericos: {len(numeric_cols)}",
        "",
        "RECOMENDACAO DE CORTE DO DENDROGRAMA",
        f"Distancia sugerida para corte: {threshold:.6f}",
        f"Maior salto entre fusoes: {jump_size:.6f} (indice de fusao {jump_idx})",
        f"N clusters resultante do corte sugerido: {n_clusters_cut}",
        "",
        "VALIDACAO AUXILIAR (SILHOUETTE PARA k=2..10)",
        f"Melhor k por silhouette: {k_best}",
        f"Melhor silhouette: {sil_best:.6f}",
        "",
        "METRICAS INTERNAS (PARTICOES ANALISADAS)",
        (
            "Corte sugerido -> "
            f"silhouette={metrics_cut['silhouette']:.6f}, "
            f"calinski_harabasz={metrics_cut['calinski_harabasz']:.3f}, "
            f"davies_bouldin={metrics_cut['davies_bouldin']:.6f}"
        ),
        (
            "Best k silhouette -> "
            f"silhouette={metrics_bestk['silhouette']:.6f}, "
            f"calinski_harabasz={metrics_bestk['calinski_harabasz']:.3f}, "
            f"davies_bouldin={metrics_bestk['davies_bouldin']:.6f}"
        ),
        "",
        "FICHEIROS GERADOS",
        f"- {output_dir / 'dendrograma_lastp.png'}",
        f"- {output_dir / 'dendrograma_level.png'}",
        f"- {output_dir / 'distancias_fusoes_dendrograma.csv'}",
        f"- {output_dir / 'silhouette_hierarquico_k2_k10.csv'}",
        f"- {output_dir / 'silhouette_hierarquico_k2_k10.png'}",
        f"- {output_dir / 'pca_clusters_corte_sugerido.png'}",
        f"- {output_dir / 'pca_clusters_bestk_silhouette.png'}",
        f"- {output_dir / 'metricas_internas_clusters.csv'}",
        f"- {output_dir / 'caracterizacao_clusters_corte_sugerido.csv'}",
        f"- {output_dir / 'caracterizacao_clusters_bestk_silhouette.csv'}",
        f"- {output_dir / 'exemplos_representativos_corte_sugerido.csv'}",
        f"- {output_dir / 'exemplos_representativos_bestk_silhouette.csv'}",
        f"- {output_dir / 'amostra_clusters_corte_sugerido.csv'}",
    ]

    (output_dir / "resumo_clustering_hierarquico.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nSuggested dendrogram cut (distance):", f"{threshold:.6f}")
    print("Clusters at suggested cut:", n_clusters_cut)
    print("Best k by silhouette (2..10):", k_best, f"(score={sil_best:.4f})")
    print(
        "Metrics (cut):",
        f"sil={metrics_cut['silhouette']:.4f}",
        f"ch={metrics_cut['calinski_harabasz']:.2f}",
        f"db={metrics_cut['davies_bouldin']:.4f}",
    )
    print(
        "Metrics (best-k):",
        f"sil={metrics_bestk['silhouette']:.4f}",
        f"ch={metrics_bestk['calinski_harabasz']:.2f}",
        f"db={metrics_bestk['davies_bouldin']:.4f}",
    )
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
