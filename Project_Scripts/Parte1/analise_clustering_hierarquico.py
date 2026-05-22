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
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import f_oneway
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)


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


def plot_silhouette_samples(X: pd.DataFrame, labels: np.ndarray, output_path: Path, title: str) -> None:
    """Plot silhouette values for each sample, grouped by cluster."""
    sil_samples = silhouette_samples(X, labels)
    n_clusters = len(np.unique(labels))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_lower = 10
    
    for cluster_id in range(n_clusters):
        cluster_sil_samples = sil_samples[labels == cluster_id]
        cluster_sil_samples.sort()
        
        size_cluster = cluster_sil_samples.shape[0]
        y_upper = y_lower + size_cluster
        
        color = plt.cm.tab20(cluster_id % 20)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_sil_samples,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id))
        y_lower = y_upper + 10
    
    ax.set_title(title)
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=sil_samples.mean(), color="red", linestyle="--", label=f"Mean = {sil_samples.mean():.3f}")
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_sizes(labels: np.ndarray, output_path: Path, title: str) -> None:
    """Plot distribution of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(unique.astype(str), counts, color=plt.cm.tab20(np.arange(len(unique)) % 20), edgecolor="black", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Samples")
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_boxplots(X: pd.DataFrame, labels: np.ndarray, output_path: Path, title: str, top_n: int = 8) -> None:
    """Plot boxplots for top distinguishing features by cluster."""
    work = X.copy()
    work["cluster"] = labels.astype(str)
    
    # Find top features with highest variance across clusters
    global_std = X.std()
    cluster_means = work.groupby("cluster").mean()
    feature_variance = cluster_means.var(axis=0)
    top_features = feature_variance.nlargest(top_n).index.tolist()
    
    n_features = len(top_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        sns.boxplot(data=work, x="cluster", y=feature, ax=ax, palette="Set2", legend=False)
        ax.set_title(f"{feature} by Cluster", fontsize=10, fontweight="bold")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(feature)
        ax.grid(axis="y", alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(top_features), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_heatmap(X: pd.DataFrame, labels: np.ndarray, output_path: Path, title: str, top_n: int = 15) -> None:
    """Plot heatmap of mean values per feature and cluster."""
    work = X.copy()
    work["cluster"] = labels.astype(int)
    
    cluster_means = work.groupby("cluster").mean()
    
    # Select top features by variance
    feature_variance = cluster_means.var(axis=0)
    top_features = feature_variance.nlargest(top_n).index.tolist()
    
    heatmap_data = cluster_means[top_features]
    
    # Normalize for better visualization
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_normalized.T,
        annot=heatmap_data.T,
        fmt=".2f",
        cmap="RdYlGn_r",
        cbar_kws={"label": "Normalized Value"},
        linewidths=0.5,
    )
    plt.title(title)
    plt.xlabel("Cluster ID")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def perform_anova_analysis(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Perform ANOVA test for each feature across clusters."""
    rows = []
    
    for feature in X.columns:
        groups = [X[labels == cluster_id][feature].values for cluster_id in np.unique(labels)]
        if len(groups) > 1:
            f_stat, p_value = f_oneway(*groups)
            rows.append({
                "feature": feature,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": "YES" if p_value < 0.05 else "NO",
            })
    
    return pd.DataFrame(rows).sort_values("f_statistic", ascending=False)


def compute_intracluster_statistics(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute detailed statistics for each cluster."""
    rows = []
    
    for cluster_id in sorted(np.unique(labels)):
        cluster_data = X[labels == cluster_id]
        
        row = {
            "cluster": int(cluster_id),
            "n_samples": len(cluster_data),
            "percentage": float((len(cluster_data) / len(X)) * 100),
        }
        
        # Compute various statistics
        for feature in X.columns:
            row[f"{feature}_mean"] = float(cluster_data[feature].mean())
            row[f"{feature}_std"] = float(cluster_data[feature].std())
            row[f"{feature}_min"] = float(cluster_data[feature].min())
            row[f"{feature}_max"] = float(cluster_data[feature].max())
        
        rows.append(row)
    
    return pd.DataFrame(rows)



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
        default=15000,
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
    print("Performing hierarchical clustering...")

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
    print("Generating dendrograms...")
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

    # PCA visualization
    print("Generating PCA visualizations...")
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

    # Characterization
    print("Computing cluster characterization...")
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

    # NEW: Silhouette samples visualization
    print("Generating silhouette analysis...")
    plot_silhouette_samples(
        X_sample,
        labels_cut,
        output_dir / "silhueta_amostras_corte_sugerido.png",
        f"Silhouette Analysis - Corte Sugerido (n={n_clusters_cut})",
    )
    plot_silhouette_samples(
        X_sample,
        labels_bestk,
        output_dir / "silhueta_amostras_best_k.png",
        f"Silhouette Analysis - Best k (k={k_best})",
    )

    # NEW: Cluster size distribution
    print("Generating cluster size analysis...")
    plot_cluster_sizes(
        labels_cut,
        output_dir / "distribuicao_tamanho_clusters_corte.png",
        f"Cluster Size Distribution - Corte Sugerido (n={n_clusters_cut})",
    )
    plot_cluster_sizes(
        labels_bestk,
        output_dir / "distribuicao_tamanho_clusters_best_k.png",
        f"Cluster Size Distribution - Best k (k={k_best})",
    )

    # NEW: Feature boxplots and heatmaps
    print("Generating feature analysis plots...")
    plot_feature_boxplots(
        X_sample,
        labels_cut,
        output_dir / "boxplots_features_corte_sugerido.png",
        f"Feature Distribution by Cluster - Corte Sugerido",
    )
    plot_feature_boxplots(
        X_sample,
        labels_bestk,
        output_dir / "boxplots_features_best_k.png",
        f"Feature Distribution by Cluster - Best k",
    )

    plot_feature_heatmap(
        X_sample,
        labels_cut,
        output_dir / "heatmap_features_corte_sugerido.png",
        f"Feature Heatmap - Corte Sugerido",
    )
    plot_feature_heatmap(
        X_sample,
        labels_bestk,
        output_dir / "heatmap_features_best_k.png",
        f"Feature Heatmap - Best k",
    )

    # NEW: ANOVA analysis
    print("Performing ANOVA analysis...")
    anova_cut = perform_anova_analysis(X_sample, labels_cut)
    anova_bestk = perform_anova_analysis(X_sample, labels_bestk)
    anova_cut.to_csv(output_dir / "anova_analise_corte_sugerido.csv", index=False)
    anova_bestk.to_csv(output_dir / "anova_analise_best_k.csv", index=False)

    # NEW: Intracluster statistics
    print("Computing detailed cluster statistics...")
    stats_cut = compute_intracluster_statistics(X_sample, labels_cut)
    stats_bestk = compute_intracluster_statistics(X_sample, labels_bestk)
    stats_cut.to_csv(output_dir / "estatisticas_detalhadas_corte_sugerido.csv", index=False)
    stats_bestk.to_csv(output_dir / "estatisticas_detalhadas_best_k.csv", index=False)

    # Generate comprehensive report
    print("Generating comprehensive report...")
    summary_lines = generate_comprehensive_report(
        input_path, len(X), len(X_sample), numeric_cols, numeric_cols,
        threshold, jump_idx, jump_size, n_clusters_cut, k_best, sil_best,
        metrics_cut, metrics_bestk, anova_cut, anova_bestk,
        labels_cut, labels_bestk, output_dir
    )

    (output_dir / "resumo_clustering_hierarquico.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SUMMARY OF RESULTS")
    print("=" * 90)
    print(f"Suggested dendrogram cut (distance): {threshold:.6f}")
    print(f"Clusters at suggested cut: {n_clusters_cut}")
    print(f"Best k by silhouette (2..10): {k_best} (score={sil_best:.4f})")
    print(f"\nMetrics (cut):")
    print(f"  Silhouette: {metrics_cut['silhouette']:.4f}")
    print(f"  Calinski-Harabasz: {metrics_cut['calinski_harabasz']:.2f}")
    print(f"  Davies-Bouldin: {metrics_cut['davies_bouldin']:.4f}")
    print(f"\nMetrics (best-k):")
    print(f"  Silhouette: {metrics_bestk['silhouette']:.4f}")
    print(f"  Calinski-Harabasz: {metrics_bestk['calinski_harabasz']:.2f}")
    print(f"  Davies-Bouldin: {metrics_bestk['davies_bouldin']:.4f}")
    print(f"\nOutput folder: {output_dir}")
    print("=" * 90)


def generate_comprehensive_report(
    input_path, total_rows, sample_rows, numeric_cols, features,
    threshold, jump_idx, jump_size, n_clusters_cut, k_best, sil_best,
    metrics_cut, metrics_bestk, anova_cut, anova_bestk,
    labels_cut, labels_bestk, output_dir
) -> list[str]:
    """Generate a comprehensive report with detailed analysis."""
    lines = []
    
    lines.append("=" * 100)
    lines.append("COMPREHENSIVE HIERARCHICAL CLUSTERING ANALYSIS REPORT (WARD METHOD)")
    lines.append("=" * 100)
    lines.append("")
    
    # 1. DATASET INFORMATION
    lines.append("1. DATASET INFORMATION")
    lines.append("-" * 100)
    lines.append(f"   Input file: {input_path}")
    lines.append(f"   Total rows (no missing): {total_rows:,}")
    lines.append(f"   Sample size used: {sample_rows:,} ({100*sample_rows/total_rows:.1f}%)")
    lines.append(f"   Number of features: {len(numeric_cols)}")
    lines.append(f"   Features analyzed: {', '.join(numeric_cols[:10])}")
    if len(numeric_cols) > 10:
        lines.append(f"                      {', '.join(numeric_cols[10:])}")
    lines.append("")
    
    # 2. DENDROGRAM CUT RECOMMENDATION
    lines.append("2. DENDROGRAM CUT RECOMMENDATION")
    lines.append("-" * 100)
    lines.append(f"   Recommended distance threshold: {threshold:.6f}")
    lines.append(f"   Largest jump in distances: {jump_size:.6f} (at merge index {jump_idx})")
    lines.append(f"   Interpretation: The largest gap in merge distances suggests a natural partition")
    lines.append(f"   Number of clusters from recommended cut: {n_clusters_cut}")
    lines.append("")
    
    # 3. SILHOUETTE ANALYSIS
    lines.append("3. SILHOUETTE ANALYSIS (k=2..10)")
    lines.append("-" * 100)
    lines.append(f"   Best k value found: {k_best}")
    lines.append(f"   Best silhouette score: {sil_best:.6f}")
    lines.append(f"   Interpretation: Silhouette ranges from -1 to +1")
    lines.append(f"     - Values close to 1: Well-separated clusters")
    lines.append(f"     - Values close to 0: Overlapping clusters")
    lines.append(f"     - Negative values: Potential mislabeling")
    
    score_interpretation = ""
    if sil_best > 0.5:
        score_interpretation = "EXCELLENT clustering structure"
    elif sil_best > 0.3:
        score_interpretation = "GOOD clustering structure"
    elif sil_best > 0.1:
        score_interpretation = "WEAK clustering structure (overlap between clusters)"
    else:
        score_interpretation = "VERY WEAK clustering structure (significant overlap)"
    
    lines.append(f"   Current score ({sil_best:.4f}): {score_interpretation}")
    lines.append("")
    
    # 4. METRICS COMPARISON
    lines.append("4. CLUSTERING METRICS COMPARISON")
    lines.append("-" * 100)
    lines.append(f"   Partition 1: Dendrogram Cut (n={n_clusters_cut} clusters)")
    lines.append(f"     - Silhouette: {metrics_cut['silhouette']:.6f}")
    lines.append(f"     - Calinski-Harabasz: {metrics_cut['calinski_harabasz']:.3f} (higher is better)")
    lines.append(f"     - Davies-Bouldin: {metrics_cut['davies_bouldin']:.6f} (lower is better)")
    lines.append(f"")
    lines.append(f"   Partition 2: Best k by Silhouette (k={k_best} clusters)")
    lines.append(f"     - Silhouette: {metrics_bestk['silhouette']:.6f}")
    lines.append(f"     - Calinski-Harabasz: {metrics_bestk['calinski_harabasz']:.3f}")
    lines.append(f"     - Davies-Bouldin: {metrics_bestk['davies_bouldin']:.6f}")
    lines.append(f"")
    lines.append(f"   Metric Definitions:")
    lines.append(f"     - Silhouette: Measures how similar a point is to its cluster vs other clusters")
    lines.append(f"     - Calinski-Harabasz: Ratio of between-cluster to within-cluster dispersion")
    lines.append(f"     - Davies-Bouldin: Average similarity between each cluster and its most similar cluster")
    lines.append("")
    
    # 5. SIGNIFICANT FEATURES (ANOVA)
    lines.append("5. SIGNIFICANT FEATURES (ANOVA ANALYSIS)")
    lines.append("-" * 100)
    lines.append(f"   For Dendrogram Cut (n={n_clusters_cut}):")
    sig_features_cut = anova_cut[anova_cut['significant'] == 'YES'].head(10)
    for idx, row in sig_features_cut.iterrows():
        lines.append(f"     - {row['feature']:30s} | F-stat: {row['f_statistic']:10.2f} | p-value: {row['p_value']:.2e}")
    lines.append(f"   Total significant features (p<0.05): {len(anova_cut[anova_cut['significant'] == 'YES'])} / {len(anova_cut)}")
    lines.append(f"")
    lines.append(f"   For Best k (k={k_best}):")
    sig_features_bestk = anova_bestk[anova_bestk['significant'] == 'YES'].head(10)
    for idx, row in sig_features_bestk.iterrows():
        lines.append(f"     - {row['feature']:30s} | F-stat: {row['f_statistic']:10.2f} | p-value: {row['p_value']:.2e}")
    lines.append(f"   Total significant features (p<0.05): {len(anova_bestk[anova_bestk['significant'] == 'YES'])} / {len(anova_bestk)}")
    lines.append("")
    
    # 6. CLUSTER DISTRIBUTION
    lines.append("6. CLUSTER SIZE DISTRIBUTION")
    lines.append("-" * 100)
    unique_cut, counts_cut = np.unique(labels_cut, return_counts=True)
    unique_bestk, counts_bestk = np.unique(labels_bestk, return_counts=True)
    
    lines.append(f"   Dendrogram Cut (n={n_clusters_cut}):")
    for cluster_id, count in zip(unique_cut, counts_cut):
        pct = 100 * count / len(labels_cut)
        lines.append(f"     - Cluster {cluster_id:2d}: {count:6d} samples ({pct:5.1f}%)")
    
    lines.append(f"")
    lines.append(f"   Best k (k={k_best}):")
    for cluster_id, count in zip(unique_bestk, counts_bestk):
        pct = 100 * count / len(labels_bestk)
        lines.append(f"     - Cluster {cluster_id:2d}: {count:6d} samples ({pct:5.1f}%)")
    lines.append("")
    
    # 7. OUTPUT FILES
    lines.append("7. GENERATED OUTPUT FILES")
    lines.append("-" * 100)
    lines.append(f"   Visualizations:")
    lines.append(f"     - dendrograma_lastp.png: Dendrogram (last p=20 merges)")
    lines.append(f"     - dendrograma_level.png: Dendrogram (level p=5)")
    lines.append(f"     - pca_clusters_corte_sugerido.png: PCA projection (suggested cut)")
    lines.append(f"     - pca_clusters_bestk_silhouette.png: PCA projection (best k)")
    lines.append(f"     - silhueta_amostras_corte_sugerido.png: Silhouette analysis (suggested cut)")
    lines.append(f"     - silhueta_amostras_best_k.png: Silhouette analysis (best k)")
    lines.append(f"     - distribuicao_tamanho_clusters_corte.png: Cluster sizes (suggested cut)")
    lines.append(f"     - distribuicao_tamanho_clusters_best_k.png: Cluster sizes (best k)")
    lines.append(f"     - boxplots_features_corte_sugerido.png: Feature boxplots (suggested cut)")
    lines.append(f"     - boxplots_features_best_k.png: Feature boxplots (best k)")
    lines.append(f"     - heatmap_features_corte_sugerido.png: Feature heatmap (suggested cut)")
    lines.append(f"     - heatmap_features_best_k.png: Feature heatmap (best k)")
    lines.append(f"     - silhouette_hierarquico_k2_k10.png: Silhouette curve (k=2..10)")
    lines.append(f"")
    lines.append(f"   Data Files:")
    lines.append(f"     - caracterizacao_clusters_corte_sugerido.csv: Top features per cluster")
    lines.append(f"     - caracterizacao_clusters_bestk_silhouette.csv: Top features per cluster")
    lines.append(f"     - exemplos_representativos_corte_sugerido.csv: Representative samples")
    lines.append(f"     - exemplos_representativos_bestk_silhouette.csv: Representative samples")
    lines.append(f"     - estatisticas_detalhadas_corte_sugerido.csv: Detailed stats (mean, std, min, max)")
    lines.append(f"     - estatisticas_detalhadas_best_k.csv: Detailed stats (mean, std, min, max)")
    lines.append(f"     - anova_analise_corte_sugerido.csv: ANOVA results for all features")
    lines.append(f"     - anova_analise_best_k.csv: ANOVA results for all features")
    lines.append(f"     - metricas_internas_clusters.csv: All clustering metrics")
    lines.append(f"     - distancias_fusoes_dendrograma.csv: Merge distances and jumps")
    lines.append(f"     - silhouette_hierarquico_k2_k10.csv: Silhouette scores for k=2..10")
    lines.append(f"     - amostra_clusters_corte_sugerido.csv: Sample data with cluster labels")
    lines.append("")
    
    # 8. RECOMMENDATIONS
    lines.append("8. RECOMMENDATIONS")
    lines.append("-" * 100)
    if sil_best > 0.5:
        lines.append(f"   ✓ EXCELLENT: Use k={k_best} for strong clustering results")
    elif sil_best > 0.3:
        lines.append(f"   ✓ GOOD: k={k_best} shows acceptable clustering quality")
    else:
        lines.append(f"   ! CAUTION: Both partitions show weak separation. Consider:")
        lines.append(f"     - Feature engineering or preprocessing")
        lines.append(f"     - Different linkage methods (complete, average)")
        lines.append(f"     - Alternative clustering algorithms (DBSCAN, K-means)")
        lines.append(f"     - Analyzing data for natural groupings")
    
    lines.append(f"")
    lines.append(f"   Next Steps:")
    lines.append(f"     - Review the generated visualizations to understand cluster characteristics")
    lines.append(f"     - Use detailed statistics (CSV files) for domain-specific interpretation")
    lines.append(f"     - Compare with other clustering methods (K-means, DBSCAN)")
    lines.append(f"     - Validate findings using external domain knowledge")
    lines.append("")
    
    lines.append("=" * 100)
    lines.append(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 100)
    
    return lines



if __name__ == "__main__":
    main()
