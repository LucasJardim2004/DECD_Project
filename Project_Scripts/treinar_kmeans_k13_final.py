"""
Train the final KMeans model with k=9 and generate output artifacts.

Default input:
- output_preparacao/CVD_numeric_zscore.csv

Outputs:
- output_clustering/kmeans_k9_model.joblib
- output_clustering/kmeans_k9_cluster_summary.csv
- output_clustering/kmeans_k9_pca_scatter.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final KMeans model with k=9")
    parser.add_argument(
        "--csv",
        default="output_preparacao/CVD_numeric_zscore.csv",
        help="Input CSV path (relative to script folder or absolute).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_clustering",
        help="Output directory for model and plots.",
    )
    parser.add_argument(
        "--sample-plot-size",
        type=int,
        default=30000,
        help="Max rows used for PCA scatter plot.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="KMeans n_init.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    output_dir = base_dir / args.output_dir
    ensure_dir(output_dir)

    df = pd.read_csv(csv_path)
    X_df = df.select_dtypes(include="number").dropna().reset_index(drop=True)
    if X_df.empty:
        raise ValueError("No numeric data available after dropna().")

    kmeans = KMeans(n_clusters=9, random_state=args.random_state, n_init=args.n_init)
    labels = kmeans.fit_predict(X_df)

    # Save trained model.
    model_path = output_dir / "kmeans_k9_model.joblib"
    dump(kmeans, model_path)

    # Save cluster summary.
    summary = (
        pd.Series(labels, name="cluster")
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="count")
    )
    summary["pct"] = (summary["count"] / len(X_df) * 100).round(4)
    summary_path = output_dir / "kmeans_k9_cluster_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Build a PCA 2D scatter visualization on a sample for readability/performance.
    plot_n = min(args.sample_plot_size, len(X_df))
    plot_df = X_df.sample(n=plot_n, random_state=args.random_state) if plot_n < len(X_df) else X_df
    plot_labels = kmeans.predict(plot_df)

    pca = PCA(n_components=2, random_state=args.random_state)
    coords = pca.fit_transform(plot_df)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=plot_labels,
        cmap="tab20",
        s=10,
        alpha=0.7,
    )
    plt.title("KMeans Final (k=9) - PCA 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    legend = plt.legend(*scatter.legend_elements(num=9), title="Cluster", loc="best", fontsize=8)
    plt.gca().add_artist(legend)
    plt.tight_layout()

    plot_path = output_dir / "kmeans_k9_pca_scatter.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Final KMeans model training finished.")
    print(f"Input CSV: {csv_path}")
    print(f"Rows used (training): {len(X_df)}")
    print(f"Columns used: {', '.join(X_df.columns)}")
    print(f"Model saved: {model_path}")
    print(f"Cluster summary saved: {summary_path}")
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
