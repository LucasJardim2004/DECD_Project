"""
KMeans elbow analysis using inertia (SSE).

Default input:
- output_preparacao/CVD_numeric_zscore.csv

Outputs:
- output_clustering/kmeans_elbow_metrics.csv
- output_clustering/elbow_kmeans.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KMeans elbow analysis")
    parser.add_argument(
        "--csv",
        default="output_preparacao/CVD_numeric_zscore.csv",
        help="Path to input CSV (relative to script folder or absolute).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_clustering",
        help="Folder to save metrics and plots.",
    )
    parser.add_argument("--k-min", type=int, default=2, help="Minimum k to test.")
    parser.add_argument("--k-max", type=int, default=30, help="Maximum k to test.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30000,
        help="Max number of rows used for clustering metrics.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="KMeans n_init parameter.",
    )
    return parser.parse_args()


def load_numeric_data(csv_path: Path, sample_size: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include="number").dropna().reset_index(drop=True)

    if numeric_df.empty:
        raise ValueError("Input CSV has no usable numeric rows after dropna().")

    if sample_size > 0 and len(numeric_df) > sample_size:
        sample_df = numeric_df.sample(n=sample_size, random_state=random_state)
    else:
        sample_df = numeric_df

    return numeric_df, sample_df


def run_kmeans_grid(
    X: np.ndarray,
    k_values: list[int],
    random_state: int,
    n_init: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        model.fit(X)

        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
            }
        )

    res = pd.DataFrame(rows)
    res["delta_inertia"] = res["inertia"].diff().fillna(0.0)
    res["inertia_drop_pct"] = (
        (-res["delta_inertia"] / res["inertia"].shift(1))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        * 100
    )
    return res


def recommend_k_elbow(res: pd.DataFrame, elbow_threshold_pct: float) -> int:
    # Elbow proxy: first k where additional inertia drop falls below threshold.
    elbow_candidates = res.loc[res["inertia_drop_pct"] < elbow_threshold_pct, "k"]
    if not elbow_candidates.empty:
        return int(elbow_candidates.iloc[0])
    return int(res.iloc[-1]["k"])


def plot_elbow(res: pd.DataFrame, k_elbow: int, output_file: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(res["k"], res["inertia"], marker="o", color="tab:blue")
    plt.axvline(k_elbow, linestyle="--", color="orange", label=f"Elbow proxy k={k_elbow}")
    plt.title("Metodo do Cotovelo - KMeans")
    plt.xlabel("k")
    plt.ylabel("Inertia (SSE)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    if args.k_min < 2:
        raise ValueError("k-min must be >= 2.")
    if args.k_max <= args.k_min:
        raise ValueError("k-max must be > k-min.")

    base_dir = Path(__file__).resolve().parent

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path

    output_dir = base_dir / args.output_dir
    ensure_dir(output_dir)

    full_df, sample_df = load_numeric_data(
        csv_path=csv_path,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    X = sample_df.to_numpy()

    k_values = list(range(args.k_min, args.k_max + 1))
    res = run_kmeans_grid(
        X=X,
        k_values=k_values,
        random_state=args.random_state,
        n_init=args.n_init,
    )

    k_recommended = recommend_k_elbow(res, elbow_threshold_pct=10.0)

    metrics_path = output_dir / "kmeans_elbow_metrics.csv"
    elbow_plot_path = output_dir / "elbow_kmeans.png"

    res.to_csv(metrics_path, index=False)
    plot_elbow(res, k_recommended, elbow_plot_path)

    print("KMeans analysis finished.")
    print(f"Input CSV: {csv_path}")
    print(f"Rows total: {len(full_df)}")
    print(f"Rows used: {len(sample_df)}")
    print(f"Columns used: {', '.join(full_df.columns)}")
    print(f"Recommended k (elbow proxy): {k_recommended}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved elbow plot: {elbow_plot_path}")


if __name__ == "__main__":
    main()
