"""
Train the final KMeans model with k=5 and generate output artifacts.

Default input:
- output_preparacao/CVD_numeric_zscore.csv

Outputs:
- output_clustering/kmeans_k5_model.joblib
- output_clustering/kmeans_k5_cluster_summary.csv
- output_clustering/kmeans_k5_cluster_profiles.csv
- output_clustering/kmeans_k5_report.md
- output_clustering/kmeans_k5_pca_scatter.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from transformadores import BINARY_COLS, CONTINUOUS_COLS, ORDINAL_ORDER


REPORT_FEATURES = [
    "General_Health",
    "Checkup",
    "Exercise",
    "Heart_Disease",
    "Skin_Cancer",
    "Other_Cancer",
    "Depression",
    "Diabetes",
    "Arthritis",
    "Sex",
    "Age_Category",
    "Height_(cm)",
    "Weight_(kg)",
    "BMI",
    "Smoking_History",
    "Alcohol_Consumption",
    "Fruit_Consumption",
    "Green_Vegetables_Consumption",
    "FriedPotato_Consumption",
]


COLUMN_LABELS = {
    "General_Health": "saúde geral",
    "Checkup": "frequência de check-up",
    "Exercise": "prática de exercício",
    "Heart_Disease": "doença cardíaca",
    "Skin_Cancer": "cancro da pele",
    "Other_Cancer": "outro cancro",
    "Depression": "depressão",
    "Diabetes": "diabetes",
    "Arthritis": "artrite",
    "Sex": "sexo masculino",
    "Age_Category": "faixa etária",
    "Height_(cm)": "altura",
    "Weight_(kg)": "peso",
    "BMI": "IMC",
    "Smoking_History": "histórico de tabagismo",
    "Alcohol_Consumption": "consumo de álcool",
    "Fruit_Consumption": "consumo de fruta",
    "Green_Vegetables_Consumption": "consumo de vegetais verdes",
    "FriedPotato_Consumption": "consumo de batata frita",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final KMeans model with k=5")
    parser.add_argument(
        "--csv",
        default="output_preparacao/CVD_numeric_zscore.csv",
        help="Input CSV path (relative to script folder or absolute).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of clusters for KMeans.",
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


def format_number(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def format_percent(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def feature_display_name(feature: str) -> str:
    return COLUMN_LABELS.get(feature, feature)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def cluster_profile_table(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    features: list[str],
) -> pd.DataFrame:
    cluster_df = df.copy()
    cluster_df["cluster"] = labels

    cluster_slice = cluster_df[cluster_df["cluster"] == cluster_id]
    profile = pd.DataFrame(index=features)
    profile["mean"] = cluster_slice[features].mean()
    profile["global_mean"] = cluster_df[features].mean()
    profile["delta"] = profile["mean"] - profile["global_mean"]
    profile["abs_delta"] = profile["delta"].abs()
    return profile


def describe_feature(feature: str, delta: float) -> str:
    label = feature_display_name(feature)
    if feature in BINARY_COLS:
        if delta > 0:
            return f"maior prevalência de {label}"
        if delta < 0:
            return f"menor prevalência de {label}"
        return f"prevalência semelhante de {label}"

    if feature == "Age_Category":
        if delta > 0:
            return "perfil etário mais avançado"
        if delta < 0:
            return "perfil etário mais jovem"
        return "perfil etário médio"

    if feature == "General_Health":
        if delta > 0:
            return "melhor avaliação de saúde geral"
        if delta < 0:
            return "pior avaliação de saúde geral"
        return "avaliação de saúde geral semelhante"

    if feature == "Checkup":
        if delta > 0:
            return "maior regularidade de check-up"
        if delta < 0:
            return "menor regularidade de check-up"
        return "regularidade de check-up semelhante"

    if feature in CONTINUOUS_COLS:
        if delta > 0:
            return f"{label} acima da média"
        if delta < 0:
            return f"{label} abaixo da média"
        return f"{label} próximo da média"

    if delta > 0:
        return f"tendência acima da média em {label}"
    if delta < 0:
        return f"tendência abaixo da média em {label}"
    return f"{label} semelhante à média"


def top_feature_bullets(profile: pd.DataFrame, features: list[str], top_n: int = 3) -> list[str]:
    subset = profile.loc[profile.index.intersection(features)].sort_values(
        "delta",
        key=lambda s: s.abs(),
        ascending=False,
    )
    bullets: list[str] = []

    for feature, row in subset.head(top_n).iterrows():
        bullets.append(
            f"{feature_display_name(feature)}: {describe_feature(feature, float(row['delta']))}"
        )

    return bullets


def build_cluster_report(
    df: pd.DataFrame,
    labels: np.ndarray,
    model: KMeans,
    args: argparse.Namespace,
    csv_path: Path,
) -> tuple[str, pd.DataFrame]:
    cluster_df = df.copy()
    cluster_df["cluster"] = labels
    report_features = [feature for feature in REPORT_FEATURES if feature in df.columns]

    counts = (
        pd.Series(labels, name="cluster")
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="count")
    )
    counts["pct"] = (counts["count"] / len(df) * 100).round(2)

    profile_rows: list[pd.DataFrame] = []
    report_lines = [
        "# Relatório de Clustering KMeans",
        "",
        "## Metodologia",
        f"- Dataset de entrada: `{csv_path}`",
        f"- Observações usadas no treino: {len(df)}",
        f"- Variáveis usadas: {len(df.columns)}",
        f"- Algoritmo: KMeans com `k={args.clusters}`",
        f"- `n_init`: {args.n_init}",
        f"- `random_state`: {args.random_state}",
        f"- Inertia final: {format_number(float(model.inertia_))}",
        "- Métrica interna do KMeans: distância euclidiana aos centróides",
        "- Verificação da distância entre clusters: este script não utiliza distância máxima entre clusters; isso seria um critério de linkage hierárquico (complete linkage), não KMeans",
        "",
        "## Resumo Global",
        "",
        dataframe_to_markdown_table(counts),
        "",
    ]

    for cluster_id in sorted(counts["cluster"].tolist()):
        profile = cluster_profile_table(cluster_df, labels, cluster_id, report_features)
        profile_rows.append(
            profile.assign(cluster=cluster_id)
            .reset_index()
            .rename(columns={"index": "feature"})
        )

        cluster_count = int(counts.loc[counts["cluster"] == cluster_id, "count"].iloc[0])
        cluster_pct = float(counts.loc[counts["cluster"] == cluster_id, "pct"].iloc[0])

        positive_features = [
            feature
            for feature in profile.index
            if feature in CONTINUOUS_COLS or feature in ORDINAL_ORDER or feature in BINARY_COLS
        ]
        top_positive = profile.loc[positive_features].sort_values("delta", ascending=False).head(3)
        top_negative = profile.loc[positive_features].sort_values("delta", ascending=True).head(3)
        binary_features = [feature for feature in profile.index if feature in BINARY_COLS]
        dominant_binary = profile.loc[binary_features].sort_values("abs_delta", ascending=False).head(3)

        report_lines.extend(
            [
                f"## Cluster {cluster_id}",
                f"- Observações: {cluster_count} ({format_percent(cluster_pct)}%)",
            ]
        )

        if not top_positive.empty:
            report_lines.append("- Principais traços acima da média:")
            for feature, row in top_positive.iterrows():
                report_lines.append(
                    f"  - {feature_display_name(feature)}: {describe_feature(feature, float(row['delta']))} ({format_number(float(row['mean']))})"
                )

        if not top_negative.empty:
            report_lines.append("- Principais traços abaixo da média:")
            for feature, row in top_negative.iterrows():
                report_lines.append(
                    f"  - {feature_display_name(feature)}: {describe_feature(feature, float(row['delta']))} ({format_number(float(row['mean']))})"
                )

        if not dominant_binary.empty:
            report_lines.append("- Indicadores binários mais distintivos:")
            for feature, row in dominant_binary.iterrows():
                report_lines.append(
                    f"  - {feature_display_name(feature)}: {format_number(float(row['mean']))} no cluster vs {format_number(float(row['global_mean']))} no global"
                )

        narrative = top_feature_bullets(profile, positive_features, top_n=4)
        if narrative:
            report_lines.append("- Leitura interpretativa:")
            report_lines.append("  - " + "; ".join(narrative) + ".")

        report_lines.append("")

    profile_df = pd.concat(profile_rows, ignore_index=True)
    profile_df = profile_df[["cluster", "feature", "mean", "global_mean", "delta", "abs_delta"]]
    profile_df = profile_df.sort_values(["cluster", "abs_delta"], ascending=[True, False])

    return "\n".join(report_lines), profile_df


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

    if args.clusters < 2:
        raise ValueError("clusters must be >= 2.")

    kmeans = KMeans(n_clusters=args.clusters, random_state=args.random_state, n_init=args.n_init)
    labels = kmeans.fit_predict(X_df)

    # Save trained model.
    model_path = output_dir / f"kmeans_k{args.clusters}_model.joblib"
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
    summary_path = output_dir / f"kmeans_k{args.clusters}_cluster_summary.csv"
    summary.to_csv(summary_path, index=False)

    report_text, profile_df = build_cluster_report(
        df=X_df,
        labels=labels,
        model=kmeans,
        args=args,
        csv_path=csv_path,
    )

    profile_path = output_dir / f"kmeans_k{args.clusters}_cluster_profiles.csv"
    profile_df.to_csv(profile_path, index=False)

    report_path = output_dir / f"kmeans_k{args.clusters}_report.md"
    report_path.write_text(report_text, encoding="utf-8")

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

    plot_path = output_dir / f"kmeans_k{args.clusters}_pca_scatter.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Final KMeans model training finished.")
    print(f"Input CSV: {csv_path}")
    print(f"Rows used (training): {len(X_df)}")
    print(f"Columns used: {', '.join(X_df.columns)}")
    print(f"Clusters: {args.clusters}")
    print(f"Model saved: {model_path}")
    print(f"Cluster summary saved: {summary_path}")
    print(f"Cluster profiles saved: {profile_path}")
    print(f"Report saved: {report_path}")
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
