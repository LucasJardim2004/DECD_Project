from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class CorrelationPlotGenerator:
    def __init__(self, base_dir: Path, output_dir: str = "output_correlacao") -> None:
        self.base_dir = base_dir
        self.output_dir = self.base_dir / output_dir
        self.plots_dir = self.output_dir / "plots"
        self._ensure_dir(self.output_dir)
        self._ensure_dir(self.plots_dir)

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _correlation_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
        cols = matrix.columns.tolist()
        rows = []
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                value = matrix.loc[c1, c2]
                rows.append(
                    {
                        "var1": c1,
                        "var2": c2,
                        "correlation": value,
                        "abs_correlation": abs(value),
                    }
                )

        pairs_df = pd.DataFrame(rows)
        if pairs_df.empty:
            return pairs_df
        return pairs_df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    @staticmethod
    def _cramers_v(series_x: pd.Series, series_y: pd.Series) -> float:
        contingency = pd.crosstab(series_x, series_y)
        if contingency.empty:
            return np.nan

        observed = contingency.to_numpy(dtype=float)
        n = observed.sum()
        if n == 0:
            return np.nan

        row_sums = observed.sum(axis=1, keepdims=True)
        col_sums = observed.sum(axis=0, keepdims=True)
        expected = row_sums @ col_sums / n

        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.nansum((observed - expected) ** 2 / expected)

        phi2 = chi2 / n
        r, k = observed.shape
        if r <= 1 or k <= 1:
            return 0.0

        # Bias-corrected Cramer's V to avoid inflated values in sparse tables.
        phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0.0
        r_corr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
        k_corr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
        denom = min((k_corr - 1), (r_corr - 1))
        if denom <= 0:
            return 0.0
        return float(np.sqrt(phi2_corr / denom))

    def _plot_heatmap(
        self,
        matrix: pd.DataFrame,
        title: str,
        output_file: Path,
        cmap: str = "coolwarm",
        center: float | None = 0.0,
        vmin: float | None = -1.0,
        vmax: float | None = 1.0,
    ) -> None:
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 8},
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_numeric_correlation(self, numeric_csv: Path, top_k: int = 20) -> None:
        df = pd.read_csv(numeric_csv)
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.dropna(axis=1, how="all")

        if numeric_df.shape[1] < 2:
            raise ValueError("Not enough numeric columns to compute correlations.")

        pearson = numeric_df.corr(method="pearson")
        spearman = numeric_df.corr(method="spearman")

        pearson.to_csv(self.output_dir / "matriz_correlacao_pearson.csv")
        spearman.to_csv(self.output_dir / "matriz_correlacao_spearman.csv")
        self._correlation_pairs(pearson).head(top_k).to_csv(
            self.output_dir / "top_correlacoes_pearson.csv", index=False
        )
        self._correlation_pairs(spearman).head(top_k).to_csv(
            self.output_dir / "top_correlacoes_spearman.csv", index=False
        )

        self._plot_heatmap(
            pearson,
            "Matriz de Correlacao (Pearson) - Dados Numericos",
            self.plots_dir / "heatmap_pearson_numeric.png",
            cmap="coolwarm",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
        )
        self._plot_heatmap(
            spearman,
            "Matriz de Correlacao (Spearman) - Dados Numericos",
            self.plots_dir / "heatmap_spearman_numeric.png",
            cmap="coolwarm",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
        )

    def generate_categorical_correlation(self, categorical_csv: Path) -> None:
        df = pd.read_csv(categorical_csv).astype("string")
        columns = df.columns.tolist()
        matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)

        for i, col_i in enumerate(columns):
            for j in range(i + 1, len(columns)):
                col_j = columns[j]
                valid = df[[col_i, col_j]].dropna()
                score = self._cramers_v(valid[col_i], valid[col_j])
                matrix.loc[col_i, col_j] = score
                matrix.loc[col_j, col_i] = score

        matrix.to_csv(self.output_dir / "matriz_associacao_cramers_v.csv")
        self._plot_heatmap(
            matrix,
            "Matriz de Associacao (Cramer's V) - Dados Categoricos",
            self.plots_dir / "heatmap_cramers_v_categorical.png",
            cmap="YlGnBu",
            center=None,
            vmin=0.0,
            vmax=1.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation analysis for prepared CVD datasets")
    parser.add_argument(
        "--numeric-csv",
        default="output_preparacao/CVD_numeric.csv",
        help="Numeric CSV file name/path",
    )
    parser.add_argument(
        "--categorical-csv",
        default="output_preparacao/CVD_categorical.csv",
        help="Categorical CSV file name/path",
    )
    parser.add_argument("--output-dir", default="output_correlacao", help="Output folder name")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K correlated pairs to save")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    numeric_csv = Path(args.numeric_csv)
    categorical_csv = Path(args.categorical_csv)
    if not numeric_csv.is_absolute():
        numeric_csv = base_dir / numeric_csv
    if not categorical_csv.is_absolute():
        categorical_csv = base_dir / categorical_csv

    generator = CorrelationPlotGenerator(base_dir=base_dir, output_dir=args.output_dir)
    generator.generate_numeric_correlation(numeric_csv=numeric_csv, top_k=args.top_k)
    generator.generate_categorical_correlation(categorical_csv=categorical_csv)

    print("Correlation analysis completed.")
    print(f"Numeric CSV read: {numeric_csv}")
    print(f"Categorical CSV read: {categorical_csv}")
    print(f"Output folder: {generator.output_dir}")


if __name__ == "__main__":
    main()
