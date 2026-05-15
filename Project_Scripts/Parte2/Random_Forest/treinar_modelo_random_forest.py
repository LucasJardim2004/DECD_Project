"""
Modelo de Random Forest para previsão de Heart_Disease.

Este script segue o estilo do material da pasta Source_Material e usa:
- RandomForestClassifier
- class_weight='balanced_subsample'
- classificação binária com avaliação por classification_report e confusion_matrix
- visualizações e relatório em pasta própria

Dados usados:
- Treino: CVD_train_1to5.csv
- Teste:  CVD_test_15pct.csv

Objetivo:
- Criar um modelo mais robusto do que uma árvore única em problemas desbalanceados
- Usar várias árvores treinadas em subconjuntos diferentes dos dados e das variáveis
- Guardar outputs completos para análise posterior
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedShuffleSplit


sns.set_theme(style="whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = SCRIPT_DIR / "CVD_train_1to5.csv"
TEST_FILE = SCRIPT_DIR / "CVD_test_15pct.csv"
OUTPUT_DIR = SCRIPT_DIR / "output_random_forest"
TARGET_COL_CANDIDATES = ["heart_disease", "Heart_Disease"]
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

MODEL_CANDIDATES = [
    {"n_estimators": 200, "max_depth": 8, "min_samples_split": 20, "min_samples_leaf": 10},
    {"n_estimators": 300, "max_depth": 10, "min_samples_split": 20, "min_samples_leaf": 8},
    {"n_estimators": 400, "max_depth": 12, "min_samples_split": 20, "min_samples_leaf": 8},
    {"n_estimators": 500, "max_depth": 14, "min_samples_split": 30, "min_samples_leaf": 10},
    {"n_estimators": 600, "max_depth": None, "min_samples_split": 40, "min_samples_leaf": 15},
]

THRESHOLD_GRID = np.arange(0.10, 0.91, 0.01)


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title.center(100))
    print("=" * 100)


def resolve_target_column(columns: pd.Index) -> str:
    col_map = {column.lower(): column for column in columns}
    for candidate in TARGET_COL_CANDIDATES:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    raise ValueError("Coluna alvo não encontrada. Esperado: heart_disease ou Heart_Disease.")


def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {file_path}")
    return pd.read_csv(file_path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    target_col = resolve_target_column(df.columns)
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y, target_col


def print_class_balance(y: pd.Series, label: str) -> None:
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    total = len(y)
    print(f"\nDistribuição da variável alvo em {label}:")
    print(f"  - Sem doença cardíaca (0): {int(counts.loc[0])} ({(counts.loc[0] / total) * 100:.2f}%)")
    print(f"  - Com doença cardíaca (1): {int(counts.loc[1])} ({(counts.loc[1] / total) * 100:.2f}%)")


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Sem Doença", "Com Doença"],
            zero_division=0,
        ),
    }


def predict_with_threshold(model: RandomForestClassifier, X: pd.DataFrame, threshold: float) -> np.ndarray:
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)


def evaluate_model_with_threshold(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> dict[str, object]:
    y_pred = predict_with_threshold(model, X, threshold)
    metrics = evaluate_predictions(y, y_pred)
    metrics["y_pred"] = y_pred
    metrics["y_proba"] = model.predict_proba(X)[:, 1]
    metrics["threshold"] = threshold
    return metrics


def make_model(params: dict[str, object]) -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )


def select_best_parameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame, float, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)
    fit_index, val_index = next(splitter.split(X_train, y_train))

    X_fit = X_train.iloc[fit_index]
    y_fit = y_train.iloc[fit_index]
    X_val = X_train.iloc[val_index]
    y_val = y_train.iloc[val_index]

    candidate_rows: list[dict[str, object]] = []

    for params in MODEL_CANDIDATES:
        model = make_model(params)
        model.fit(X_fit, y_fit)
        val_pred = model.predict(X_val)
        metrics = evaluate_predictions(y_val, val_pred)

        candidate_rows.append(
            {
                **params,
                "validation_accuracy": metrics["accuracy"],
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_f1": metrics["f1"],
                "validation_f2": metrics["f2"],
            }
        )

    candidates_df = pd.DataFrame(candidate_rows).sort_values(
        by=["validation_f2", "validation_recall", "validation_f1", "validation_accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    best_row = candidates_df.iloc[0]
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        "min_samples_split": int(best_row["min_samples_split"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
    }

    best_model = make_model(best_params)
    best_model.fit(X_fit, y_fit)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    threshold_rows: list[dict[str, float]] = []
    best_threshold = 0.5
    best_score = -1.0
    best_recall = -1.0

    for threshold in THRESHOLD_GRID:
        val_pred = (val_proba >= threshold).astype(int)
        metrics = evaluate_predictions(y_val, val_pred)
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f2": metrics["f2"],
            }
        )

        if metrics["f2"] > best_score or (np.isclose(metrics["f2"], best_score) and metrics["recall"] > best_recall):
            best_threshold = float(threshold)
            best_score = float(metrics["f2"])
            best_recall = float(metrics["recall"])

    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        by=["f2", "recall", "f1", "precision", "accuracy"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    return best_params, candidates_df, best_threshold, threshold_df


def save_confusion_matrix(cm: np.ndarray, filename: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sem Doença", "Com Doença"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(feature_importances: pd.Series, filename: Path) -> None:
    top_features = feature_importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax, palette="viridis")
    ax.set_xlabel("Importância")
    ax.set_ylabel("Atributo")
    ax.set_title("Top 15 Importâncias das Variáveis")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(y_true: pd.Series, y_proba: np.ndarray, filename: Path) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AUC = {pr_auc:.4f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return float(pr_auc)


def write_report(
    report_path: Path,
    target_col: str,
    best_params: dict[str, object],
    best_threshold: float,
    candidates_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    train_metrics: dict[str, object],
    test_metrics: dict[str, object],
    pr_auc_test: float,
) -> None:
    best_candidate = candidates_df.iloc[0]

    lines = [
        "# Relatório do Modelo Random Forest",
        "",
        f"- Coluna alvo: {target_col}",
        f"- Limiar de decisão selecionado: {best_threshold:.2f}",
        "",
        "## Hiperparâmetros selecionados",
        f"- n_estimators: {best_params['n_estimators']}",
        f"- max_depth: {best_params['max_depth']}",
        f"- min_samples_split: {best_params['min_samples_split']}",
        f"- min_samples_leaf: {best_params['min_samples_leaf']}",
        f"- class_weight: balanced_subsample",
        "",
        "## Melhor validação interna",
        f"- validation_f2: {best_candidate['validation_f2']:.6f}",
        f"- validation_recall: {best_candidate['validation_recall']:.6f}",
        f"- validation_f1: {best_candidate['validation_f1']:.6f}",
        f"- validation_accuracy: {best_candidate['validation_accuracy']:.6f}",
        "",
        "## Métricas no treino",
        f"- accuracy: {train_metrics['accuracy']:.6f}",
        f"- precision: {train_metrics['precision']:.6f}",
        f"- recall: {train_metrics['recall']:.6f}",
        f"- f1: {train_metrics['f1']:.6f}",
        f"- f2: {train_metrics['f2']:.6f}",
        "",
        "## Métricas no teste",
        f"- accuracy: {test_metrics['accuracy']:.6f}",
        f"- precision: {test_metrics['precision']:.6f}",
        f"- recall: {test_metrics['recall']:.6f}",
        f"- f1: {test_metrics['f1']:.6f}",
        f"- f2: {test_metrics['f2']:.6f}",
        f"- pr_auc: {pr_auc_test:.6f}",
        "",
        "## Observação",
        "- Random Forest combina várias árvores e usa balanced_subsample para ajustar o peso das classes em cada amostra bootstrap.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("CARREGAMENTO DOS DADOS")
    train_df = load_dataset(TRAIN_FILE)
    test_df = load_dataset(TEST_FILE)

    X_train, y_train, target_col_train = split_features_target(train_df)
    X_test, y_test, target_col_test = split_features_target(test_df)

    if target_col_train != target_col_test:
        raise ValueError(
            f"Coluna alvo inconsistente entre treino ({target_col_train}) e teste ({target_col_test})."
        )

    target_col = target_col_train

    print(f"Ficheiro de treino: {TRAIN_FILE.name}")
    print(f"Ficheiro de teste:  {TEST_FILE.name}")
    print(f"Coluna alvo:        {target_col}")
    print(f"Treino: {X_train.shape[0]} registos, {X_train.shape[1]} variáveis")
    print(f"Teste:  {X_test.shape[0]} registos, {X_test.shape[1]} variáveis")
    print_class_balance(y_train, "treino")
    print_class_balance(y_test, "teste")

    print_section("SELEÇÃO DE HIPERPARÂMETROS")
    best_params, candidates_df, best_threshold, threshold_df = select_best_parameters(X_train, y_train)
    candidates_df.to_csv(OUTPUT_DIR / "rf_validation_candidates.csv", index=False)
    threshold_df.to_csv(OUTPUT_DIR / "rf_threshold_candidates.csv", index=False)
    (OUTPUT_DIR / "rf_best_threshold.txt").write_text(f"{best_threshold:.4f}\n", encoding="utf-8")
    print(f"✓ Ficheiros guardados: {OUTPUT_DIR / 'rf_validation_candidates.csv'}")
    print(f"✓ Ficheiros guardados: {OUTPUT_DIR / 'rf_threshold_candidates.csv'}")
    print(f"✓ Limiar selecionado: {best_threshold:.4f}")
    print("Melhor configuração encontrada:")
    print(f"  - n_estimators: {best_params['n_estimators']}")
    print(f"  - max_depth: {best_params['max_depth']}")
    print(f"  - min_samples_split: {best_params['min_samples_split']}")
    print(f"  - min_samples_leaf: {best_params['min_samples_leaf']}")
    print(f"  - class_weight: balanced_subsample")

    print_section("TREINO DO MODELO FINAL")
    final_model = make_model(best_params)
    final_model.fit(X_train, y_train)

    train_metrics = evaluate_model_with_threshold(final_model, X_train, y_train, best_threshold)
    test_metrics = evaluate_model_with_threshold(final_model, X_test, y_test, best_threshold)

    print("Métricas no treino:")
    print(f"  - accuracy: {train_metrics['accuracy']:.6f}")
    print(f"  - precision: {train_metrics['precision']:.6f}")
    print(f"  - recall: {train_metrics['recall']:.6f}")
    print(f"  - f1: {train_metrics['f1']:.6f}")
    print(f"  - f2: {train_metrics['f2']:.6f}")

    print("\nMétricas no teste:")
    print(f"  - accuracy: {test_metrics['accuracy']:.6f}")
    print(f"  - precision: {test_metrics['precision']:.6f}")
    print(f"  - recall: {test_metrics['recall']:.6f}")
    print(f"  - f1: {test_metrics['f1']:.6f}")
    print(f"  - f2: {test_metrics['f2']:.6f}")

    joblib.dump(final_model, OUTPUT_DIR / "random_forest_model.joblib")
    print(f"✓ Modelo guardado: {OUTPUT_DIR / 'random_forest_model.joblib'}")

    feature_importance = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feature_importance_df = feature_importance.reset_index()
    feature_importance_df.columns = ["feature", "importance"]
    feature_importance_df.to_csv(OUTPUT_DIR / "random_forest_feature_importances.csv", index=False)
    print(f"✓ Ficheiro guardado: {OUTPUT_DIR / 'random_forest_feature_importances.csv'}")
    save_feature_importance_plot(feature_importance, OUTPUT_DIR / "random_forest_feature_importances_top15.png")
    print(f"✓ Gráfico guardado: {OUTPUT_DIR / 'random_forest_feature_importances_top15.png'}")

    print_section("GERAÇÃO DE VISUALIZAÇÕES E RELATÓRIO")
    save_confusion_matrix(
        train_metrics["confusion_matrix"],
        OUTPUT_DIR / "random_forest_confusion_matrix_train.png",
        "Matriz de Confusão - Treino",
    )
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        OUTPUT_DIR / "random_forest_confusion_matrix_test.png",
        "Matriz de Confusão - Teste",
    )
    print(f"✓ Matriz de confusão de treino guardada: {OUTPUT_DIR / 'random_forest_confusion_matrix_train.png'}")
    print(f"✓ Matriz de confusão de teste guardada: {OUTPUT_DIR / 'random_forest_confusion_matrix_test.png'}")

    pr_auc_test = save_precision_recall_curve(y_test, test_metrics["y_proba"], OUTPUT_DIR / "random_forest_precision_recall_curve_test.png")
    print(f"✓ Curva Precision-Recall guardada: {OUTPUT_DIR / 'random_forest_precision_recall_curve_test.png'}")

    pd.DataFrame(
        [
            {
                "split": "train",
                "accuracy": train_metrics["accuracy"],
                "precision": train_metrics["precision"],
                "recall": train_metrics["recall"],
                "f1": train_metrics["f1"],
                "f2": train_metrics["f2"],
            },
            {
                "split": "test",
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "f2": test_metrics["f2"],
            },
        ]
    ).to_csv(OUTPUT_DIR / "random_forest_metrics_summary.csv", index=False)
    print(f"✓ Ficheiro guardado: {OUTPUT_DIR / 'random_forest_metrics_summary.csv'}")

    write_report(
        OUTPUT_DIR / "random_forest_report.md",
        target_col,
        best_params,
        best_threshold,
        candidates_df,
        threshold_df,
        train_metrics,
        test_metrics,
        pr_auc_test,
    )
    print(f"✓ Relatório guardado: {OUTPUT_DIR / 'random_forest_report.md'}")

    print_section("CONCLUÍDO")
    print(f"Todos os outputs foram guardados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
