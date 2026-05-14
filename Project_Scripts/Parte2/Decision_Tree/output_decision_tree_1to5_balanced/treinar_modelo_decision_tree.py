"""
Modelo de Árvores de Decisão melhorado para previsão de Heart_Disease
usando o dataset de treino 1:5.

Melhorias incluídas:
- class_weight='balanced' para penalizar mais os erros na classe minoritária
- seleção de hiperparâmetros com validação estratificada
- escolha de limiar de decisão na validação para favorecer recall/F2
- poda mais conservadora para reduzir sobreajuste
- outputs completos numa pasta própria

Dados usados:
- Treino: ../output_decision_tree_1to5/CVD_train_1to5.csv
- Teste:  ../CVD_test_15pct.csv

Este script segue o estilo do material da pasta Source_Material, com:
- DecisionTreeClassifier
- criterion='entropy'
- plot_tree
- classification_report e confusion_matrix
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


sns.set_theme(style="whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
TRAIN_FILE = BASE_DIR / "output_decision_tree_1to5" / "CVD_train_1to5.csv"
TEST_FILE = BASE_DIR / "CVD_test_15pct.csv"
OUTPUT_DIR = SCRIPT_DIR
TARGET_COL_CANDIDATES = ["heart_disease", "Heart_Disease"]
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.01)

MODEL_CANDIDATES = [
    {"max_depth": 3, "min_samples_split": 20, "min_samples_leaf": 20},
    {"max_depth": 4, "min_samples_split": 20, "min_samples_leaf": 15},
    {"max_depth": 5, "min_samples_split": 20, "min_samples_leaf": 10},
    {"max_depth": 6, "min_samples_split": 20, "min_samples_leaf": 10},
    {"max_depth": 8, "min_samples_split": 30, "min_samples_leaf": 10},
    {"max_depth": None, "min_samples_split": 50, "min_samples_leaf": 20},
]


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


def entropy_from_labels(y: pd.Series) -> float:
    probabilities = y.value_counts(normalize=True)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


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


def compute_feature_information_gain(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for feature in X.columns:
        stump = DecisionTreeClassifier(
            criterion="entropy",
            class_weight="balanced",
            max_depth=1,
            random_state=RANDOM_STATE,
        )
        stump.fit(X[[feature]], y)

        tree = stump.tree_
        root_entropy = float(tree.impurity[0])
        weighted_child_entropy = 0.0
        left_child = tree.children_left[0]
        right_child = tree.children_right[0]
        total_samples = tree.weighted_n_node_samples[0]

        for child in (left_child, right_child):
            child_samples = tree.weighted_n_node_samples[child]
            child_impurity = tree.impurity[child]
            weighted_child_entropy += (child_samples / total_samples) * child_impurity

        info_gain = root_entropy - weighted_child_entropy
        threshold = float(tree.threshold[0]) if left_child != right_child else np.nan

        records.append(
            {
                "feature": feature,
                "best_threshold": threshold,
                "root_entropy": root_entropy,
                "weighted_child_entropy": weighted_child_entropy,
                "information_gain": info_gain,
            }
        )

    return pd.DataFrame(records).sort_values("information_gain", ascending=False).reset_index(drop=True)


def make_model(params: dict[str, object]) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="entropy",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        **params,
    )


def predict_with_threshold(model: DecisionTreeClassifier, X: pd.DataFrame, threshold: float) -> np.ndarray:
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)


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


def evaluate_model_with_threshold(model: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict[str, object]:
    y_pred = predict_with_threshold(model, X, threshold)
    metrics = evaluate_predictions(y, y_pred)
    metrics["y_pred"] = y_pred
    metrics["y_proba"] = model.predict_proba(X)[:, 1]
    metrics["threshold"] = threshold
    return metrics


def select_best_parameters(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict[str, object], pd.DataFrame, float, pd.DataFrame]:
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
        val_proba = model.predict_proba(X_val)[:, 1]
        default_pred = (val_proba >= 0.5).astype(int)
        metrics = evaluate_predictions(y_val, default_pred)

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
        "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        "min_samples_split": int(best_row["min_samples_split"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
    }

    best_model = make_model(best_params)
    best_model.fit(X_fit, y_fit)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    threshold_rows: list[dict[str, float]] = []
    best_threshold = 0.5
    best_threshold_score = -1.0
    best_threshold_recall = -1.0

    for threshold in THRESHOLD_GRID:
        y_pred = (val_proba >= threshold).astype(int)
        metrics = evaluate_predictions(y_val, y_pred)
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

        if (
            metrics["f2"] > best_threshold_score
            or (np.isclose(metrics["f2"], best_threshold_score) and metrics["recall"] > best_threshold_recall)
        ):
            best_threshold = float(threshold)
            best_threshold_score = float(metrics["f2"])
            best_threshold_recall = float(metrics["recall"])

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


def save_information_gain_plot(info_gain_df: pd.DataFrame, filename: Path) -> None:
    top_15 = info_gain_df.head(15).sort_values("information_gain")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top_15, x="information_gain", y="feature", ax=ax, palette="magma")
    ax.set_xlabel("Information Gain")
    ax.set_ylabel("Atributo")
    ax.set_title("Top 15 Atributos por Information Gain (stump de profundidade 1)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(y_true: pd.Series, y_proba: np.ndarray, filename: Path) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
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
    train_entropy: float,
    info_gain_df: pd.DataFrame,
    best_params: dict[str, object],
    best_threshold: float,
    candidates_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    train_metrics: dict[str, object],
    test_metrics: dict[str, object],
    pr_auc_test: float,
) -> None:
    lines = [
        "# Relatório do Modelo de Árvores de Decisão Melhorado",
        "",
        f"- Coluna alvo: {target_col}",
        f"- Entropia do conjunto de treino: {train_entropy:.6f}",
        f"- Limiar de decisão selecionado na validação: {best_threshold:.2f}",
        "",
        "## Melhores atributos por information gain",
    ]

    for _, row in info_gain_df.head(10).iterrows():
        lines.append(
            f"- {row['feature']}: IG={row['information_gain']:.6f}, threshold={row['best_threshold']:.6f}"
        )

    lines.extend(
        [
            "",
            "## Hiperparâmetros selecionados",
            f"- max_depth: {best_params['max_depth']}",
            f"- min_samples_split: {best_params['min_samples_split']}",
            f"- min_samples_leaf: {best_params['min_samples_leaf']}",
            f"- class_weight: balanced",
            "",
            "## Melhor validação interna",
        ]
    )

    best_candidate = candidates_df.iloc[0]
    lines.extend(
        [
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
            "- O modelo usa class_weight='balanced' e limiar de decisão ajustado para favorecer recall, reduzindo falsos negativos.",
        ]
    )

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

    print_section("ENTROPIA E INFORMATION GAIN")
    train_entropy = entropy_from_labels(y_train)
    print(f"Entropia do conjunto de treino: {train_entropy:.6f}")

    info_gain_df = compute_feature_information_gain(X_train, y_train)
    info_gain_df.to_csv(OUTPUT_DIR / "decision_tree_information_gain.csv", index=False)
    print(f"✓ Ficheiro guardado: {OUTPUT_DIR / 'decision_tree_information_gain.csv'}")
    for _, row in info_gain_df.head(10).iterrows():
        print(f"  - {row['feature']}: IG={row['information_gain']:.6f}, threshold={row['best_threshold']:.6f}")
    save_information_gain_plot(info_gain_df, OUTPUT_DIR / "decision_tree_information_gain_top15.png")
    print(f"✓ Gráfico guardado: {OUTPUT_DIR / 'decision_tree_information_gain_top15.png'}")

    print_section("SELEÇÃO DE HIPERPARÂMETROS E LIMIAR")
    best_params, candidates_df, best_threshold, threshold_df = select_best_parameters(X_train, y_train)
    candidates_df.to_csv(OUTPUT_DIR / "decision_tree_validation_candidates.csv", index=False)
    threshold_df.to_csv(OUTPUT_DIR / "decision_tree_threshold_candidates.csv", index=False)
    (OUTPUT_DIR / "decision_tree_best_threshold.txt").write_text(f"{best_threshold:.4f}\n", encoding="utf-8")
    print(f"✓ Ficheiros guardados: {OUTPUT_DIR / 'decision_tree_validation_candidates.csv'}")
    print(f"✓ Ficheiros guardados: {OUTPUT_DIR / 'decision_tree_threshold_candidates.csv'}")
    print(f"✓ Limiar selecionado: {best_threshold:.4f}")
    print("Melhor configuração encontrada:")
    print(f"  - max_depth: {best_params['max_depth']}")
    print(f"  - min_samples_split: {best_params['min_samples_split']}")
    print(f"  - min_samples_leaf: {best_params['min_samples_leaf']}")
    print(f"  - class_weight: balanced")

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

    joblib.dump(final_model, OUTPUT_DIR / "decision_tree_model.joblib")
    print(f"✓ Modelo guardado: {OUTPUT_DIR / 'decision_tree_model.joblib'}")

    feature_importance = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feature_importance_df = feature_importance.reset_index()
    feature_importance_df.columns = ["feature", "importance"]
    feature_importance_df.to_csv(OUTPUT_DIR / "decision_tree_feature_importances.csv", index=False)
    print(f"✓ Ficheiro guardado: {OUTPUT_DIR / 'decision_tree_feature_importances.csv'}")
    save_feature_importance_plot(feature_importance, OUTPUT_DIR / "decision_tree_feature_importances_top15.png")
    print(f"✓ Gráfico guardado: {OUTPUT_DIR / 'decision_tree_feature_importances_top15.png'}")

    print_section("GERAÇÃO DE VISUALIZAÇÕES E RELATÓRIO")
    save_confusion_matrix(
        train_metrics["confusion_matrix"],
        OUTPUT_DIR / "decision_tree_confusion_matrix_train.png",
        "Matriz de Confusão - Treino",
    )
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        OUTPUT_DIR / "decision_tree_confusion_matrix_test.png",
        "Matriz de Confusão - Teste",
    )
    print(f"✓ Matriz de confusão de treino guardada: {OUTPUT_DIR / 'decision_tree_confusion_matrix_train.png'}")
    print(f"✓ Matriz de confusão de teste guardada: {OUTPUT_DIR / 'decision_tree_confusion_matrix_test.png'}")

    tree_depth_for_plot = 4 if final_model.get_depth() > 4 else final_model.get_depth()
    fig, ax = plt.subplots(figsize=(28, 14))
    plot_tree(
        final_model,
        feature_names=X_train.columns,
        class_names=["Sem Doença", "Com Doença"],
        impurity=False,
        rounded=True,
        filled=True,
        max_depth=tree_depth_for_plot,
        fontsize=8,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_tree_plot_top_levels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Árvore guardada: {OUTPUT_DIR / 'decision_tree_plot_top_levels.png'}")

    tree_text = export_text(
        final_model,
        feature_names=list(X_train.columns),
        max_depth=6,
        show_weights=True,
    )
    (OUTPUT_DIR / "decision_tree_rules.txt").write_text(tree_text, encoding="utf-8")
    print(f"✓ Regras da árvore guardadas: {OUTPUT_DIR / 'decision_tree_rules.txt'}")

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
    ).to_csv(OUTPUT_DIR / "decision_tree_metrics_summary.csv", index=False)
    print(f"✓ Ficheiro guardado: {OUTPUT_DIR / 'decision_tree_metrics_summary.csv'}")

    pr_auc_test = save_precision_recall_curve(y_test, test_metrics["y_proba"], OUTPUT_DIR / "decision_tree_precision_recall_curve_test.png")
    print(f"✓ Curva Precision-Recall guardada: {OUTPUT_DIR / 'decision_tree_precision_recall_curve_test.png'}")

    write_report(
        OUTPUT_DIR / "decision_tree_report.md",
        target_col,
        train_entropy,
        info_gain_df,
        best_params,
        best_threshold,
        candidates_df,
        threshold_df,
        train_metrics,
        test_metrics,
        pr_auc_test,
    )
    print(f"✓ Relatório guardado: {OUTPUT_DIR / 'decision_tree_report.md'}")

    print_section("CONCLUÍDO")
    print(f"Todos os outputs foram guardados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
