"""
Naive Bayes Classification - Heart Disease Prediction
Análise com modelo supervisionado Gaussian Naive Bayes

Este script:
1. Carrega dados de treino e teste de uma pasta de análise
2. Avalia o modelo com validação cruzada (5-folds) no treino
3. Treina modelo GaussianNB no conjunto de treino completo
4. Valida com dados de teste (15%)
5. Gera visualizações: confusion matrix, métricas, curva ROC, distribuição de probabilidades
6. Gera relatório com resultados

Uso:
  python analise_naive_bayes.py <pasta_de_analise>

Exemplos:
  python analise_naive_bayes.py Analise_100
  python analise_naive_bayes.py Analise_Reduzida
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

sns.set_theme()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if len(sys.argv) < 2:
    print("Uso: python analise_naive_bayes.py <pasta_de_analise>")
    print("  Ex: python analise_naive_bayes.py Analise_100")
    print("  Ex: python analise_naive_bayes.py Analise_Reduzida")
    sys.exit(1)

DATA_DIR = SCRIPT_DIR / sys.argv[1]
if not DATA_DIR.is_dir():
    print(f"Erro: pasta '{DATA_DIR}' não encontrada.")
    sys.exit(1)

TEST_FILE = DATA_DIR / "CVD_test_15pct.csv"
train_candidates = sorted(DATA_DIR.glob("CVD_train_*.csv"))
if not train_candidates:
    print(f"Erro: nenhum ficheiro CVD_train_*.csv encontrado em '{DATA_DIR}'.")
    sys.exit(1)
TRAIN_FILE = train_candidates[0]

OUTPUT_DIR = DATA_DIR / "output_naive_bayes"
TARGET_COL = "Heart_Disease"
RANDOM_STATE = 42
CV_FOLDS = 5

print(f"Pasta de análise: {DATA_DIR.name}")
print(f"Ficheiro de treino: {TRAIN_FILE.name}")
print(f"Ficheiro de teste:  {TEST_FILE.name}")

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title.center(90))
    print("=" * 90)


def display_evaluation(labels, predictions, class_names):
    """Mostrar classification report e confusion matrix (baseado no Source Material)."""
    print(classification_report(labels, predictions, target_names=class_names))
    cm = confusion_matrix(labels, predictions)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot()
    plt.tight_layout()
    return cm


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carregar dados de treino e teste."""
    print("Carregando dados...")

    df_train = pd.read_csv(TRAIN_FILE)
    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL]

    df_test = pd.read_csv(TEST_FILE)
    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    print(f"✓ Dados carregados:")
    print(f"  - Treino: {X_train.shape[0]} registos, {X_train.shape[1]} features")
    print(f"  - Teste:  {X_test.shape[0]} registos, {X_test.shape[1]} features")
    print(f"  - Features: {list(X_train.columns)}")
    print(f"  - Distribuição no treino - {TARGET_COL}=0: {(y_train==0).sum()}, "
          f"{TARGET_COL}=1: {(y_train==1).sum()}")
    print(f"  - Distribuição no teste  - {TARGET_COL}=0: {(y_test==0).sum()}, "
          f"{TARGET_COL}=1: {(y_test==1).sum()}")

    return X_train, y_train, X_test, y_test


# ============================================================================
# VALIDAÇÃO CRUZADA NO TREINO
# ============================================================================

def cross_validation_analysis(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Validação cruzada com GaussianNB no conjunto de treino.
    Baseado no Source Material (08-classification):
      - cross_val_predict para obter predições de cada fold
      - display_evaluation para confusion matrix e classification report
    """
    print_section("VALIDAÇÃO CRUZADA (5-FOLDS) NO CONJUNTO DE TREINO")

    nb_cls = GaussianNB()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # cross_val_predict: predição de cada exemplo quando faz parte do fold de validação
    cv_pred = cross_val_predict(nb_cls, X_train, y_train, cv=cv)

    # cross_val_score para múltiplas métricas
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_scores = {}
    for metric in scoring_metrics:
        scores = cross_val_score(nb_cls, X_train, y_train, cv=cv, scoring=metric)
        cv_scores[metric] = {'mean': scores.mean(), 'std': scores.std(), 'values': scores}
        print(f"  CV {metric:>10s}: {scores.mean():.4f} ± {scores.std():.4f}  "
              f"(folds: {', '.join(f'{s:.4f}' for s in scores)})")

    # Mostrar classificação detalhada (Source Material pattern)
    class_names = ['Sem Doença', 'Com Doença']
    print(f"\nClassification Report (Cross-Validation):")
    cm_cv = display_evaluation(y_train, cv_pred, class_names)
    plt.savefig(OUTPUT_DIR / "nb_cv_confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo: {OUTPUT_DIR}/nb_cv_confusion_matrix.png")
    plt.close()

    # Visualização de predições corretas vs incorretas (Source Material pattern)
    fig, ax = plt.subplots(figsize=(10, 6))
    correct = cv_pred == y_train
    colors = correct.map({True: 'green', False: 'red'})
    ax.scatter(range(len(y_train)), y_train, c=colors, alpha=0.3, s=5)
    ax.set_xlabel('Índice do Registo')
    ax.set_ylabel('Classe Real')
    ax.set_title(f'Naive Bayes - Validação Cruzada\n'
                 f'Verde = Correto, Vermelho = Incorreto '
                 f'(Accuracy: {cv_scores["accuracy"]["mean"]:.4f})')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_names)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nb_cv_predictions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo: {OUTPUT_DIR}/nb_cv_predictions.png")
    plt.close()

    return {
        'cv_pred': cv_pred,
        'cv_scores': cv_scores,
        'cv_confusion_matrix': cm_cv,
    }


# ============================================================================
# TREINO E AVALIAÇÃO FINAL
# ============================================================================

def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Treinar GaussianNB no treino completo e avaliar no teste.
    Baseado no Source Material:
      classifiers['Naive Bayes'] = nb_cls.fit(X_train, y_train)
      evaluation on test set with .predict() and display_evaluation()
    """
    print_section("TREINO DO MODELO FINAL (CONJUNTO COMPLETO DE TREINO)")

    nb_cls = GaussianNB()
    nb_cls.fit(X_train, y_train)

    train_accuracy = nb_cls.score(X_train, y_train)
    print(f"✓ Modelo GaussianNB treinado")
    print(f"  - Acurácia no treino: {train_accuracy:.4f}")

    # Salvar modelo
    joblib.dump(nb_cls, OUTPUT_DIR / "naive_bayes_model.joblib")
    print(f"✓ Modelo salvo: {OUTPUT_DIR}/naive_bayes_model.joblib")

    # --- Avaliação no conjunto de teste ---
    print_section("AVALIAÇÃO NO CONJUNTO DE TESTE (15%)")

    y_pred = nb_cls.predict(X_test)
    y_pred_proba = nb_cls.predict_proba(X_test)[:, 1]

    # Métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }

    print(f"Resultados do Modelo Naive Bayes no Teste:")
    for name, value in metrics.items():
        print(f"  - {name:>10s}: {value:.4f}")

    # Classification Report e Confusion Matrix (Source Material pattern)
    class_names = ['Sem Doença', 'Com Doença']
    print(f"\nClassification Report (Teste):")
    cm = display_evaluation(y_test, y_pred, class_names)
    plt.savefig(OUTPUT_DIR / "nb_test_confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo: {OUTPUT_DIR}/nb_test_confusion_matrix.png")
    plt.close()

    evaluation = {
        **metrics,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'train_accuracy': train_accuracy,
    }

    return nb_cls, evaluation


# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

def plot_evaluation(evaluation: dict, y_test: pd.Series) -> None:
    """Gerar gráficos de avaliação do modelo."""
    print_section("GERANDO VISUALIZAÇÕES")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- 1. Confusion Matrix com heatmap ---
    ax = axes[0, 0]
    cm = evaluation['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Sem Doença', 'Com Doença'],
                yticklabels=['Sem Doença', 'Com Doença'])
    ax.set_xlabel('Predito', fontweight='bold')
    ax.set_ylabel('Real', fontweight='bold')
    ax.set_title('Matriz de Confusão (Teste)', fontweight='bold')

    # --- 2. Barras de métricas ---
    ax = axes[0, 1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metric_values = [evaluation['accuracy'], evaluation['precision'],
                     evaluation['recall'], evaluation['f1'], evaluation['roc_auc']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.8)
    ax.set_xlim([0, 1])
    ax.set_xlabel('Score', fontweight='bold')
    ax.set_title('Métricas de Desempenho', fontweight='bold')
    for bar, value in zip(bars, metric_values):
        ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{value:.4f}', va='center', fontweight='bold')

    # --- 3. Curva ROC ---
    ax = axes[1, 0]
    y_pred_proba = evaluation['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = evaluation['roc_auc']
    ax.plot(fpr, tpr, linewidth=2, label=f'Naive Bayes (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Classificador Aleatório')
    ax.set_xlabel('Taxa de Falsos Positivos', fontweight='bold')
    ax.set_ylabel('Taxa de Verdadeiros Positivos', fontweight='bold')
    ax.set_title('Curva ROC', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- 4. Distribuição de probabilidades preditas ---
    ax = axes[1, 1]
    ax.hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]],
            bins=40, label=['Sem Doença', 'Com Doença'], alpha=0.7,
            color=['#1f77b4', '#d62728'])
    ax.set_xlabel('Probabilidade Predita (Heart Disease)', fontweight='bold')
    ax.set_ylabel('Frequência', fontweight='bold')
    ax.set_title('Distribuição de Probabilidades Preditas', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Naive Bayes - Avaliação do Modelo (Heart Disease Prediction)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nb_evaluation_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo: {OUTPUT_DIR}/nb_evaluation_metrics.png")
    plt.close()


def plot_feature_analysis(model: GaussianNB, feature_names: list) -> None:
    """Analisar parâmetros aprendidos pelo Naive Bayes (média e variância por classe)."""
    print("\nAnálise dos parâmetros aprendidos pelo modelo:")

    # GaussianNB guarda theta_ (médias) e var_ (variâncias) por classe
    means = pd.DataFrame(model.theta_, columns=feature_names,
                         index=['Sem Doença', 'Com Doença'])
    variances = pd.DataFrame(model.var_, columns=feature_names,
                             index=['Sem Doença', 'Com Doença'])

    # Diferença de médias entre classes (indica importância das features)
    mean_diff = (means.loc['Com Doença'] - means.loc['Sem Doença']).abs().sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Diferença absoluta de médias
    ax = axes[0]
    mean_diff.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
    ax.set_xlabel('Diferença Absoluta de Médias entre Classes', fontweight='bold')
    ax.set_title('Importância das Features\n(Diferença de Médias)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Médias por classe
    ax = axes[1]
    means.T.plot(kind='barh', ax=ax, alpha=0.8)
    ax.set_xlabel('Média', fontweight='bold')
    ax.set_title('Médias por Classe\n(Parâmetros GaussianNB)', fontweight='bold')
    ax.legend(title='Classe')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nb_feature_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo: {OUTPUT_DIR}/nb_feature_analysis.png")
    plt.close()

    # Salvar parâmetros em CSV
    means.to_csv(OUTPUT_DIR / "nb_class_means.csv")
    variances.to_csv(OUTPUT_DIR / "nb_class_variances.csv")
    mean_diff.to_frame('abs_mean_diff').to_csv(OUTPUT_DIR / "nb_feature_importance.csv")
    print(f"✓ Parâmetros salvos: nb_class_means.csv, nb_class_variances.csv, nb_feature_importance.csv")


# ============================================================================
# RELATÓRIO
# ============================================================================

def save_results(cv_results: dict, evaluation: dict,
                 X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Salvar resumo completo dos resultados."""

    cv = cv_results['cv_scores']
    cm = evaluation['confusion_matrix']

    summary = f"""
================================================================================
   NAIVE BAYES CLASSIFICATION - HEART DISEASE PREDICTION
================================================================================

CONFIGURAÇÃO:
  - Modelo: GaussianNB (sklearn)
  - Dataset Treino: {TRAIN_FILE.name} ({len(X_train)} registos)
  - Dataset Teste:  {TEST_FILE.name}  ({len(X_test)} registos)
  - Target: {TARGET_COL} (Classificação Binária)
  - Validação Cruzada: {CV_FOLDS} folds (StratifiedKFold)

RESULTADOS DA VALIDAÇÃO CRUZADA (TREINO):
  - Accuracy:  {cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}
  - Precision: {cv['precision']['mean']:.4f} ± {cv['precision']['std']:.4f}
  - Recall:    {cv['recall']['mean']:.4f} ± {cv['recall']['std']:.4f}
  - F1-Score:  {cv['f1']['mean']:.4f} ± {cv['f1']['std']:.4f}
  - ROC-AUC:   {cv['roc_auc']['mean']:.4f} ± {cv['roc_auc']['std']:.4f}

RESULTADOS NO CONJUNTO DE TESTE:
  - Accuracy:  {evaluation['accuracy']:.4f}
  - Precision: {evaluation['precision']:.4f}
  - Recall:    {evaluation['recall']:.4f}
  - F1-Score:  {evaluation['f1']:.4f}
  - ROC-AUC:   {evaluation['roc_auc']:.4f}

  - Acurácia no Treino: {evaluation['train_accuracy']:.4f}

MATRIZ DE CONFUSÃO (TESTE):
  - Verdadeiros Negativos (TN): {cm[0,0]:>6d}
  - Falsos Positivos (FP):      {cm[0,1]:>6d}
  - Falsos Negativos (FN):      {cm[1,0]:>6d}
  - Verdadeiros Positivos (TP): {cm[1,1]:>6d}

FICHEIROS GERADOS:
  - naive_bayes_model.joblib        - Modelo treinado
  - nb_cv_confusion_matrix.png      - Confusion matrix (validação cruzada)
  - nb_cv_predictions.png           - Predições corretas vs incorretas (CV)
  - nb_test_confusion_matrix.png    - Confusion matrix (teste)
  - nb_evaluation_metrics.png       - Métricas, curva ROC, distribuições
  - nb_feature_analysis.png         - Análise de features (parâmetros NB)
  - nb_class_means.csv              - Médias por classe
  - nb_class_variances.csv          - Variâncias por classe
  - nb_feature_importance.csv       - Importância das features
  - nb_results_summary.txt          - Este ficheiro

================================================================================
"""

    with open(OUTPUT_DIR / "nb_results_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"✓ Resumo salvo: {OUTPUT_DIR}/nb_results_summary.txt")

    # CSV com métricas de CV por fold
    cv_df = pd.DataFrame({
        metric: data['values'] for metric, data in cv.items()
    })
    cv_df.index = [f'Fold_{i+1}' for i in range(CV_FOLDS)]
    cv_df.loc['Mean'] = cv_df.mean()
    cv_df.loc['Std'] = cv_df.std()
    cv_df.to_csv(OUTPUT_DIR / "nb_cv_fold_results.csv")
    print(f"✓ Resultados CV por fold: {OUTPUT_DIR}/nb_cv_fold_results.csv")


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    print_section("ANÁLISE NAIVE BAYES - HEART DISEASE PREDICTION")
    print("Modelo supervisionado Gaussian Naive Bayes")

    # 1. Carregar dados
    X_train, y_train, X_test, y_test = load_data()

    # 2. Validação cruzada no treino
    cv_results = cross_validation_analysis(X_train, y_train)

    # 3. Treinar modelo final e avaliar no teste
    model, evaluation = train_and_evaluate(X_train, y_train, X_test, y_test)

    # 4. Visualizações de avaliação
    plot_evaluation(evaluation, y_test)

    # 5. Análise de features (parâmetros do NB)
    plot_feature_analysis(model, list(X_train.columns))

    # 6. Salvar resultados
    save_results(cv_results, evaluation, X_train, X_test)

    print_section("ANÁLISE CONCLUÍDA COM SUCESSO")
    print(f"✓ Todos os resultados em: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"   - {f.name}")


if __name__ == "__main__":
    main()
