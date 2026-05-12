"""
K-NN Classification with Elbow Method Analysis
Predicting Heart Disease using K-Nearest Neighbors

Este script:
1. Carrega dados de treino (85%) e teste (15%)
2. Testa diferentes valores de K (1 a 30)
3. Usa validação cruzada para avaliar cada K
4. Aplica método do cotovelo para encontrar K ótimo
5. Treina modelo final e valida com dados de teste
6. Gera visualizações e relatórios
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

TRAIN_FILE = "CVD_train_85pct.csv"
TEST_FILE = "CVD_test_15pct.csv"
OUTPUT_DIR = Path("output_knn")
TARGET_COL = "Heart_Disease"
RANDOM_STATE = 42
K_MAX = 30
CV_FOLDS = 5

# Criar diretório de output
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def print_section(title: str) -> None:
    """Imprimir secção formatada."""
    print("\n" + "=" * 90)
    print(title.center(90))
    print("=" * 90)


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carregar dados de treino e teste."""
    print("Carregando dados...")
    
    # Treino
    df_train = pd.read_csv(TRAIN_FILE)
    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL]
    
    # Teste
    df_test = pd.read_csv(TEST_FILE)
    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]
    
    print(f"✓ Dados carregados:")
    print(f"  - Treino: {X_train.shape[0]} registos, {X_train.shape[1]} features")
    print(f"  - Teste: {X_test.shape[0]} registos, {X_test.shape[1]} features")
    print(f"  - Distribuição no treino - Heart_Disease=0: {(y_train==0).sum()}, "
          f"Heart_Disease=1: {(y_train==1).sum()}")
    
    return X_train, y_train, X_test, y_test


def elbow_method_analysis(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Método do cotovelo: testar K de 1 a 30 com validação cruzada.
    
    Returns:
        Dicionário com resultados: k_values, train_scores, cv_scores, best_k
    """
    print_section("MÉTODO DO COTOVELO - TESTANDO DIFERENTES VALORES DE K")
    
    k_values = range(1, K_MAX + 1)
    train_scores = []
    cv_scores = []
    cv_stds = []
    
    print("Testando K de 1 a 30 com validação cruzada (5 folds)...")
    
    for k in k_values:
        # Criar modelo
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        
        # Treino no conjunto todo
        knn.fit(X_train, y_train)
        train_score = knn.score(X_train, y_train)
        train_scores.append(train_score)
        
        # Validação cruzada
        cv_score = cross_val_score(knn, X_train, y_train, cv=CV_FOLDS, scoring='accuracy')
        cv_scores.append(cv_score.mean())
        cv_stds.append(cv_score.std())
        
        if k % 5 == 0 or k == 1:
            print(f"  K={k:2d}: Treino={train_score:.4f}, CV={cv_score.mean():.4f} (±{cv_score.std():.4f})")
    
    # Encontrar melhor K
    best_k = np.argmax(cv_scores) + 1
    best_cv_score = cv_scores[best_k - 1]
    
    print(f"\n✓ Melhor K encontrado: {best_k} com CV Score: {best_cv_score:.4f}")
    
    return {
        'k_values': list(k_values),
        'train_scores': train_scores,
        'cv_scores': cv_scores,
        'cv_stds': cv_stds,
        'best_k': best_k
    }


def plot_elbow_curve(results: dict) -> None:
    """Plotar curva do cotovelo."""
    k_values = results['k_values']
    train_scores = results['train_scores']
    cv_scores = results['cv_scores']
    cv_stds = results['cv_stds']
    best_k = results['best_k']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot
    ax.plot(k_values, train_scores, 'o-', linewidth=2, markersize=4, 
            label='Acurácia no Treino', alpha=0.7)
    ax.plot(k_values, cv_scores, 's-', linewidth=2, markersize=4,
            label=f'Acurácia CV (5-folds)', alpha=0.7)
    ax.fill_between(k_values, 
                     np.array(cv_scores) - np.array(cv_stds),
                     np.array(cv_scores) + np.array(cv_stds),
                     alpha=0.2)
    
    # Destaque K ótimo
    best_cv = cv_scores[best_k - 1]
    ax.plot(best_k, best_cv, 'r*', markersize=20, label=f'Melhor K={best_k}')
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Valor de K', fontsize=12, fontweight='bold')
    ax.set_ylabel('Acurácia', fontsize=12, fontweight='bold')
    ax.set_title('Método do Cotovelo para K-NN\nEncontrar Melhor Número de Vizinhos', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, K_MAX + 1, 2))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "knn_elbow_curve.png", dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo: {OUTPUT_DIR}/knn_elbow_curve.png")
    plt.close()


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, best_k: int):
    """Treinar modelo final com melhor K."""
    print_section("TREINANDO MODELO FINAL")
    
    knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    knn.fit(X_train, y_train)
    
    # Treino
    train_accuracy = knn.score(X_train, y_train)
    print(f"✓ Modelo treinado com K={best_k}")
    print(f"  - Acurácia no treino: {train_accuracy:.4f}")
    
    # Salvar modelo
    joblib.dump(knn, OUTPUT_DIR / f"knn_k{best_k}_model.joblib")
    print(f"✓ Modelo salvo: {OUTPUT_DIR}/knn_k{best_k}_model.joblib")
    
    return knn


def evaluate_model(model: KNeighborsClassifier, X_test: pd.DataFrame, 
                   y_test: pd.Series, best_k: int) -> dict:
    """Avaliar modelo no conjunto de teste."""
    print_section("AVALIAÇÃO NO CONJUNTO DE TESTE")
    
    # Predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Resultados do Modelo K-NN (K={best_k}):")
    print(f"  - Acurácia:  {accuracy:.4f}")
    print(f"  - Precisão: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"  - ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatriz de Confusão:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Classification Report
    print(f"\nDetalhes por classe:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Sem Doença', 'Com Doença']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }


def plot_evaluation_metrics(evaluation: dict) -> None:
    """Plotar métricas de avaliação."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Confusion Matrix
    ax = axes[0, 0]
    cm = evaluation['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusão')
    ax.set_xticklabels(['Sem Doença', 'Com Doença'])
    ax.set_yticklabels(['Sem Doença', 'Com Doença'])
    
    # Métricas
    ax = axes[0, 1]
    metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [evaluation['accuracy'], evaluation['precision'], 
              evaluation['recall'], evaluation['f1'], evaluation['roc_auc']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.barh(metrics, values, color=colors, alpha=0.7)
    ax.set_xlim([0, 1])
    ax.set_xlabel('Score')
    ax.set_title('Métricas de Desempenho')
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + 0.02, i, f'{value:.3f}', va='center')
    
    # ROC Curve
    ax = axes[1, 0]
    y_test_data = evaluation['y_test']
    y_pred_proba = evaluation['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test_data, y_pred_proba)
    roc_auc = evaluation['roc_auc']
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Classificador Aleatório')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curva ROC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribuição de Probabilidades
    ax = axes[1, 1]
    ax.hist([y_pred_proba[y_test_data == 0], y_pred_proba[y_test_data == 1]],
            bins=30, label=['Sem Doença', 'Com Doença'], alpha=0.7)
    ax.set_xlabel('Probabilidade Predita')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição de Probabilidades Preditas')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "knn_evaluation_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo: {OUTPUT_DIR}/knn_evaluation_metrics.png")
    plt.close()


def save_results_summary(results: dict, evaluation: dict, best_k: int) -> None:
    """Salvar resumo dos resultados em ficheiro."""
    
    summary_text = f"""
================================================================================
K-NN CLASSIFICATION MODEL - HEART DISEASE PREDICTION
================================================================================

CONFIGURAÇÃO:
- Dataset de Treino: CVD_train_85pct.csv ({len(results['X_train'])} registos)
- Dataset de Teste: CVD_test_15pct.csv ({len(results['X_test'])} registos)
- Target: Heart_Disease (Classificação Binária)
- Método: K-Nearest Neighbors com Método do Cotovelo

MÉTODO DO COTOVELO:
- Testados valores de K: 1 a 30
- Validação Cruzada: {CV_FOLDS} folds
- Melhor K encontrado: {best_k}
- CV Score (melhor K): {results['best_cv_score']:.4f}

RESULTADOS NO CONJUNTO DE TESTE:
- Acurácia:  {evaluation['accuracy']:.4f}
- Precisão: {evaluation['precision']:.4f}
- Recall:    {evaluation['recall']:.4f}
- F1-Score:  {evaluation['f1']:.4f}
- ROC-AUC:   {evaluation['roc_auc']:.4f}

MATRIZ DE CONFUSÃO:
- Verdadeiros Negativos (TN):  {evaluation['confusion_matrix'][0,0]}
- Falsos Positivos (FP):       {evaluation['confusion_matrix'][0,1]}
- Falsos Negativos (FN):       {evaluation['confusion_matrix'][1,0]}
- Verdadeiros Positivos (TP):  {evaluation['confusion_matrix'][1,1]}

ARQUIVOS GERADOS:
- knn_k{best_k}_model.joblib - Modelo treinado
- knn_elbow_curve.png - Gráfico do método do cotovelo
- knn_evaluation_metrics.png - Métricas de desempenho
- knn_results_summary.txt - Este ficheiro

================================================================================
"""
    
    with open(OUTPUT_DIR / "knn_results_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"✓ Resumo salvo: {OUTPUT_DIR}/knn_results_summary.txt")


def save_detailed_results(results: dict, evaluation: dict) -> None:
    """Salvar resultados detalhados em CSV."""
    
    # Resultados do método do cotovelo
    elbow_df = pd.DataFrame({
        'K': results['k_values'],
        'Train_Accuracy': results['train_scores'],
        'CV_Accuracy': results['cv_scores'],
        'CV_Std': results['cv_stds']
    })
    elbow_df.to_csv(OUTPUT_DIR / "knn_elbow_results.csv", index=False)
    print(f"✓ Resultados do cotovelo: {OUTPUT_DIR}/knn_elbow_results.csv")


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    """Executar análise completa."""
    
    print_section("ANÁLISE K-NN COM MÉTODO DO COTOVELO")
    print("Prevendo Heart Disease usando K-Nearest Neighbors")
    
    # 1. Carregar dados
    X_train, y_train, X_test, y_test = load_data()
    
    # 2. Método do cotovelo
    results = elbow_method_analysis(X_train, y_train)
    results['X_train'] = X_train
    results['X_test'] = X_test
    results['y_train'] = y_train
    results['y_test'] = y_test
    results['best_cv_score'] = results['cv_scores'][results['best_k'] - 1]
    
    # 3. Plotar curva do cotovelo
    plot_elbow_curve(results)
    
    # 4. Treinar modelo final
    best_k = results['best_k']
    model = train_final_model(X_train, y_train, best_k)
    
    # 5. Avaliar modelo
    evaluation = evaluate_model(model, X_test, y_test, best_k)
    evaluation['y_test'] = y_test  # Adicionar y_test para visualização
    
    # 6. Plotar métricas
    plot_evaluation_metrics(evaluation)
    
    # 7. Salvar resultados
    save_results_summary(results, evaluation, best_k)
    save_detailed_results(results, evaluation)
    
    print_section("ANÁLISE CONCLUÍDA COM SUCESSO")
    print(f"✓ Todos os resultados foram salvos em: {OUTPUT_DIR}/")
    print(f"✓ Ficheiros gerados:")
    print(f"   - knn_k{best_k}_model.joblib")
    print(f"   - knn_elbow_curve.png")
    print(f"   - knn_evaluation_metrics.png")
    print(f"   - knn_elbow_results.csv")
    print(f"   - knn_results_summary.txt")


if __name__ == "__main__":
    main()