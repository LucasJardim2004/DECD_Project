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
REDUCED_TRAIN_FILE = "CVD_train_balanced_1to2.csv"
OUTPUT_DIR = Path("output_knn_reduced")
TARGET_COL = "Heart_Disease"
RANDOM_STATE = 42
K_MAX = 15
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


def generate_reduced_dataset() -> str:
    """
    Gera dataset balanceado com proporção 1:2 (doentes:saudáveis).
    Estratégia: Todos os 21,215 registos com doença + 42,430 registos sem doença aleatórios.
    Objetivo: Melhorar Recall para evitar Falsos Negativos.
    """
    if Path(REDUCED_TRAIN_FILE).exists():
        print(f"✓ Dataset balanceado já existe: {REDUCED_TRAIN_FILE}")
        return REDUCED_TRAIN_FILE
    
    print("\n" + "=" * 90)
    print("GERANDO DATASET BALANCEADO (1:2 - DOENTES:SAUDÁVEIS)".center(90))
    print("=" * 90)
    
    df_train = pd.read_csv(TRAIN_FILE)
    n_original = len(df_train)
    print(f"\n1. Dataset original: {n_original} registos")
    print(f"   - Heart_Disease=0 (saudáveis): {(df_train[TARGET_COL]==0).sum()}")
    print(f"   - Heart_Disease=1 (doentes): {(df_train[TARGET_COL]==1).sum()}")
    
    # Separar em dois grupos
    df_disease = df_train[df_train[TARGET_COL] == 1].copy()
    df_healthy = df_train[df_train[TARGET_COL] == 0].copy()
    
    n_disease = len(df_disease)
    n_healthy_target = n_disease * 2  # Para proporção 1:2
    
    print(f"\n2. Estratégia 1:2:")
    print(f"   - Doentes (TODOS): {n_disease} registos")
    print(f"   - Saudáveis (amostra): {n_healthy_target} de {len(df_healthy)} disponíveis")
    
    # Sample aleatório dos saudáveis
    df_healthy_sample = df_healthy.sample(n=n_healthy_target, random_state=RANDOM_STATE)
    
    # Concatenar e embaralhar
    df_balanced = pd.concat([df_disease, df_healthy_sample], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Salvar
    df_balanced.to_csv(REDUCED_TRAIN_FILE, index=False)
    
    # Estatísticas finais
    n_final = len(df_balanced)
    n_disease_final = (df_balanced[TARGET_COL] == 1).sum()
    n_healthy_final = (df_balanced[TARGET_COL] == 0).sum()
    
    print(f"\n3. Dataset balanceado gerado com sucesso:")
    print(f"   - Total: {n_final} registos ({n_final/n_original*100:.1f}% dos originais)")
    print(f"   - Heart_Disease=0: {n_healthy_final} ({n_healthy_final/n_final*100:.2f}%)")
    print(f"   - Heart_Disease=1: {n_disease_final} ({n_disease_final/n_final*100:.2f}%)")
    print(f"   - Proporção: 1:{n_healthy_final/n_disease_final:.2f} (doentes:saudáveis)")
    print(f"   - Ficheiro: {REDUCED_TRAIN_FILE}\n")
    
    return REDUCED_TRAIN_FILE


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carregar dados de treino e teste."""
    print("Carregando dados...")
    
    # Treino (dataset reduzido)
    df_train = pd.read_csv(REDUCED_TRAIN_FILE)
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
    Método do cotovelo: testar apenas K IMPARES de 3 a 15 com validação cruzada.
    
    Returns:
        Dicionário com resultados: k_values, train_scores, cv_scores, best_k
    """
    print_section("MÉTODO DO COTOVELO - TESTANDO DIFERENTES VALORES DE K IMPARES")
    
    # Apenas K impares: 3, 5, 7, 9, 11, 13, 15
    k_values = range(3, K_MAX + 1, 2)
    train_scores = []

    # Vamos calcular várias métricas em CV e escolher K com base numa métrica alvo
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_means: dict = {m: [] for m in scoring}
    cv_stds: dict = {m: [] for m in scoring}

    print(f"Testando K impares de 3 a {K_MAX} com validação cruzada ({CV_FOLDS} folds)...")

    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train, y_train)
        train_score = knn.score(X_train, y_train)
        train_scores.append(train_score)

        # cross_validate para múltiplas métricas
        res = cross_validate(knn, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        for m in scoring:
            scores = res[f'test_{m}']
            cv_means[m].append(scores.mean())
            cv_stds[m].append(scores.std())

        # Mostrar progresso para cada K
        print(f"  K={k:2d}: Treino={train_score:.4f}, CV_recall={cv_means['recall'][-1]:.4f}, CV_f1={cv_means['f1'][-1]:.4f}")

    # Escolha do K: por RECALL (para otimizar detecção de doentes cardíacos)
    metric_to_optimize = 'recall'
    best_k_idx = int(np.argmax(cv_means[metric_to_optimize]))
    best_k = list(k_values)[best_k_idx]
    best_cv_score = cv_means[metric_to_optimize][best_k_idx]

    print(f"\n✓ Melhor K encontrado (por {metric_to_optimize}): {best_k} com CV {metric_to_optimize}: {best_cv_score:.4f}")

    # Para compatibilidade com o resto do script, mantenho 'cv_scores' como accuracy
    cv_scores_accuracy = cv_means['accuracy']

    return {
        'k_values': list(k_values),
        'train_scores': train_scores,
        'cv_scores': cv_scores_accuracy,
        'cv_stds': cv_stds,
        'cv_means': cv_means,
        'best_k': best_k,
        'best_cv_score': best_cv_score,
        'optimized_metric': metric_to_optimize
    }


def plot_elbow_curve(results: dict) -> None:
    """Plotar curva do cotovelo."""
    k_values = results['k_values']
    train_scores = results['train_scores']
    best_k = results['best_k']
    
    # Obter a métrica que foi usada para otimização
    metric = results['optimized_metric']
    cv_means = results['cv_means'][metric]
    cv_stds = results['cv_stds'][metric]
    
    # Labels formatados para display
    metric_label = metric.upper() if metric != 'roc_auc' else 'ROC-AUC'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot - mostrar a métrica de otimização, não acurácia
    ax.plot(k_values, train_scores, 'o-', linewidth=2, markersize=4, 
        label='Acurácia no Treino', alpha=0.7)
    ax.plot(k_values, cv_means, 's-', linewidth=2, markersize=4,
        label=f'{metric_label} CV (5-folds)', alpha=0.7)
    ax.fill_between(
    k_values,
    np.array(cv_means) - np.array(cv_stds),
    np.array(cv_means) + np.array(cv_stds),
    alpha=0.2,
    )
    
    # Destaque K ótimo
    best_k_idx = list(k_values).index(best_k)
    best_cv = cv_means[best_k_idx]
    ax.plot(best_k, best_cv, 'r*', markersize=20, label=f'Melhor K={best_k}')
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Valor de K', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_label}', fontsize=12, fontweight='bold')
    ax.set_title(f'Método do Cotovelo para K-NN\nOtimizado por {metric_label}', 
         fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
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
- Dataset de Treino: {REDUCED_TRAIN_FILE} ({len(results['X_train'])} registos)
- Dataset de Teste: CVD_test_15pct.csv ({len(results['X_test'])} registos)
- Target: Heart_Disease (Classificação Binária)
- Método: K-Nearest Neighbors com Método do Cotovelo

MÉTODO DO COTOVELO:
- Testados valores de K IMPARES: 3, 5, 7, 9, 11, 13, 15
- Validação Cruzada: {CV_FOLDS} folds
    - Melhor K encontrado: {best_k}
    - Métrica usada para selecionar K: {results.get('optimized_metric', 'accuracy').upper()}
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

    # Informação adicional em output terminal
    print(f"✓ Métrica usada para selecionar K: {results.get('optimized_metric', 'accuracy').upper()}")


def save_detailed_results(results: dict, evaluation: dict) -> None:
    """Salvar resultados detalhados em CSV."""
    
    # Resultados do método do cotovelo - salvar múltiplas métricas
    elbow_df = pd.DataFrame({
        'K': results['k_values'],
        'Train_Accuracy': results['train_scores'],
        'CV_Accuracy': results['cv_means']['accuracy'],
        'CV_Precision': results['cv_means']['precision'],
        'CV_Recall': results['cv_means']['recall'],
        'CV_F1': results['cv_means']['f1'],
        'CV_ROC_AUC': results['cv_means']['roc_auc'],
        'CV_Std_Recall': results['cv_stds']['recall']
    })
    elbow_df.to_csv(OUTPUT_DIR / "knn_elbow_results.csv", index=False)
    print(f"✓ Resultados do cotovelo: {OUTPUT_DIR}/knn_elbow_results.csv")


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    """Executar análise completa."""
    
    print_section("ANÁLISE K-NN COM MÉTODO DO COTOVELO (DATASET REDUZIDO)")
    print("Prevendo Heart Disease usando K-Nearest Neighbors")
    
    # 0. Gerar dataset reduzido (se ainda não existir)
    generate_reduced_dataset()
    
    # 1. Carregar dados
    X_train, y_train, X_test, y_test = load_data()
    
    # 2. Método do cotovelo
    results = elbow_method_analysis(X_train, y_train)
    results['X_train'] = X_train
    results['X_test'] = X_test
    results['y_train'] = y_train
    results['y_test'] = y_test
    # best_cv_score já é definido na função (para a métrica optimizada)
    
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