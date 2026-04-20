"""
Análise de Visualização - CVD_cleaned.csv

Script de análise completa de dados usando todas as técnicas de visualização
de pandas, matplotlib e seaborn conforme apresentado no ficheiro 04-visualization.ipynb
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurar o estilo do seaborn
sns.set_theme()
plt.rcParams['figure.figsize'] = (12, 6)


# ============================================================================
# 1. CARREGAR E EXPLORAR OS DADOS
# ============================================================================

print("=" * 80)
print("1. CARREGANDO E EXPLORANDO OS DADOS")
print("=" * 80)

# Carregar o dataset
df = pd.read_csv('./CVD_cleaned.csv')

# Informações básicas do dataset
print(f"\nForma do dataset: {df.shape}")
print("\nPrimeiras linhas:")
print(df.head())

print("\nTipos de dados:")
print(df.dtypes)

print("\nValores em falta:")
print(df.isnull().sum())

print("\nEstatísticas básicas:")
print(df.describe())

# Identificar colunas numéricas e categóricas
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nColunas numéricas: {numerical_cols}")
print(f"\nColunas categóricas: {categorical_cols}")


# ============================================================================
# 2. ANÁLISE UNIVARIADA - VARIÁVEIS NUMÉRICAS
# ============================================================================

print("\n" + "=" * 80)
print("2. ANÁLISE UNIVARIADA - VARIÁVEIS NUMÉRICAS")
print("=" * 80)

# 2.1 Histogramas (usando pandas plot)
print("\n2.1 - Gerando Histogramas...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols[:6]):
    df[col].plot(kind='hist', ax=axes[idx], title=f'Histograma - {col}', bins=30)
    axes[idx].set_ylabel('Frequência')

plt.tight_layout()
plt.savefig('histogramas.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.2 Box Plots (Diagramas de Caixa)
print("2.2 - Gerando Box Plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols[:6]):
    df[col].plot(kind='box', ax=axes[idx], title=f'Box Plot - {col}')

plt.tight_layout()
plt.savefig('box_plots.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.3 Histogramas com KDE (Seaborn)
print("2.3 - Gerando Histogramas com KDE...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols[:6]):
    sns.histplot(data=df, x=col, kde=True, ax=axes[idx])
    axes[idx].set_title(f'Histograma com KDE - {col}')

plt.tight_layout()
plt.savefig('histogramas_kde.png', dpi=100, bbox_inches='tight')
plt.show()


# ============================================================================
# 3. ANÁLISE UNIVARIADA - VARIÁVEIS CATEGÓRICAS
# ============================================================================

print("\n" + "=" * 80)
print("3. ANÁLISE UNIVARIADA - VARIÁVEIS CATEGÓRICAS")
print("=" * 80)

# 3.1 Gráficos de Barras
print("\n3.1 - Gerando Gráficos de Barras...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols[:6]):
    df[col].value_counts().plot(kind='bar', ax=axes[idx], title=f'Contagem - {col}')
    axes[idx].set_ylabel('Frequência')
    axes[idx].set_xlabel(col)
    plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('graficos_barras.png', dpi=100, bbox_inches='tight')
plt.show()

# 3.2 Gráficos de Pizza
print("3.2 - Gerando Gráficos de Pizza...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Selecionar 3 variáveis categóricas principais
main_cats = categorical_cols[:3]

for idx, col in enumerate(main_cats):
    df[col].value_counts().plot(kind='pie', ax=axes[idx], title=f'Proporção - {col}', autopct='%1.1f%%')
    axes[idx].set_ylabel('')

plt.tight_layout()
plt.savefig('graficos_pizza.png', dpi=100, bbox_inches='tight')
plt.show()

# 3.3 Count Plots (Seaborn)
print("3.3 - Gerando Count Plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, col in enumerate(main_cats):
    sns.countplot(data=df, x=col, ax=axes[idx])
    axes[idx].set_title(f'Contagem - {col}')
    plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('count_plots.png', dpi=100, bbox_inches='tight')
plt.show()


# ============================================================================
# 4. ANÁLISE BIVARIADA - RELACIONAMENTOS ENTRE VARIÁVEIS
# ============================================================================

print("\n" + "=" * 80)
print("4. ANÁLISE BIVARIADA - RELACIONAMENTOS ENTRE VARIÁVEIS")
print("=" * 80)

# 4.1 Scatter Plots
print("\n4.1 - Gerando Scatter Plots...")
if len(numerical_cols) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Criar scatter plots entre diferentes pares
    pairs = [(0, 1), (0, 2), (1, 2), (2, 3)] if len(numerical_cols) >= 4 else [(0, 1)]
    
    for idx, (i, j) in enumerate(pairs[:4]):
        if i < len(numerical_cols) and j < len(numerical_cols):
            df.plot(kind='scatter', x=numerical_cols[i], y=numerical_cols[j], 
                   ax=axes[idx], alpha=0.5)
            axes[idx].set_title(f'{numerical_cols[i]} vs {numerical_cols[j]}')
    
    plt.tight_layout()
    plt.savefig('scatter_plots.png', dpi=100, bbox_inches='tight')
    plt.show()

# 4.2 Joint Plots (Distância Conjunta + Marginais)
print("4.2 - Gerando Joint Plots...")
if len(numerical_cols) >= 2:
    # Scatter com marginal (Hex)
    sns.jointplot(data=df, x=numerical_cols[0], y=numerical_cols[1], kind='hex')
    plt.suptitle(f'{numerical_cols[0]} vs {numerical_cols[1]} (Hex)', y=1.00)
    plt.savefig('jointplot_hex.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Regressão linear
    sns.jointplot(data=df, x=numerical_cols[0], y=numerical_cols[1], kind='reg')
    plt.suptitle(f'{numerical_cols[0]} vs {numerical_cols[1]} (Regressão)', y=1.00)
    plt.savefig('jointplot_reg.png', dpi=100, bbox_inches='tight')
    plt.show()

# 4.3 Categorical Plots
print("4.3 - Gerando Categorical Plots...")
if len(numerical_cols) >= 1 and len(categorical_cols) >= 1:
    with sns.axes_style('ticks'):
        g = sns.catplot(data=df, x=categorical_cols[0], y=numerical_cols[0], 
                       kind='box', height=6, aspect=2)
        g.set_axis_labels(categorical_cols[0], numerical_cols[0])
        plt.savefig('catplot_box.png', dpi=100, bbox_inches='tight')
        plt.show()
        
    # Count plot por categoria
    with sns.axes_style('white'):
        g = sns.catplot(data=df, x=categorical_cols[0], kind='count', 
                       height=5, aspect=2, color='steelblue')
        g.set_ylabels('Contagem')
        plt.savefig('catplot_count.png', dpi=100, bbox_inches='tight')
        plt.show()


# ============================================================================
# 5. ANÁLISE MULTIVARIADA - INTERAÇÕES ENTRE MÚLTIPLAS VARIÁVEIS
# ============================================================================

print("\n" + "=" * 80)
print("5. ANÁLISE MULTIVARIADA - INTERAÇÕES ENTRE MÚLTIPLAS VARIÁVEIS")
print("=" * 80)

# 5.1 Pair Plot (Matriz de Scatter Plot)
print("\n5.1 - Gerando Pair Plot...")
if len(numerical_cols) >= 2:
    # Selecionar apenas as primeiras 4 variáveis numéricas para melhor visualização
    subset_cols = numerical_cols[:4]
    g = sns.pairplot(df[subset_cols], diag_kind='hist', plot_kws={'alpha': 0.6})
    g.fig.suptitle('Pair Plot - Variáveis Numéricas', y=1.00)
    plt.savefig('pairplot.png', dpi=100, bbox_inches='tight')
    plt.show()

# 5.2 FacetGrid - Histogramas Segmentados
print("5.2 - Gerando FacetGrid...")
if len(numerical_cols) >= 1 and len(categorical_cols) >= 1:
    try:
        g = sns.FacetGrid(df, col=categorical_cols[0], col_wrap=3, height=4)
        g.map(sns.histplot, numerical_cols[0], bins=20)
        g.fig.suptitle(f'Distribuição de {numerical_cols[0]} por {categorical_cols[0]}', 
                       y=1.00)
        plt.savefig('facetgrid.png', dpi=100, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Não foi possível criar o FacetGrid: {e}")


# ============================================================================
# 6. ANÁLISE DE CORRELAÇÃO E HEATMAPS
# ============================================================================

print("\n" + "=" * 80)
print("6. ANÁLISE DE CORRELAÇÃO E HEATMAPS")
print("=" * 80)

# 6.1 Calcular correlação
print("\n6.1 - Calculando Matriz de Correlação...")
correlation_matrix = df[numerical_cols].corr()
print("\nMatriz de Correlação:")
print(correlation_matrix)

# 6.2 Heatmap de Correlação
print("\n6.2 - Gerando Heatmap de Correlação...")
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Variáveis Numéricas', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('heatmap_correlacao.png', dpi=100, bbox_inches='tight')
plt.show()

# 6.3 KDE Plots Bi-dimensionais
print("6.3 - Gerando KDE Plots Bi-dimensionais...")
if len(numerical_cols) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # KDE plot 2D com fill
    sns.kdeplot(data=df, x=numerical_cols[0], y=numerical_cols[1], 
               fill=True, ax=axes[0])
    axes[0].set_title(f'KDE 2D - {numerical_cols[0]} vs {numerical_cols[1]}')
    
    # Histplot 2D
    sns.histplot(data=df, x=numerical_cols[0], y=numerical_cols[1], 
                ax=axes[1], bins=20)
    axes[1].set_title(f'Histograma 2D - {numerical_cols[0]} vs {numerical_cols[1]}')
    
    plt.tight_layout()
    plt.savefig('kde_histplot_2d.png', dpi=100, bbox_inches='tight')
    plt.show()


# ============================================================================
# 7. ANÁLISE DE CORRELAÇÕES FORTES E INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("7. ANÁLISE DE CORRELAÇÕES FORTES E INSIGHTS")
print("=" * 80)

# 7.1 Identificar correlações fortes
print("\nCORRELAÇÕES FORTES (|r| > 0.7):")
print("-" * 80)

strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            strong_correlations.append({
                'Variável 1': correlation_matrix.columns[i],
                'Variável 2': correlation_matrix.columns[j],
                'Correlação': correlation_matrix.iloc[i, j]
            })

if strong_correlations:
    strong_corr_df = pd.DataFrame(strong_correlations)
    print(strong_corr_df.to_string(index=False))
else:
    print("Nenhuma correlação forte (|r| > 0.7) encontrada.")


# ============================================================================
# 8. RESUMO FINAL
# ============================================================================

print("\n" + "=" * 80)
print("RESUMO DA ANÁLISE EXPLORATÓRIA")
print("=" * 80)
print(f"\nDimensões do dataset: {df.shape[0]} linhas, {df.shape[1]} colunas")
print(f"\nVariáveis numéricas: {len(numerical_cols)}")
print(f"  {', '.join(numerical_cols)}")
print(f"\nVariáveis categóricas: {len(categorical_cols)}")
print(f"  {', '.join(categorical_cols)}")
print(f"\nValores faltantes: {df.isnull().sum().sum()}")
print("\nTécnicas de visualização aplicadas:")
print("  ✓ Histogramas e Densidade (KDE)")
print("  ✓ Diagramas de Caixa (Box Plots)")
print("  ✓ Gráficos de Dispersão (Scatter Plots)")
print("  ✓ Gráficos de Barras e Contagem")
print("  ✓ Gráficos de Pizza")
print("  ✓ Joint Plots (Distribuição Conjunta)")
print("  ✓ Pair Plots (Matriz de Relacionamentos)")
print("  ✓ FacetGrid (Segmentação)")
print("  ✓ Categorical Plots")
print("  ✓ Heatmaps de Correlação")
print("  ✓ KDE 2D e Histogramas 2D")
print("\n" + "=" * 80)
print("Análise concluída! Todos os gráficos foram salvos em PNG.")
print("=" * 80)
