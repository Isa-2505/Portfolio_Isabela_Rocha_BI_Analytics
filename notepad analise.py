import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Cores Serasa (padrão)
azul_serasa = '#1a1a3e'
rosa_serasa = '#E91E63'
rosa_claro = '#F48FB1'

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#f5f5f5'
plt.rcParams['font.family'] = 'sans-serif'

print("=" * 60)
print("MODELAGEM DE RISCO DE CRÉDITO - SERASA")
print("=" * 60)

# Carregar dados
dados = pd.read_csv('dados_risco.csv')
print(f"\n✅ Dados carregados: {len(dados)} clientes")
print("\nPrimeiros 5 clientes:")
print(dados.head())

# Preparar dados
X = dados[['idade', 'renda', 'tempo_cliente']]
y = dados['risco']

# Dividir dados (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Treino: {len(X_train)} | Teste: {len(X_test)}")

# Treinar modelo
print("\n🤖 Treinando modelo Random Forest...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print(f"\n" + "=" * 60)
print("RESULTADOS DO MODELO")
print("=" * 60)
print(f"\n✅ Acurácia: {acuracia:.2%}")
print("\n📋 Classificação por Risco:")
print("  0 = BAIXO (verde)")
print("  1 = MÉDIO (amarelo)")
print("  2 = ALTO (vermelho)")

print("\n📈 Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Baixo', 'Médio', 'Alto']))

# Importância das features
importancias = modelo.feature_importances_
features = ['Idade', 'Renda', 'Tempo Cliente']

print("\n🎯 Importância das Features:")
for feat, imp in zip(features, importancias):
    print(f"  {feat}: {imp:.2%}")

# Criar gráficos com design Serasa
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Dashboard de Risco de Crédito - Serasa', fontsize=22, fontweight='bold', y=0.98, color=azul_serasa)

# Gráfico 1: Matriz de Confusão
ax1 = plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', cbar=True, ax=ax1, 
            xticklabels=['Baixo', 'Médio', 'Alto'], yticklabels=['Baixo', 'Médio', 'Alto'],
            cbar_kws={'label': 'Quantidade'}, linewidths=2, linecolor='white')
ax1.set_title('Matriz de Confusão', fontsize=13, fontweight='bold', pad=15, color=azul_serasa)
ax1.set_ylabel('Real', fontsize=11, color=azul_serasa, fontweight='bold')
ax1.set_xlabel('Predito', fontsize=11, color=azul_serasa, fontweight='bold')
ax1.set_facecolor('#ffffff')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Gráfico 2: Importância das Features
ax2 = plt.subplot(2, 2, 2)
cores_feat = [rosa_serasa, azul_serasa, '#FF9800']
ax2.barh(features, importancias, color=cores_feat, alpha=0.85, edgecolor='white', linewidth=2)
ax2.set_title('Importância das Features', fontsize=13, fontweight='bold', pad=15, color=azul_serasa)
ax2.set_xlabel('Importância (%)', fontsize=11, color=azul_serasa, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, color=rosa_claro)
ax2.set_facecolor('#ffffff')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for i, v in enumerate(importancias):
    ax2.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=10, fontweight='bold', color=azul_serasa)

# Gráfico 3: Distribuição de Risco
ax3 = plt.subplot(2, 2, 3)
riscos = dados['risco'].value_counts().sort_index()
cores_risco = ['#4CAF50', '#FFC107', '#F44336']
labels_risco = ['Baixo (0)', 'Médio (1)', 'Alto (2)']
valores_risco = [riscos.get(i, 0) for i in [0, 1, 2]]
bars = ax3.bar(labels_risco, valores_risco, 
               color=cores_risco, alpha=0.85, edgecolor='white', linewidth=2)
ax3.set_title('Distribuição de Risco', fontsize=13, fontweight='bold', pad=15, color=azul_serasa)
ax3.set_ylabel('Quantidade de Clientes', fontsize=11, color=azul_serasa, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, color=rosa_claro)
ax3.set_facecolor('#ffffff')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
for bar, val in zip(bars, valores_risco):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=azul_serasa)

# Gráfico 4: Acurácia
ax4 = plt.subplot(2, 2, 4)
acertos = (y_pred == y_test).sum()
total = len(y_test)
erros = total - acertos
categorias = ['✅ Acertos', '❌ Erros']
valores = [acertos, erros]
cores_acc = [rosa_serasa, '#CCCCCC']
bars = ax4.bar(categorias, valores, color=cores_acc, alpha=0.85, edgecolor='white', linewidth=2)
ax4.set_title(f'Precisão: {acuracia:.1%}', fontsize=13, fontweight='bold', pad=15, color=azul_serasa)
ax4.set_ylabel('Quantidade', fontsize=11, color=azul_serasa, fontweight='bold')
ax4.set_facecolor('#ffffff')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
for bar, val in zip(bars, valores):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold', color=azul_serasa)

plt.tight_layout()
plt.savefig('risco_credito.png', dpi=300, bbox_inches='tight', facecolor='#f5f5f5')
print("\n✅ Gráficos salvos em 'risco_credito.png'!")

plt.show()

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 60)