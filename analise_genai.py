import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# CORES SERASA
AZUL_SERASA = '#1a1a3e'
ROSA_SERASA = '#E91E63'

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#f5f5f5'

print("\n" + "=" * 80)
print("🤖 ANÁLISE INTELIGENTE DE CLIENTES - GENAI")
print("=" * 80)

# Carregar dados
dados = pd.read_csv('dados_clientes.csv')
print(f"\n✅ {len(dados)} clientes carregados")

# Segmentação RFM
dados['score_r'] = pd.qcut(dados['recencia_dias'], 4, labels=[4,3,2,1], duplicates='drop')
dados['score_f'] = pd.qcut(dados['frequencia'].rank(method='first'), 4, labels=[1,2,3,4], duplicates='drop')
dados['score_m'] = pd.qcut(dados['compras_totais'], 4, labels=[1,2,3,4], duplicates='drop')

def segmentar(row):
    score = (int(row['score_r']) + int(row['score_f']) + int(row['score_m'])) / 3
    if score >= 3.5: return 'VIP'
    elif score >= 2.5: return 'Premium'
    elif score >= 1.5: return 'Regular'
    else: return 'Em Risco'

dados['segmento'] = dados.apply(segmentar, axis=1)

# Insights GenAI
print("\n📊 SEGMENTAÇÃO DE CLIENTES:")
for seg in ['VIP', 'Premium', 'Regular', 'Em Risco']:
    count = len(dados[dados['segmento'] == seg])
    total = dados[dados['segmento'] == seg]['compras_totais'].sum()
    nps = dados[dados['segmento'] == seg]['nps'].mean()
    print(f"  {seg:12} | {count:2} clientes | R$ {total:>10,.0f} | NPS {nps:.1f}")

# Recomendações
print("\n💡 RECOMENDAÇÕES AUTOMÁTICAS (GENAI):")
alto_nps = len(dados[dados['nps'] >= 9])
print(f"  ✨ {alto_nps} clientes com NPS excelente - Ativar indicações")
inativos = len(dados[dados['recencia_dias'] > 20])
print(f"  🔔 {inativos} clientes inativos - Campanha de reativação")
categoria_top = dados['categoria'].value_counts().idxmax()
print(f"  📦 Categoria '{categoria_top}' lidera - Investir em estoque")

# Gráficos
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Dashboard Inteligente - Serasa Experian', fontsize=22, fontweight='bold', y=0.98, color=AZUL_SERASA)

# G1: Segmentação
ax1 = plt.subplot(2, 2, 1)
seg_counts = dados['segmento'].value_counts()
cores_seg = {'VIP': ROSA_SERASA, 'Premium': '#FF9800', 'Regular': '#4CAF50', 'Em Risco': '#F44336'}
cores = [cores_seg.get(s, '#CCC') for s in seg_counts.index]
ax1.bar(seg_counts.index, seg_counts.values, color=cores, alpha=0.85, edgecolor='white', linewidth=2)
ax1.set_title('Clientes por Segmento', fontsize=13, fontweight='bold', pad=15, color=AZUL_SERASA)
ax1.set_facecolor('#ffffff')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# G2: Faturamento
ax2 = plt.subplot(2, 2, 2)
faturamento = dados.groupby('segmento')['compras_totais'].sum().sort_values(ascending=False)
ax2.barh(faturamento.index, faturamento.values, color=ROSA_SERASA, alpha=0.85, edgecolor='white', linewidth=2)
ax2.set_title('Faturamento por Segmento', fontsize=13, fontweight='bold', pad=15, color=AZUL_SERASA)
ax2.set_facecolor('#ffffff')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# G3: Scatter
ax3 = plt.subplot(2, 2, 3)
for seg in dados['segmento'].unique():
    dados_seg = dados[dados['segmento'] == seg]
    cor = cores_seg.get(seg, '#CCC')
    ax3.scatter(dados_seg['frequencia'], dados_seg['compras_totais'], label=seg, s=150, alpha=0.7, edgecolor='white', linewidth=1.5, color=cor)
ax3.set_title('Frequência vs Compras', fontsize=13, fontweight='bold', pad=15, color=AZUL_SERASA)
ax3.set_xlabel('Frequência', fontsize=11, color=AZUL_SERASA)
ax3.set_ylabel('Compras (R$)', fontsize=11, color=AZUL_SERASA)
ax3.legend(loc='upper left')
ax3.set_facecolor('#ffffff')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# G4: NPS
ax4 = plt.subplot(2, 2, 4)
nps_seg = dados.groupby('segmento')['nps'].mean().sort_values(ascending=False)
ax4.bar(nps_seg.index, nps_seg.values, color=AZUL_SERASA, alpha=0.85, edgecolor='white', linewidth=2)
ax4.set_title('NPS Médio por Segmento', fontsize=13, fontweight='bold', pad=15, color=AZUL_SERASA)
ax4.set_ylim(0, 10)
ax4.set_facecolor('#ffffff')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('analise_genai.png', dpi=300, bbox_inches='tight', facecolor='#f5f5f5')
print("\n✅ Gráficos salvos em 'analise_genai.png'!")
plt.show()

print("\n" + "=" * 80)
print("✅ ANÁLISE CONCLUÍDA!")
print("=" * 80 + "\n")