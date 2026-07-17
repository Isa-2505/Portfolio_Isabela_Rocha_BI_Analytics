import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#f5f5f5'
plt.rcParams['font.family'] = 'sans-serif'

print("=" * 50)
print("ANÁLISE DE CLIENTES - SERASA STYLE")
print("=" * 50)

dados = pd.read_csv('clientes.csv')

print("\nPrimeiros 5 clientes:")
print(dados.head())

print("\n" + "=" * 50)
print("ESTATÍSTICAS DOS DADOS")
print("=" * 50)

print(f"\nTotal de clientes: {len(dados)}")
print(f"Idade média: {dados['idade'].mean():.1f} anos")
print(f"Renda média: R$ {dados['renda'].mean():.2f}")
print(f"Tempo médio de cliente: {dados['tempo_cliente'].mean():.1f} anos")

print("\n" + "=" * 50)
print("CLIENTES POR ESTADO")
print("=" * 50)
print(dados['estado'].value_counts())

fig = plt.figure(figsize=(16, 10))
fig.suptitle('Dashboard de Análise de Clientes - Serasa', fontsize=22, fontweight='bold', y=0.98, color='#1a1a3e')

# Cores Serasa
azul_serasa = '#1a1a3e'  # Azul escuro profissional
rosa_serasa = '#E91E63'  # Rosa vibrante
rosa_claro = '#F48FB1'   # Rosa claro
azul_claro = '#3F51B5'   # Azul claro complementar

ax1 = plt.subplot(2, 2, 1)
plt.hist(dados['idade'], bins=12, color=azul_serasa, alpha=0.85, edgecolor='white', linewidth=2)
plt.title('Distribuição de Idade dos Clientes', fontsize=13, fontweight='bold', pad=15, color=azul_serasa)
plt.xlabel('Idade (anos)', fontsize=11, color=azul_serasa)
plt.ylabel('Quantidade de Clientes', fontsize=11, color=azul_serasa)
plt.grid(axis='y', alpha=0.3, color=rosa_serasa)
ax1.set_facecolor('#ffffff')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = plt.subplot(2, 2, 2)
plt.hist(dados['renda'], bins=12, color=rosa_serasa, alpha=0.85, edgecolor='white', linewidth=2)
plt.title('Distribuição de Renda dos Clientes', fontsize=13, fontweight='bold', pad=15, color=azul_serasa)
plt.xlabel('Renda (R$)', fontsize=11, color=azul_serasa)
plt.ylabel('Quantidade de Clientes', fontsize=11, color=azul_serasa)
plt.grid(axis='y', alpha=0.3, color=azul_claro)
ax2.set_facecolor('#ffffff')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax3 = plt.subplot(2, 2, 3)
estados = dados['estado'].value_counts().sort_values(ascending=True)
cores_bar = [rosa_serasa if i % 2 == 0 else azul_claro for i in range(len(estados))]
plt.barh(estados.index, estados.values, color=rosa_serasa, alpha=0.85, edgecolor='white', linewidth=2)
plt.title('Clientes por Estado', fontsize=13, fontweight='bold', pad=15, color=azul_serasa)
plt.xlabel('Quantidade de Clientes', fontsize=11, color=azul_serasa)
plt.ylabel('Estado', fontsize=11, color=azul_serasa)
plt.grid(axis='x', alpha=0.3, color=rosa_claro)
ax3.set_facecolor('#ffffff')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax4 = plt.subplot(2, 2, 4)
scatter = plt.scatter(dados['idade'], dados['renda'], 
                     c=dados['tempo_cliente'], cmap='RdPu', 
                     s=200, alpha=0.75, edgecolor=azul_serasa, linewidth=1.5)
plt.title('Idade vs Renda (colorido por Tempo de Cliente)', fontsize=13, fontweight='bold', pad=15, color=azul_serasa)
plt.xlabel('Idade (anos)', fontsize=11, color=azul_serasa)
plt.ylabel('Renda (R$)', fontsize=11, color=azul_serasa)
plt.grid(alpha=0.3, color=rosa_claro)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Anos como Cliente', fontsize=10, color=azul_serasa)
cbar.ax.tick_params(colors=azul_serasa)
ax4.set_facecolor('#ffffff')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('analise_clientes.png', dpi=300, bbox_inches='tight', facecolor='#f5f5f5')
print("\n✅ Gráficos salvos com sucesso em 'analise_clientes.png'!")

plt.show()

print("\n" + "=" * 50)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 50)