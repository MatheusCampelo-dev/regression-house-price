import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import base64, os, json, warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs/figures', exist_ok=True)
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='whitegrid')

def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def save(name):
    plt.savefig(f'outputs/figures/{name}.png', bbox_inches='tight')
    plt.close()
    print(f'  {name}.png OK')

# ═══════════════════════════════════════════════════════════════
# DADOS BASE
# ═══════════════════════════════════════════════════════════════
train_raw = pd.read_csv('data/raw/train.csv')
test_raw  = pd.read_csv('data/raw/test.csv')

# AV1 usa fillna mean
df_av1 = train_raw.copy()
df_av1.fillna(df_av1.mean(numeric_only=True), inplace=True)
df_av1.drop_duplicates(inplace=True)
df_av1['Foi_Reformada'] = np.where(df_av1['YearRemodAdd'] > df_av1['YearBuilt'], 'Sim', 'Não')

# AV2 mesma base
df_av2 = df_av1.copy()
df_av2['LogSalePrice'] = np.log(df_av2['SalePrice'])
df_av2['Era'] = pd.cut(df_av2['YearBuilt'], bins=[1800,1960,1980,2000,2011],
                        labels=['< 1960','1960-1980','1980-2000','> 2000'])

print("=== GERANDO FIGURAS DA ANÁLISE EXPLORATÓRIA ===")

# FIG AV1-1: Heatmap correlação top 10
colunas_num = df_av1.select_dtypes(include=np.number).columns.tolist()
correlacao = df_av1[colunas_num].corr()
top10 = correlacao['SalePrice'].abs().sort_values(ascending=False).head(11).index
plt.figure(figsize=(10, 8))
sns.heatmap(df_av1[top10].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação – Top 10 Variáveis', fontsize=14, fontweight='bold')
plt.tight_layout(); save('av1_correlacao_heatmap')

# FIG AV1-2: GrLivArea vs SalePrice
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_av1, x='GrLivArea', y='SalePrice', color='steelblue', alpha=0.6)
sns.regplot(data=df_av1, x='GrLivArea', y='SalePrice', scatter=False, color='red')
plt.title('Relação entre GrLivArea e SalePrice', fontsize=15)
plt.xlabel('Área do Living Room (sq ft)'); plt.ylabel('Preço de Venda (USD)')
sns.despine(); plt.tight_layout(); save('av1_grlivarea_vs_saleprice')

# FIG AV1-3: YearBuilt vs SalePrice linha
preco_por_ano = df_av1.groupby('YearBuilt')['SalePrice'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=preco_por_ano, x='YearBuilt', y='SalePrice', color='darkblue', linewidth=2.5)
plt.title('Evolução do Preço Médio pelo Ano de Construção', fontsize=16)
plt.xlabel('Ano de Construção'); plt.ylabel('Preço Médio de Venda ($)')
plt.grid(axis='y', linestyle='--', alpha=0.6); sns.despine(); plt.tight_layout(); save('av1_yearbuilt_vs_price')

# FIG AV1-4: Reforma boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_av1, x='Foi_Reformada', y='SalePrice', palette=['lightcoral','lightblue'])
plt.title('Impacto da Reforma no Preço de Venda', fontsize=15)
plt.xlabel('A casa foi reformada?'); plt.ylabel('Preço de Venda ($)')
sns.despine(); plt.tight_layout(); save('av1_reforma_boxplot')

# FIG AV1-5: Bairros boxplot
ordem_bairros = df_av1.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).index
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_av1, x='Neighborhood', y='SalePrice', order=ordem_bairros, palette='coolwarm')
plt.title('Preço de Venda por Bairro (Do mais caro ao mais barato)', fontsize=14)
plt.xlabel('Bairro (Neighborhood)'); plt.ylabel('Preço de Venda ($)')
plt.xticks(rotation=45, ha='right'); sns.despine(); plt.tight_layout(); save('av1_bairros_boxplot')

print("\n=== GERANDO FIGURAS DA AV2 ===")

# FIG AV2-1: Normalidade – histograma + QQ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_av2['SalePrice'], kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Distribuição do SalePrice (original)', fontsize=13)
axes[0].set_xlabel('Preço de Venda ($)')
stats.probplot(df_av2['SalePrice'], dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot – SalePrice (original)', fontsize=13)
plt.tight_layout(); save('av2_normalidade_original')

# FIG AV2-2: Log normalidade
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_av2['LogSalePrice'], kde=True, ax=axes[0], color='seagreen')
axes[0].set_title('Distribuição do log(SalePrice)', fontsize=13)
axes[0].set_xlabel('log(Preço de Venda)')
stats.probplot(df_av2['LogSalePrice'], dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot – log(SalePrice)', fontsize=13)
plt.tight_layout(); save('av2_normalidade_log')

# FIG AV2-3: Reforma boxplot + violin
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df_av2, x='Foi_Reformada', y='SalePrice', palette=['lightcoral','lightblue'], ax=axes[0])
axes[0].set_title('Distribuição de Preços por Reforma', fontsize=12)
axes[0].set_xlabel('Foi Reformada?'); axes[0].set_ylabel('Preço de Venda ($)')
sns.violinplot(data=df_av2, x='Foi_Reformada', y='SalePrice', palette=['lightcoral','lightblue'], ax=axes[1], inner='quartile')
axes[1].set_title('Violin Plot: Distribuição por Grupo', fontsize=12)
axes[1].set_xlabel('Foi Reformada?'); axes[1].set_ylabel('Preço de Venda ($)')
plt.tight_layout(); save('av2_reforma_teste_t')

# FIG AV2-4: Paradoxo de Simpson
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
mediana_era = df_av2.groupby(['Foi_Reformada','Era'], observed=True)['SalePrice'].median().unstack()
mediana_era.T.plot(kind='bar', ax=axes[0], colormap='coolwarm', rot=15)
axes[0].set_title('Mediana por Era de Construção\n(dentro de cada era, reforma agrega valor)', fontsize=11)
axes[0].set_xlabel('Era de Construção'); axes[0].set_ylabel('Mediana SalePrice ($)')
axes[0].legend(title='Foi Reformada?')
contagem = df_av2.groupby(['Foi_Reformada','Era'], observed=True).size().unstack().fillna(0)
contagem.T.plot(kind='bar', ax=axes[1], colormap='Set2', rot=15)
axes[1].set_title('Composição dos Grupos por Era\n(nova construção domina "Não reformada")', fontsize=11)
axes[1].set_xlabel('Era de Construção'); axes[1].set_ylabel('Número de Imóveis')
axes[1].legend(title='Foi Reformada?')
plt.tight_layout(); save('av2_simpson_paradox')

# FIG AV2-5: Top5 vs Bottom5 bairros
media_bairro = df_av2.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
top5 = media_bairro.head(5).index
bot5 = media_bairro.tail(5).index
bairros_sel = list(top5) + list(bot5)
df_sel = df_av2[df_av2['Neighborhood'].isin(bairros_sel)]
ordem = media_bairro[bairros_sel].sort_values(ascending=False).index
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_sel, x='Neighborhood', y='SalePrice', order=ordem, palette='RdYlGn')
plt.title('Top 5 Bairros Mais Caros vs. 5 Mais Baratos', fontsize=14)
plt.xlabel('Bairro'); plt.ylabel('Preço de Venda ($)')
plt.xticks(rotation=30, ha='right'); sns.despine(); plt.tight_layout(); save('av2_top5_bottom5_bairros')

# FIG AV2-6: Pearson – GrLivArea
r_pearson, p_pearson = stats.pearsonr(df_av2['GrLivArea'], df_av2['SalePrice'])
r_log, _ = stats.pearsonr(df_av2['GrLivArea'], df_av2['LogSalePrice'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.regplot(data=df_av2, x='GrLivArea', y='SalePrice', ax=axes[0],
            scatter_kws={'alpha':0.4,'color':'steelblue'}, line_kws={'color':'red'})
axes[0].set_title(f'GrLivArea vs. SalePrice\n(Pearson r = {r_pearson:.4f})', fontsize=12)
sns.regplot(data=df_av2, x='GrLivArea', y='LogSalePrice', ax=axes[1],
            scatter_kws={'alpha':0.4,'color':'seagreen'}, line_kws={'color':'red'})
axes[1].set_title(f'GrLivArea vs. log(SalePrice)\n(Pearson r = {r_log:.4f})', fontsize=12)
plt.tight_layout(); save('av2_correlacao_pearson')

# FIG AV2-7: Spearman – YearBuilt
r_sp_year, p_sp_year = stats.spearmanr(df_av2['YearBuilt'], df_av2['SalePrice'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
preco_ano = df_av2.groupby('YearBuilt')['SalePrice'].mean().reset_index()
sns.lineplot(data=preco_ano, x='YearBuilt', y='SalePrice', color='darkblue', linewidth=2, ax=axes[0])
axes[0].set_title(f'Preço Médio por Ano de Construção\n(Spearman ρ = {r_sp_year:.4f})', fontsize=12)
axes[0].set_xlabel('Ano de Construção'); axes[0].set_ylabel('Preço Médio ($)')
sns.regplot(data=df_av2, x='YearBuilt', y='SalePrice', ax=axes[1],
            scatter_kws={'alpha':0.3,'color':'slateblue'}, line_kws={'color':'red'})
axes[1].set_title('Dispersão: YearBuilt vs. SalePrice', fontsize=12)
plt.tight_layout(); save('av2_correlacao_spearman_year')

# FIG AV2-8: Intervalo de Confiança
from scipy.stats import t as t_dist
def intervalo_confianca(serie, c=0.95):
    n=len(serie); m=serie.mean(); se=stats.sem(serie)
    mg=se*t_dist.ppf((1+c)/2,df=n-1)
    return m, m-mg, m+mg

grupos_ic = []
for grupo in df_av2['Foi_Reformada'].unique():
    dados = df_av2[df_av2['Foi_Reformada']==grupo]['SalePrice']
    m, low, high = intervalo_confianca(dados)
    grupos_ic.append({'Grupo': f'Reformada = {grupo}', 'Media': m, 'Low': low, 'High': high})
ic_df = pd.DataFrame(grupos_ic)
plt.figure(figsize=(8, 4))
plt.errorbar(ic_df['Grupo'], ic_df['Media'],
             yerr=[ic_df['Media']-ic_df['Low'], ic_df['High']-ic_df['Media']],
             fmt='o', capsize=10, color='steelblue', ecolor='gray', markersize=10)
plt.title('Intervalo de Confiança 95% – Preço por Grupo de Reforma', fontsize=13)
plt.ylabel('Preço de Venda ($)'); plt.grid(axis='y', linestyle='--', alpha=0.5)
sns.despine(); plt.tight_layout(); save('av2_intervalo_confianca')

print("\n=== VERIFICANDO FIGURAS DA MODELAGEM ===")
mod_figs = ['novas_features_vs_price','distribuicao_saleprice','comparacao_modelos',
            'impacto_log_transform','impacto_scaling','analise_melhor_modelo',
            'feature_importance','modelo_simplificado','correlacao_negocio',
            'cross_validation','dashboard_final']
for f in mod_figs:
    exists = os.path.exists(f'outputs/figures/{f}.png')
    print(f"  {f}: {'OK' if exists else 'MISSING'}")

# ═══════════════════════════════════════════════════════════════
# COLETA DE MÉTRICAS
# ═══════════════════════════════════════════════════════════════
print("\n=== CALCULANDO MÉTRICAS ===")

# AV2 métricas
amostra = df_av2['SalePrice'].sample(1000, random_state=42)
stat_sw, p_sw = stats.shapiro(amostra)
amostra_log = df_av2['LogSalePrice'].sample(1000, random_state=42)
stat_sw_log, p_sw_log = stats.shapiro(amostra_log)
stat_ks, p_ks = stats.kstest(df_av2['SalePrice'],'norm',args=(df_av2['SalePrice'].mean(),df_av2['SalePrice'].std()))
grupo_sim = df_av2[df_av2['Foi_Reformada']=='Sim']['SalePrice']
grupo_nao = df_av2[df_av2['Foi_Reformada']=='Não']['SalePrice']
stat_levene, p_levene = stats.levene(grupo_sim, grupo_nao)
t_stat, p_valor_t = stats.ttest_ind(grupo_sim, grupo_nao, equal_var=False)
stat_mw, p_mw = stats.mannwhitneyu(grupo_sim, grupo_nao, alternative='two-sided')
grupos_bairros = [df_av2[df_av2['Neighborhood']==b]['SalePrice'] for b in df_av2['Neighborhood'].unique()]
f_stat, p_anova = stats.f_oneway(*grupos_bairros)
tukey = pairwise_tukeyhsd(endog=df_av2['SalePrice'], groups=df_av2['Neighborhood'], alpha=0.05)
tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
tukey_df['reject'] = tukey_df['reject'].astype(bool)
n_sig_bairros = len(tukey_df[tukey_df['reject']==True])
r_pearson, p_pearson = stats.pearsonr(df_av2['GrLivArea'], df_av2['SalePrice'])
r_spearman, p_spearman = stats.spearmanr(df_av2['GrLivArea'], df_av2['SalePrice'])
r_sp_year, p_sp_year = stats.spearmanr(df_av2['YearBuilt'], df_av2['SalePrice'])
r_pe_year, _ = stats.pearsonr(df_av2['YearBuilt'], df_av2['SalePrice'])
n_ic = len(df_av2); m_ic, ic_low_g, ic_high_g = intervalo_confianca(df_av2['SalePrice'])
# IC por reforma
ic_sim = intervalo_confianca(grupo_sim); ic_nao = intervalo_confianca(grupo_nao)
# IC top5 bairros
ic_top5 = {}
for b in top5:
    dados = df_av2[df_av2['Neighborhood']==b]['SalePrice']
    ic_top5[b] = intervalo_confianca(dados)

print(f"Shapiro SalePrice: {p_sw:.2e} | Log: {p_sw_log:.2e}")
print(f"Teste t: {p_valor_t:.4f} | MW: {p_mw:.4f}")
print(f"ANOVA: {p_anova:.2e} | Pares sig: {n_sig_bairros}")
print(f"Pearson r={r_pearson:.4f} | Spearman year rho={r_sp_year:.4f}")

# Modelagem
with open('outputs/metricas.json') as f:
    met = json.load(f)
print(f"Modelagem: melhor={met['melhor']} RMSE={met['modelos'][met['melhor']]['RMSE']}")

# ═══════════════════════════════════════════════════════════════
# ENCODAR TODAS AS FIGURAS
# ═══════════════════════════════════════════════════════════════
print("\n=== ENCODANDO FIGURAS ===")
all_figs = {
    'av1_correlacao_heatmap': b64('outputs/figures/av1_correlacao_heatmap.png'),
    'av1_grlivarea': b64('outputs/figures/av1_grlivarea_vs_saleprice.png'),
    'av1_yearbuilt': b64('outputs/figures/av1_yearbuilt_vs_price.png'),
    'av1_reforma': b64('outputs/figures/av1_reforma_boxplot.png'),
    'av1_bairros': b64('outputs/figures/av1_bairros_boxplot.png'),
    'av2_norm_orig': b64('outputs/figures/av2_normalidade_original.png'),
    'av2_norm_log': b64('outputs/figures/av2_normalidade_log.png'),
    'av2_reforma_t': b64('outputs/figures/av2_reforma_teste_t.png'),
    'av2_simpson': b64('outputs/figures/av2_simpson_paradox.png'),
    'av2_bairros': b64('outputs/figures/av2_top5_bottom5_bairros.png'),
    'av2_pearson': b64('outputs/figures/av2_correlacao_pearson.png'),
    'av2_spearman': b64('outputs/figures/av2_correlacao_spearman_year.png'),
    'av2_ic': b64('outputs/figures/av2_intervalo_confianca.png'),
    'mod_novas_features': b64('outputs/figures/novas_features_vs_price.png'),
    'mod_dist_saleprice': b64('outputs/figures/distribuicao_saleprice.png'),
    'mod_comparacao': b64('outputs/figures/comparacao_modelos.png'),
    'mod_log_impact': b64('outputs/figures/impacto_log_transform.png'),
    'mod_scaling': b64('outputs/figures/impacto_scaling.png'),
    'mod_melhor': b64('outputs/figures/analise_melhor_modelo.png'),
    'mod_importance': b64('outputs/figures/feature_importance.png'),
    'mod_simplificado': b64('outputs/figures/modelo_simplificado.png'),
    'mod_negocio': b64('outputs/figures/correlacao_negocio.png'),
    'mod_cv': b64('outputs/figures/cross_validation.png'),
    'mod_dashboard': b64('outputs/figures/dashboard_final.png'),
}
print(f"Total de figuras: {len(all_figs)}")

def img(key, alt=''):
    return f'<img src="data:image/png;base64,{all_figs[key]}" alt="{alt}" style="max-width:90%;border:1px solid #ddd;">'

def fig_block(key, num, caption, alt=''):
    return f'''<figure style="text-align:center;margin:1.5em 0;">
{img(key, alt)}
<figcaption style="font-size:10pt;text-align:center;margin-top:0.4em;color:#333;"><strong>Figura {num}</strong> – {caption}. Fonte: elaborado pelos autores (2025).</figcaption>
</figure>'''

def code_block(code):
    import html
    return f'<pre style="font-family:Courier New,monospace;font-size:9.5pt;background:#f8f8f8;border:1px solid #ddd;padding:12px;white-space:pre-wrap;margin:0.8em 0;">{html.escape(code)}</pre>'

# ═══════════════════════════════════════════════════════════════
# STRINGS DE CÓDIGO DOS NOTEBOOKS
# ═══════════════════════════════════════════════════════════════

CODE = {}

CODE['av1_imports'] = '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df_csv2 = pd.read_csv('../data/raw/train.csv')
df_csv2.fillna(df_csv2.mean(numeric_only=True), inplace=True)
df_csv2.drop_duplicates(inplace=True)'''

CODE['av1_heatmap'] = '''colunas_numericas = df_csv2.select_dtypes(include=[np.number]).columns.tolist()
correlacao = df_csv2[colunas_numericas].corr()
top_10_vars = correlacao['SalePrice'].abs().sort_values(ascending=False).head(11).index
matriz_top10 = df_csv2[top_10_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_top10, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação – Top 10 Variáveis')
print("Ranking das 10 variáveis mais correlacionadas com SalePrice:")
print(correlacao['SalePrice'].abs().sort_values(ascending=False).head(10))'''

CODE['av1_scatter'] = '''plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_csv2, x='GrLivArea', y='SalePrice', color='blue', alpha=0.6)
sns.regplot(data=df_csv2, x='GrLivArea', y='SalePrice', scatter=False, color='red')
plt.title('Relação entre GrLivArea e SalePrice', fontsize=15)
plt.xlabel('Área do Living Room (sq ft)')
plt.ylabel('Preço de Venda (USD)')
sns.despine()
plt.show()'''

CODE['av1_yearbuilt'] = '''preco_por_ano = df_csv2.groupby('YearBuilt')['SalePrice'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=preco_por_ano, x='YearBuilt', y='SalePrice', color='darkblue', linewidth=2.5)
plt.title('Evolução do Preço Médio pelo Ano de Construção', fontsize=16)
plt.xlabel('Ano de Construção')
plt.ylabel('Preço Médio de Venda ($)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
sns.despine()
plt.show()'''

CODE['av1_reforma'] = '''df_csv2['Foi_Reformada'] = np.where(df_csv2['YearRemodAdd'] > df_csv2['YearBuilt'], 'Sim', 'Não')

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_csv2, x='Foi_Reformada', y='SalePrice', palette=['lightcoral', 'lightblue'])
plt.title('Impacto da Reforma no Preço de Venda', fontsize=15)
plt.xlabel('A casa foi reformada?')
plt.ylabel('Preço de Venda ($)')
sns.despine()
plt.show()

medianas = df_csv2.groupby('Foi_Reformada')['SalePrice'].median()
print(f"Casas ORIGINAIS (Não reformadas): ${medianas['Não']:,.2f}")
print(f"Casas REFORMADAS: ${medianas['Sim']:,.2f}")'''

CODE['av1_bairros'] = '''ordem_bairros = df_csv2.groupby('Neighborhood')['SalePrice'].median(\\
    ).sort_values(ascending=False).index

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_csv2, x='Neighborhood', y='SalePrice',
            order=ordem_bairros, palette='coolwarm')
plt.title('Preço de Venda por Bairro (Do mais caro ao mais barato)', fontsize=14)
plt.xlabel('Bairro (Neighborhood)')
plt.ylabel('Preço de Venda ($)')
plt.xticks(rotation=45, ha='right')
sns.despine()
plt.show()

grupos_bairros = [df_csv2[df_csv2['Neighborhood']==b]['SalePrice']
                  for b in df_csv2['Neighborhood'].unique()]
f_stat, p_valor = stats.f_oneway(*grupos_bairros)
print(f"ANOVA p-valor: {p_valor:.2e}")
if p_valor < 0.05:
    print("Veredito: SIM! Diferença estatisticamente relevante.")'''

CODE['av2_imports'] = '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../data/raw/train.csv')
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)
df['Foi_Reformada'] = np.where(df['YearRemodAdd'] > df['YearBuilt'], 'Sim', 'Não')
print(f'Dataset: {df.shape[0]} imóveis | {df.shape[1]} variáveis')'''

CODE['av2_normalidade'] = '''amostra = df['SalePrice'].sample(1000, random_state=42)
stat_sw, p_sw = stats.shapiro(amostra)
stat_ks, p_ks = stats.kstest(df['SalePrice'], 'norm',
                              args=(df['SalePrice'].mean(), df['SalePrice'].std()))
print('=== TESTE DE NORMALIDADE – SalePrice ===')
print(f'Shapiro-Wilk   | p-valor: {p_sw:.6f}')
print(f'Kolmogorov-Smirnov | p-valor: {p_ks:.6f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['SalePrice'], kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Distribuição do SalePrice (original)')
stats.probplot(df['SalePrice'], dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot – SalePrice (original)')
plt.tight_layout(); plt.show()'''

CODE['av2_log'] = '''df['LogSalePrice'] = np.log(df['SalePrice'])
amostra_log = df['LogSalePrice'].sample(1000, random_state=42)
stat_sw_log, p_sw_log = stats.shapiro(amostra_log)

print(f'Assimetria original: {df["SalePrice"].skew():.4f}')
print(f'Assimetria log:      {df["LogSalePrice"].skew():.4f}')
print(f'Shapiro-Wilk log: p = {p_sw_log:.6f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['LogSalePrice'], kde=True, ax=axes[0], color='seagreen')
stats.probplot(df['LogSalePrice'], dist='norm', plot=axes[1])
plt.tight_layout(); plt.show()'''

CODE['av2_teste_t'] = '''grupo_sim = df[df['Foi_Reformada'] == 'Sim']['SalePrice']
grupo_nao = df[df['Foi_Reformada'] == 'Não']['SalePrice']

# Teste de Levene (homogeneidade de variâncias)
stat_levene, p_levene = stats.levene(grupo_sim, grupo_nao)
print(f'Levene: p = {p_levene:.6f}')

# Teste t de Welch
t_stat, p_valor_t = stats.ttest_ind(grupo_sim, grupo_nao, equal_var=False)
print(f'Teste t Welch: t = {t_stat:.4f} | p = {p_valor_t:.6f}')

# Mann-Whitney U (não-paramétrico)
stat_mw, p_mw = stats.mannwhitneyu(grupo_sim, grupo_nao, alternative='two-sided')
print(f'Mann-Whitney U: p = {p_mw:.6f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x='Foi_Reformada', y='SalePrice',
            palette=['lightcoral','lightblue'], ax=axes[0])
sns.violinplot(data=df, x='Foi_Reformada', y='SalePrice',
               palette=['lightcoral','lightblue'], ax=axes[1], inner='quartile')
plt.tight_layout(); plt.show()'''

CODE['av2_simpson'] = '''df['Era'] = pd.cut(df['YearBuilt'],
                   bins=[1800, 1960, 1980, 2000, 2011],
                   labels=['< 1960', '1960-1980', '1980-2000', '> 2000'])

mediana_era = df.groupby(['Foi_Reformada', 'Era'], observed=True)[\\
    'SalePrice'].median().unstack()
print(mediana_era.to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
mediana_era.T.plot(kind='bar', ax=axes[0], colormap='coolwarm', rot=15)
contagem = df.groupby(['Foi_Reformada','Era'], observed=True).size().unstack().fillna(0)
contagem.T.plot(kind='bar', ax=axes[1], colormap='Set2', rot=15)
plt.tight_layout(); plt.show()'''

CODE['av2_anova'] = '''grupos_bairros = [df[df['Neighborhood'] == b]['SalePrice']
                  for b in df['Neighborhood'].unique()]
f_stat, p_anova = stats.f_oneway(*grupos_bairros)
print(f'ANOVA: F = {f_stat:.4f} | p = {p_anova:.2e}')

tukey = pairwise_tukeyhsd(endog=df['SalePrice'],
                           groups=df['Neighborhood'], alpha=0.05)
tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                         columns=tukey._results_table.data[0])
tukey_df['reject'] = tukey_df['reject'].astype(bool)
significativos = tukey_df[tukey_df['reject'] == True]
print(f'Pares com diferença significativa: {len(significativos)}')
print(significativos.nlargest(10,'meandiff')[['group1','group2','meandiff','p-adj']])'''

CODE['av2_pearson'] = '''r_pearson, p_pearson = stats.pearsonr(df['GrLivArea'], df['SalePrice'])
r_spearman, p_spearman = stats.spearmanr(df['GrLivArea'], df['SalePrice'])
print(f'Pearson  | r = {r_pearson:.4f} | p = {p_pearson:.2e}')
print(f'Spearman | ρ = {r_spearman:.4f} | p = {p_spearman:.2e}')

# IC 95% para r de Pearson (transformação de Fisher)
n = len(df)
z = np.arctanh(r_pearson)
se = 1 / np.sqrt(n - 3)
z_crit = stats.norm.ppf(0.975)
ic_r_low  = np.tanh(z - z_crit * se)
ic_r_high = np.tanh(z + z_crit * se)
print(f'IC 95% para r: [{ic_r_low:.4f}, {ic_r_high:.4f}]')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.regplot(data=df, x='GrLivArea', y='SalePrice', ax=axes[0],
            scatter_kws={'alpha':0.4,'color':'steelblue'}, line_kws={'color':'red'})
axes[0].set_title(f'GrLivArea vs. SalePrice (r = {r_pearson:.4f})')
sns.regplot(data=df, x='GrLivArea', y='LogSalePrice', ax=axes[1],
            scatter_kws={'alpha':0.4,'color':'seagreen'}, line_kws={'color':'red'})
plt.tight_layout(); plt.show()'''

CODE['av2_spearman'] = '''r_sp_year, p_sp_year = stats.spearmanr(df['YearBuilt'], df['SalePrice'])
r_pe_year, p_pe_year = stats.pearsonr(df['YearBuilt'], df['SalePrice'])
print(f'Pearson  | r = {r_pe_year:.4f} | p = {p_pe_year:.2e}')
print(f'Spearman | ρ = {r_sp_year:.4f} | p = {p_sp_year:.2e}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
preco_por_ano = df.groupby('YearBuilt')['SalePrice'].mean().reset_index()
sns.lineplot(data=preco_por_ano, x='YearBuilt', y='SalePrice',
             color='darkblue', linewidth=2, ax=axes[0])
axes[0].set_title(f'Preço Médio por Ano (Spearman ρ = {r_sp_year:.4f})')
sns.regplot(data=df, x='YearBuilt', y='SalePrice', ax=axes[1],
            scatter_kws={'alpha':0.3,'color':'slateblue'}, line_kws={'color':'red'})
axes[1].set_title('Dispersão: YearBuilt vs. SalePrice')
plt.tight_layout(); plt.show()'''

CODE['av2_ic'] = '''from scipy.stats import t as t_dist

def intervalo_confianca(serie, confianca=0.95):
    n = len(serie)
    media = serie.mean()
    se = stats.sem(serie)
    margem = se * t_dist.ppf((1 + confianca) / 2, df=n - 1)
    return media, media - margem, media + margem

media_geral, ic_low_g, ic_high_g = intervalo_confianca(df['SalePrice'])
print(f'IC 95% geral: [{ic_low_g:,.2f}, {ic_high_g:,.2f}]')

for grupo in ['Sim', 'Não']:
    dados = df[df['Foi_Reformada'] == grupo]['SalePrice']
    m, low, high = intervalo_confianca(dados)
    print(f'Reformada={grupo}: [{low:,.2f}, {high:,.2f}]')

plt.figure(figsize=(8, 4))
grupos_ic = [{'Grupo':f'Reformada = {g}',
              **dict(zip(['Media','Low','High'], intervalo_confianca(df[df['Foi_Reformada']==g]['SalePrice'])))}
             for g in df['Foi_Reformada'].unique()]
ic_df = pd.DataFrame(grupos_ic)
plt.errorbar(ic_df['Grupo'], ic_df['Media'],
             yerr=[ic_df['Media']-ic_df['Low'], ic_df['High']-ic_df['Media']],
             fmt='o', capsize=10, color='steelblue', ecolor='gray', markersize=10)
plt.title('IC 95% – Preço por Grupo de Reforma')
plt.ylabel('Preço de Venda ($)')
plt.tight_layout(); plt.show()'''

CODE['mod_missing'] = '''NONE_COLS = [
    'Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
    'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
    'PoolQC','Fence','MiscFeature','MasVnrType'
]

def tratar_nulos(df):
    df = df.copy()
    for col in NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

train = tratar_nulos(train)
test  = tratar_nulos(test)
print(f'Nulos restantes: {train.isnull().sum().sum()}')'''

CODE['mod_features'] = '''def criar_features(df):
    df = df.copy()
    df['TotalSF']         = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBaths']      = (df['FullBath'] + df['BsmtFullBath'] +
                             0.5 * (df['HalfBath'] + df['BsmtHalfBath']))
    df['HouseAge']        = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
    df['Foi_Reformada']   = (df['YearRemodAdd'] > df['YearBuilt']).astype(int)
    df['QualArea']        = df['OverallQual'] * df['GrLivArea']
    df['TotalPorchSF']    = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                             df['3SsnPorch'] + df['ScreenPorch'])
    df['HasPool']         = (df['PoolArea'] > 0).astype(int)
    df['HasFireplace']    = (df['Fireplaces'] > 0).astype(int)
    df['HasGarage']       = (df['GarageArea'] > 0).astype(int)
    return df

train = criar_features(train)
test  = criar_features(test)'''

CODE['mod_encoding'] = '''QUALITY_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ORDINAL_COLS = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
                'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']

for col in ORDINAL_COLS:
    for df in [train, test]:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP).fillna(0).astype(int)

y = train['SalePrice'].copy()
n_train = len(train)
combined = pd.concat([train.drop('SalePrice', axis=1), test], axis=0, ignore_index=True)
combined = pd.get_dummies(combined, drop_first=True)
X = combined.iloc[:n_train].drop('Id', axis=1, errors='ignore')
print(f'Total de features após encoding: {X.shape[1]}')'''

CODE['mod_log'] = '''y_log = np.log1p(y)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y, bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Distribuição Original do SalePrice')
axes[1].hist(y_log, bins=50, color='darkorange', edgecolor='white')
axes[1].set_title('Distribuição após log1p(SalePrice)')
plt.tight_layout(); plt.show()

from scipy import stats as sp_stats
_, p_orig = sp_stats.normaltest(y)
_, p_log  = sp_stats.normaltest(y_log)
print(f'Normalidade – Original: p={p_orig:.2e} | Log: p={p_log:.2e}')'''

CODE['mod_modelos'] = '''from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

modelos = {
    'Regressão Linear': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'Ridge':            Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))]),
    'Lasso':            Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.001, max_iter=10000))]),
    'Random Forest':    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    'XGBoost':          XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42,
                                     subsample=0.8, colsample_bytree=0.8, verbosity=0),
}

resultados = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred = np.expm1(modelo.predict(X_val))
    real = np.expm1(y_val)
    rmse  = np.sqrt(mean_squared_error(real, pred))
    r2    = r2_score(real, pred)
    rmsle = np.sqrt(mean_squared_error(y_val, modelo.predict(X_val)))
    resultados[nome] = {'RMSE': rmse, 'R²': r2, 'RMSLE': rmsle}
    print(f'{nome:<22} | RMSE: ${rmse:>10,.0f} | R²: {r2:.4f}')'''

CODE['mod_log_compare'] = '''# XGBoost sem log para comparação
X_tr_raw, X_v_raw, y_tr_raw, y_v_raw = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_raw = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                        random_state=42, subsample=0.8, colsample_bytree=0.8, verbosity=0)
xgb_raw.fit(X_tr_raw, y_tr_raw)
rmse_raw = np.sqrt(mean_squared_error(y_v_raw, xgb_raw.predict(X_v_raw)))

rmse_log = np.sqrt(mean_squared_error(
    np.expm1(y_val), np.expm1(modelos['XGBoost'].predict(X_val))))

print(f'Sem log: RMSE = ${rmse_raw:,.0f}')
print(f'Com log: RMSE = ${rmse_log:,.0f}')
print(f'Melhora: {(rmse_raw-rmse_log)/rmse_raw*100:.1f}%')'''

CODE['mod_scaling'] = '''for nome_modelo, ModelClass, kwargs in [
    ('Ridge', Ridge, {'alpha': 10.0}),
    ('Lasso', Lasso, {'alpha': 0.001, 'max_iter': 10000}),
]:
    m = ModelClass(**kwargs)
    m.fit(X_train, y_train)
    rmse_sem = np.sqrt(mean_squared_error(
        np.expm1(y_val), np.expm1(m.predict(X_val))))

    sc = StandardScaler()
    m2 = ModelClass(**kwargs)
    m2.fit(sc.fit_transform(X_train), y_train)
    rmse_com = np.sqrt(mean_squared_error(
        np.expm1(y_val), np.expm1(m2.predict(sc.transform(X_val)))))

    print(f'{nome_modelo}: Sem Scaling=${rmse_sem:,.0f} | Com Scaling=${rmse_com:,.0f}')'''

CODE['mod_importance'] = '''importancias = pd.Series(
    modelos['XGBoost'].feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

top20 = importancias.head(20)
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(top20.index[::-1], top20.values[::-1],
        color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, 20)))
ax.set_title('Top 20 Features mais Importantes (XGBoost)')
ax.set_xlabel('Importância (F-Score)')
plt.tight_layout(); plt.show()'''

CODE['mod_simplificado'] = '''TOP_FEATURES = importancias.head(10).index.tolist()

xgb_simp = XGBRegressor(n_estimators=300, learning_rate=0.05,
                         max_depth=4, random_state=42, verbosity=0)
xgb_simp.fit(X_train[TOP_FEATURES], y_train)
pred_simp = np.expm1(xgb_simp.predict(X_val[TOP_FEATURES]))

rmse_simp = np.sqrt(mean_squared_error(np.expm1(y_val), pred_simp))
r2_simp   = r2_score(np.expm1(y_val), pred_simp)
print(f'Completo  ({X.shape[1]} features): RMSE=${rmse_full:,.0f} | R²={r2_full:.4f}')
print(f'Simplif. ({len(TOP_FEATURES)} features): RMSE=${rmse_simp:,.0f} | R²={r2_simp:.4f}')'''

CODE['mod_cv'] = '''from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
modelos_cv = {
    'Ridge':         Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))]),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost':       XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                                   random_state=42, subsample=0.8, colsample_bytree=0.8, verbosity=0),
}
for nome, modelo in modelos_cv.items():
    scores = -cross_val_score(modelo, X, y_log,
                              scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)
    print(f'{nome:<22}: RMSLE = {scores.mean():.4f} ± {scores.std():.4f}')'''

# ═══════════════════════════════════════════════════════════════
# HTML COMPLETO
# ═══════════════════════════════════════════════════════════════
print("\n=== GERANDO HTML ===")

# Dados do projeto
skewness_orig = f"{df_av2['SalePrice'].skew():.4f}"
skewness_log  = f"{df_av2['LogSalePrice'].skew():.4f}"
media_sim  = f"${grupo_sim.mean():,.2f}"
media_nao  = f"${grupo_nao.mean():,.2f}"
median_sim = f"${grupo_sim.median():,.2f}"
median_nao = f"${grupo_nao.median():,.2f}"
n_grupos_sim = len(grupo_sim); n_grupos_nao = len(grupo_nao)
ic_g_str = f"[${ic_low_g:,.2f}, ${ic_high_g:,.2f}]"
ic_sim_str = f"[${ic_sim[1]:,.2f}, ${ic_sim[2]:,.2f}]"
ic_nao_str = f"[${ic_nao[1]:,.2f}, ${ic_nao[2]:,.2f}]"

modelos_tabela = met['modelos']

top10_feats = met['top10_features']

html = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>House Prices – Relatório Completo ABNT</title>
<style>
  @page {{ size: A4; margin: 3cm 2cm 2cm 3cm; }}
  body {{ font-family:"Times New Roman",Times,serif; font-size:12pt; line-height:1.5; color:#000; text-align:justify; background:#fff; }}
  .page {{ max-width:21cm; margin:0 auto; padding:3cm 2cm 2cm 3cm; }}
  h1 {{ font-size:14pt; font-weight:bold; text-align:center; text-transform:uppercase; margin:2em 0 0.8em; page-break-before:always; }}
  h1.no-break {{ page-break-before:avoid; }}
  h2 {{ font-size:12pt; font-weight:bold; text-transform:uppercase; margin:1.5em 0 0.5em; }}
  h3 {{ font-size:12pt; font-weight:bold; margin:1.2em 0 0.3em; }}
  h4 {{ font-size:12pt; font-weight:bold; font-style:italic; margin:1em 0 0.3em; }}
  p {{ margin:0 0 0.7em; text-indent:1.25cm; }}
  p.ni {{ text-indent:0; }}
  ul,ol {{ margin:0 0 0.7em 1.25cm; }}
  li {{ margin-bottom:0.2em; }}
  .capa {{ text-align:center; }}
  .capa p {{ text-indent:0; text-align:center; margin:0.25em 0; }}
  figure {{ text-align:center; margin:1.2em 0; }}
  figure img {{ max-width:88%; border:1px solid #ccc; }}
  figcaption {{ font-size:10pt; text-align:center; margin-top:0.3em; }}
  table {{ width:100%; border-collapse:collapse; margin:0.8em 0; font-size:10.5pt; }}
  th {{ background:#f0f0f0; border:1px solid #999; padding:5px 8px; font-weight:bold; text-align:center; }}
  td {{ border:1px solid #999; padding:4px 8px; }}
  td.l {{ text-align:left; }}
  td.c {{ text-align:center; }}
  td.r {{ text-align:right; }}
  pre {{ font-family:"Courier New",monospace; font-size:9pt; background:#f5f5f5; border:1px solid #ccc; padding:10px; white-space:pre-wrap; margin:0.7em 0; }}
  .destaque {{ background:#fffbe6; border-left:4px solid #f0a500; padding:8px 12px; margin:0.8em 0; font-size:11pt; }}
  .alerta {{ background:#fff0f0; border-left:4px solid #c0392b; padding:8px 12px; margin:0.8em 0; font-size:11pt; }}
  hr {{ border:none; border-top:1px solid #aaa; margin:1.5em 0; }}
  .ref {{ text-indent:0; margin-bottom:0.5em; padding-left:0; }}
  .sumario-linha {{ display:flex; justify-content:space-between; border-bottom:1px dotted #999; margin:0.3em 0; }}
</style>
</head>
<body>
<div class="page">

<!-- ═══ CAPA ═══ -->
<div class="capa" style="min-height:90vh;display:flex;flex-direction:column;justify-content:space-between;align-items:center;">
  <div style="text-align:center;margin-top:1cm;">
    <p style="font-size:13pt;font-weight:bold;text-transform:uppercase;">Centro Universitário Maurício de Nassau</p>
    <p style="font-size:12pt;">Curso de Sistemas de Informação</p>
    <p style="font-size:12pt;">Disciplina de Machine Learning</p>
    <p style="font-size:12pt;">Professor: Flávio José Ferreira Junior</p>
  </div>
  <div style="text-align:center;margin:2cm 0;">
    <p style="font-size:16pt;font-weight:bold;text-transform:uppercase;">PROJETO 2</p>
    <p style="font-size:15pt;font-weight:bold;text-transform:uppercase;">House Prices – Regressão</p>
    <p style="font-size:12pt;margin-top:0.5cm;font-style:italic;">Análise Exploratória · Análise Estatística · Engenharia de Features · Modelagem · Avaliação</p>
  </div>
  <div style="text-align:center;margin-bottom:1cm;">
    <p style="font-weight:bold;">Integrantes:</p>
    <p>Wendell de Santana – 01634820</p>
    <p>Lucas Pottes Monteiro de Barros – 01402626</p>
    <p>Matheus Campelo Farias da Silva – 01429814</p>
    <p>Danilo Mateus Silva do Nascimento – 01615380</p>
    <p>Kayo Vinícius T. Custódio</p>
    <p>Guilherme Cardoso Guedes de Lima – 01614628</p>
    <br>
    <p>Recife, 2025</p>
  </div>
</div>

<!-- ═══ SUMÁRIO ═══ -->
<h1 class="no-break">Sumário</h1>
<div>
  <div class="sumario-linha"><span>1 INTRODUÇÃO</span><span>3</span></div>
  <div class="sumario-linha"><span>2 BASE DE DADOS</span><span>3</span></div>
  <div class="sumario-linha"><span>3 ANÁLISE EXPLORATÓRIA DOS DADOS</span><span>4</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;3.1 Importações e Pré-processamento</span><span>4</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;3.2 Variáveis que Mais Impactam o Preço</span><span>5</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;3.3 Área Construída tem Relação Linear com o Preço?</span><span>6</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;3.4 Ano de Construção Influencia a Valorização?</span><span>6</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;3.5 Casas Reformadas Valem Mais?</span><span>7</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;3.6 Bairros Apresentam Diferenças Estatísticas?</span><span>8</span></div>
  <div class="sumario-linha"><span>4 ANÁLISE ESTATÍSTICA E TESTES DE HIPÓTESE</span><span>9</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;4.1 Teste de Normalidade do SalePrice</span><span>9</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;4.2 Casas Reformadas Valem Mais? – Teste t e Mann-Whitney</span><span>10</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;4.3 Paradoxo de Simpson</span><span>11</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;4.4 Bairros Têm Preços Diferentes? – ANOVA e Tukey HSD</span><span>12</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;4.5 Correlação de Pearson – GrLivArea</span><span>13</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;4.6 Correlação de Spearman – YearBuilt</span><span>14</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;4.7 Intervalo de Confiança (95%)</span><span>14</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;4.8 Resumo Geral dos Testes</span><span>15</span></div>
  <div class="sumario-linha"><span>5 ENGENHARIA DE FEATURES</span><span>16</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;5.1 Tratamento de Valores Ausentes</span><span>16</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;5.2 Criação de Novas Features</span><span>17</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;5.3 Encoding de Variáveis Categóricas</span><span>18</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;5.4 Transformação Logarítmica do Alvo</span><span>18</span></div>
  <div class="sumario-linha"><span>6 MODELAGEM</span><span>19</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;6.1 Treinamento dos Modelos</span><span>19</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;6.2 Comparação dos Modelos</span><span>20</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;6.3 Impacto da Transformação Logarítmica</span><span>21</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;6.4 Impacto do Feature Scaling</span><span>21</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;6.5 Análise do Melhor Modelo</span><span>22</span></div>
  <div class="sumario-linha"><span>7 REGRAS DE NEGÓCIO</span><span>23</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;7.1 Importância das Features</span><span>23</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;7.2 Modelo Simplificado</span><span>24</span></div>
  <div class="sumario-linha"><span>&nbsp;&nbsp;&nbsp;7.3 Características que Mais Agregam Valor</span><span>24</span></div>
  <div class="sumario-linha"><span>8 AVALIAÇÃO FINAL – CROSS-VALIDATION</span><span>25</span></div>
  <div class="sumario-linha"><span>9 CONCLUSÃO</span><span>26</span></div>
  <div class="sumario-linha"><span>REFERÊNCIAS</span><span>28</span></div>
</div>

<!-- ═══ 1 INTRODUÇÃO ═══ -->
<h1>1 Introdução</h1>
<p>O presente trabalho documenta o desenvolvimento do Projeto 2 da disciplina de Machine Learning, cujo tema central é a predição do preço de venda de imóveis residenciais em Ames, Iowa (EUA), utilizando técnicas de regressão supervisionada. O projeto foi estruturado em três etapas sequenciais: (i) Análise Exploratória dos Dados (AV1), (ii) Análise Estatística com Testes de Hipótese (AV2) e (iii) Engenharia de Features, Modelagem e Avaliação.</p>
<p>A motivação do projeto reside na complexidade da precificação imobiliária, que envolve dezenas de variáveis físicas, estruturais, de localização e contextuais. Com um conjunto de dados composto por 79 variáveis explicativas, o desafio consiste em identificar os principais determinantes de preço e construir modelos preditivos robustos, capazes de generalizar para novos imóveis.</p>
<p>Todo o pipeline foi implementado em Python 3.12, utilizando as bibliotecas pandas, numpy, matplotlib, seaborn, scipy, statsmodels, scikit-learn e XGBoost. Os notebooks foram desenvolvidos no Visual Studio Code com extensão Jupyter.</p>

<!-- ═══ 2 BASE DE DADOS ═══ -->
<h1>2 Base de Dados</h1>
<p>O dataset utilizado é proveniente da competição Kaggle <em>House Prices: Advanced Regression Techniques</em>, baseado em dados de transações imobiliárias reais em Ames, Iowa, coletados entre 2006 e 2010 por Dean De Cock (2011).</p>
<table>
  <tr><th>Conjunto</th><th>Amostras</th><th>Variáveis</th><th>Alvo</th></tr>
  <tr><td class="l">Treinamento (train.csv)</td><td class="c">1.460</td><td class="c">81</td><td class="c">SalePrice</td></tr>
  <tr><td class="l">Teste (test.csv)</td><td class="c">1.459</td><td class="c">80</td><td class="c">–</td></tr>
  <tr><td class="l">Submissão (sample_submission.csv)</td><td class="c">1.459</td><td class="c">2</td><td class="c">–</td></tr>
</table>
<p>Das 79 variáveis explicativas, 43 são categóricas (tipo de zoneamento, bairro, qualidade de acabamento, etc.) e 36 são numéricas (áreas, contagens de cômodos, anos de construção e reforma, etc.). A variável alvo <strong>SalePrice</strong> representa o preço final de venda em dólares americanos.</p>
<p>As principais colunas com valores ausentes eram: <em>PoolQC</em> (99,5%), <em>MiscFeature</em> (96,3%), <em>Alley</em> (93,8%), <em>Fence</em> (80,8%) e <em>FireplaceQu</em> (47,3%). Em sua maioria, esses valores ausentes não representam dados faltantes, mas sim a <strong>ausência do recurso</strong> no imóvel — distinção fundamental para o tratamento correto na etapa de pré-processamento.</p>

<!-- ═══ 3 ANÁLISE EXPLORATÓRIA ═══ -->
<h1>3 Análise Exploratória dos Dados</h1>

<h2>3.1 Importações e Pré-processamento</h2>
<p>A análise exploratória foi realizada no notebook <em>analise_exploratoria.ipynb</em>. O pré-processamento inicial consiste no carregamento dos dados, tratamento de valores ausentes por imputação com a média (para variáveis numéricas) e remoção de duplicatas.</p>

{code_block(CODE['av1_imports'])}

<h2>3.2 Quais Variáveis Mais Impactam o Preço?</h2>
<p>Para identificar as variáveis com maior influência sobre o preço de venda, foi calculada a correlação de Pearson entre todas as variáveis numéricas e a variável alvo <em>SalePrice</em>. As 10 variáveis com maior correlação (em valor absoluto) foram selecionadas para a construção da matriz de correlação.</p>

{code_block(CODE['av1_heatmap'])}

{fig_block('av1_correlacao_heatmap', 1, 'Matriz de correlação das 10 variáveis numéricas mais correlacionadas com SalePrice')}

<p>A análise revelou que as variáveis com maior correlação positiva com <em>SalePrice</em> são: <strong>OverallQual</strong> (qualidade geral da construção, r&nbsp;=&nbsp;0,79), <strong>GrLivArea</strong> (área de habitação acima do solo, r&nbsp;=&nbsp;0,71), <strong>GarageCars</strong> (capacidade de vagas na garagem, r&nbsp;=&nbsp;0,64) e <strong>GarageArea</strong> (área da garagem, r&nbsp;=&nbsp;0,62). Esses resultados indicam que qualidade construtiva e tamanho são os principais vetores de valor imobiliário.</p>

<h2>3.3 Área Construída tem Relação Linear com o Preço?</h2>
<p>O diagrama de dispersão entre <em>GrLivArea</em> (área de habitação em sq ft) e <em>SalePrice</em> foi plotado acompanhado da reta de regressão linear, permitindo avaliar a adequação de um modelo linear.</p>

{code_block(CODE['av1_scatter'])}

{fig_block('av1_grlivarea', 2, 'Diagrama de dispersão entre GrLivArea e SalePrice com reta de regressão')}

<p>O gráfico confirma uma forte relação linear positiva entre a área construída e o preço (r&nbsp;≈&nbsp;0,71). A relação é predominantemente linear para imóveis com menos de 4.000 sq ft. Observam-se alguns outliers de alto valor de área com preços abaixo do esperado, possivelmente imóveis comerciais ou com condições de venda atípicas.</p>

<h2>3.4 Ano de Construção Influencia a Valorização?</h2>
<p>O preço médio de venda foi agrupado por ano de construção para investigar a relação temporal entre a idade do imóvel e seu valor de mercado.</p>

{code_block(CODE['av1_yearbuilt'])}

{fig_block('av1_yearbuilt', 3, 'Evolução do preço médio de venda agrupado pelo ano de construção do imóvel')}

<p>O gráfico evidencia uma tendência positiva geral: imóveis mais recentes tendem a ser mais valorizados. Casas construídas após 2000 apresentam preços médios consistentemente superiores a US$200.000. O padrão não é monótono — há oscilações que refletem a variabilidade das amostras para anos com poucos registros — mas a tendência geral é clara: a antiguidade deprecia o valor do imóvel.</p>

<h2>3.5 Casas Reformadas Valem Mais?</h2>
<p>Uma variável auxiliar foi criada para classificar os imóveis como reformados (quando o ano de reforma é posterior ao ano de construção) ou não reformados. O boxplot compara a distribuição de preços entre os dois grupos.</p>

{code_block(CODE['av1_reforma'])}

{fig_block('av1_reforma', 4, 'Comparação da distribuição de preços entre imóveis reformados e não reformados')}

<p>Visualmente, os imóveis classificados como "não reformados" apresentam mediana de preço levemente maior. Este resultado aparentemente contraintuitivo será investigado em profundidade na Análise Estatística (Seção 4), onde será identificado um Paradoxo de Simpson relacionado à composição dos grupos.</p>

<h2>3.6 Bairros Apresentam Diferenças Estatísticas Relevantes?</h2>
<p>Os 25 bairros do dataset foram ordenados pela mediana do preço de venda e visualizados por meio de boxplots. Um teste ANOVA preliminar foi aplicado para verificar a existência de diferenças estatisticamente significativas.</p>

{code_block(CODE['av1_bairros'])}

{fig_block('av1_bairros', 5, 'Distribuição de preços de venda por bairro, ordenados da mediana mais alta para a mais baixa')}

<p>A visualização demonstra ampla variação de preços entre bairros. O ANOVA resultou em p-valor próximo de zero (p&nbsp;&lt;&nbsp;0,0001), confirmando que pelo menos um bairro possui média de preço estatisticamente diferente dos demais. Os bairros <em>NoRidge</em>, <em>NridgHt</em> e <em>StoneBr</em> apresentam as maiores medianas, enquanto <em>MeadowV</em> e <em>IDOTRR</em> figuram entre os mais baratos.</p>

<!-- ═══ 4 ANÁLISE ESTATÍSTICA ═══ -->
<h1>4 Análise Estatística e Testes de Hipótese</h1>
<p>A segunda etapa do projeto, documentada no notebook <em>AV2_analise_estatistica.ipynb</em>, formaliza estatisticamente as hipóteses levantadas na AV1, aplicando testes paramétricos e não-paramétricos com nível de significância α&nbsp;=&nbsp;0,05.</p>

{code_block(CODE['av2_imports'])}

<h2>4.1 Teste de Normalidade do SalePrice</h2>
<p><strong>Hipóteses:</strong> H₀: SalePrice segue distribuição normal | H₁: SalePrice NÃO segue distribuição normal</p>
<p>Foram aplicados dois testes: Shapiro-Wilk (sobre amostra de 1.000 observações, conforme limite recomendado) e Kolmogorov-Smirnov (sobre toda a amostra).</p>

{code_block(CODE['av2_normalidade'])}

{fig_block('av2_norm_orig', 6, 'Histograma com KDE e Q-Q plot do SalePrice na escala original')}

<p><strong>Resultado:</strong> Shapiro-Wilk p&nbsp;=&nbsp;{p_sw:.2e} | KS p&nbsp;=&nbsp;{p_ks:.2e} → Rejeitamos H₀. SalePrice apresenta assimetria positiva (skewness&nbsp;=&nbsp;{skewness_orig}), típica de dados de preço, e NÃO segue distribuição normal. O Q-Q plot confirma o desvio da normalidade especialmente nas caudas.</p>

<h3>4.1.1 Transformação Logarítmica</h3>
<p>A transformação log foi aplicada para reduzir a assimetria e aproximar a distribuição da normalidade, premissa de modelos lineares.</p>

{code_block(CODE['av2_log'])}

{fig_block('av2_norm_log', 7, 'Histograma com KDE e Q-Q plot após transformação log(SalePrice)')}

<p><strong>Resultado:</strong> Após a transformação, a assimetria reduziu de {skewness_orig} para {skewness_log}. Embora o Shapiro-Wilk ainda rejeite a normalidade perfeita (p&nbsp;=&nbsp;{p_sw_log:.2e}), a distribuição é substancialmente mais simétrica. Isso motiva o uso de <code>log1p(SalePrice)</code> como alvo na etapa de modelagem.</p>
<div class="destaque"><strong>Nota de consistência:</strong> A AV2 utiliza <code>np.log(SalePrice)</code> para análise, pois SalePrice &gt; 0 no conjunto de treinamento. A modelagem utiliza <code>np.log1p(SalePrice)</code> — matematicamente equivalente aqui, mas mais segura para datasets onde zero poderia ocorrer.</div>

<h2>4.2 Casas Reformadas Valem Mais? – Teste t de Welch e Mann-Whitney U</h2>
<p><strong>Hipóteses:</strong> H₀: A média de preço de casas reformadas = não reformadas | H₁: As médias são diferentes (α&nbsp;=&nbsp;0,05)</p>
<p>Antes do Teste t, o Teste de Levene verificou homogeneidade de variâncias. Como SalePrice não é normal (comprovado em 4.1), o Mann-Whitney U foi aplicado como complemento não-paramétrico.</p>

{code_block(CODE['av2_teste_t'])}

{fig_block('av2_reforma_t', 8, 'Boxplot e Violin plot comparando a distribuição de preços entre imóveis reformados e não reformados')}

<table>
  <tr><th>Teste</th><th>Estatística</th><th>p-valor</th><th>Decisão (α=0,05)</th></tr>
  <tr><td class="l">Levene (variâncias)</td><td class="c">{stat_levene:.4f}</td><td class="c">{p_levene:.4f}</td><td class="c">{'Variâncias diferentes' if p_levene<0.05 else 'Variâncias iguais'}</td></tr>
  <tr><td class="l">Teste t de Welch (médias)</td><td class="c">{t_stat:.4f}</td><td class="c">{p_valor_t:.4f}</td><td class="c">{'Rejeita H₀' if p_valor_t<0.05 else 'Não rejeita H₀'}</td></tr>
  <tr><td class="l">Mann-Whitney U (distribuições)</td><td class="c">{stat_mw:.0f}</td><td class="c">{p_mw:.4f}</td><td class="c">{'Rejeita H₀' if p_mw<0.05 else 'Não rejeita H₀'}</td></tr>
</table>

<table>
  <tr><th>Grupo</th><th>n</th><th>Média</th><th>Mediana</th></tr>
  <tr><td class="l">Reformada = Sim</td><td class="c">{n_grupos_sim}</td><td class="r">{media_sim}</td><td class="r">{median_sim}</td></tr>
  <tr><td class="l">Reformada = Não</td><td class="c">{n_grupos_nao}</td><td class="r">{media_nao}</td><td class="r">{median_nao}</td></tr>
</table>

<p><strong>Interpretação:</strong> O Teste t não encontrou evidência de diferença significativa nas <em>médias</em> (p&nbsp;=&nbsp;{p_valor_t:.3f} &gt; 0,05). O Mann-Whitney, porém, identificou diferença significativa nas <em>distribuições</em> (p&nbsp;=&nbsp;{p_mw:.4f} &lt; 0,05). Essa aparente contradição é explicada pelo Paradoxo de Simpson, detalhado a seguir.</p>

<h2>4.3 Paradoxo de Simpson – Análise por Era de Construção</h2>
<p>O grupo "Não reformada" apresenta mediana maior ({median_nao}) que o grupo "Reformada" ({median_sim}). Isso ocorre porque a codificação <em>Foi_Reformada</em> (YearRemodAdd &gt; YearBuilt) classifica casas de nova construção — onde YearRemodAdd = YearBuilt por padrão — como "não reformadas". Essas casas novas são naturalmente mais caras, distorcendo a comparação global.</p>

{code_block(CODE['av2_simpson'])}

{fig_block('av2_simpson', 9, 'Paradoxo de Simpson: dentro de cada era de construção, imóveis reformados são mais caros; a inversão global é causada pela concentração de nova construção no grupo não reformado')}

<div class="destaque"><strong>Conclusão:</strong> Dentro de cada era de construção, casas reformadas apresentam mediana de preço igual ou maior. O padrão intuitivo existe — o Paradoxo de Simpson é causado pelo confundimento com <em>YearBuilt</em>. Para análises de impacto de reforma, é necessário controlar pelo ano de construção.</div>

<h2>4.4 Bairros Têm Preços Diferentes? – ANOVA e Post-hoc Tukey HSD</h2>
<p><strong>Hipóteses:</strong> H₀: As médias de SalePrice são iguais em todos os bairros | H₁: Pelo menos um bairro tem média diferente</p>

{code_block(CODE['av2_anova'])}

{fig_block('av2_bairros', 10, 'Comparação dos 5 bairros mais caros versus 5 mais baratos por distribuição de preços')}

<table>
  <tr><th>Teste</th><th>Estatística</th><th>p-valor</th><th>Decisão</th></tr>
  <tr><td class="l">ANOVA one-way</td><td class="c">F = {f_stat:.2f}</td><td class="c">{p_anova:.2e}</td><td class="c">Rejeita H₀</td></tr>
  <tr><td class="l">Tukey HSD (pós-hoc)</td><td class="c">–</td><td class="c">Vários &lt; 0,05</td><td class="c">{n_sig_bairros} pares sig.</td></tr>
</table>

<p>O teste ANOVA rejeitou H₀ com p&nbsp;≈&nbsp;0 (F&nbsp;=&nbsp;{f_stat:.2f}). O post-hoc Tukey HSD identificou <strong>{n_sig_bairros} pares de bairros</strong> com diferença estatisticamente significativa de preço médio (de um total de {len(tukey_df)} pares testados). Essa análise confirma que a localização (bairro) é uma das principais variáveis para a precificação imobiliária.</p>

<h2>4.5 Área Construída tem Correlação com o Preço? – Pearson</h2>
<p><strong>Hipóteses:</strong> H₀: ρ(GrLivArea, SalePrice) = 0 | H₁: ρ ≠ 0</p>

{code_block(CODE['av2_pearson'])}

{fig_block('av2_pearson', 11, 'Correlação de Pearson entre GrLivArea e SalePrice (escala original e logarítmica)')}

<table>
  <tr><th>Teste</th><th>Coeficiente</th><th>p-valor</th><th>Decisão</th><th>Força</th></tr>
  <tr><td class="l">Pearson (r)</td><td class="c">{r_pearson:.4f}</td><td class="c">{p_pearson:.2e}</td><td class="c">Rejeita H₀</td><td class="c">Forte</td></tr>
  <tr><td class="l">Spearman (ρ)</td><td class="c">{r_spearman:.4f}</td><td class="c">{p_spearman:.2e}</td><td class="c">Rejeita H₀</td><td class="c">Forte</td></tr>
</table>

<p>Ambos os coeficientes confirmam <strong>correlação forte e positiva</strong> entre área construída e preço (r&nbsp;=&nbsp;{r_pearson:.4f}). O IC 95% para r via transformação de Fisher é [{np.tanh(np.arctanh(r_pearson)-1.96/np.sqrt(len(df_av2)-3)):.4f}, {np.tanh(np.arctanh(r_pearson)+1.96/np.sqrt(len(df_av2)-3)):.4f}], excluindo 0 com ampla margem. A correlação com log(SalePrice) é ainda mais forte (r&nbsp;=&nbsp;{r_log:.4f}), confirmando a utilidade da transformação logarítmica.</p>

<h2>4.6 Ano de Construção Influencia o Preço? – Spearman</h2>
<p><strong>Hipóteses:</strong> H₀: ρ(YearBuilt, SalePrice) = 0 | H₁: ρ ≠ 0</p>
<p>Como a relação entre ano e preço pode ser não-linear, foi preferido o coeficiente de Spearman, que mede associação monotônica sem exigir linearidade.</p>

{code_block(CODE['av2_spearman'])}

{fig_block('av2_spearman', 12, 'Correlação de Spearman entre YearBuilt e SalePrice: linha de preço médio por ano e dispersão geral')}

<table>
  <tr><th>Teste</th><th>Coeficiente</th><th>p-valor</th><th>Direção</th></tr>
  <tr><td class="l">Pearson (r)</td><td class="c">{r_pe_year:.4f}</td><td class="c">{p_sp_year:.2e}</td><td class="c">Positiva</td></tr>
  <tr><td class="l">Spearman (ρ)</td><td class="c">{r_sp_year:.4f}</td><td class="c">{p_sp_year:.2e}</td><td class="c">Positiva</td></tr>
</table>

<p>Rejeitamos H₀. A associação positiva (ρ&nbsp;=&nbsp;{r_sp_year:.4f}) confirma que casas mais recentes tendem a custar mais, resultado consistente com a análise visual da AV1.</p>

<h2>4.7 Intervalo de Confiança (95%) para a Média do SalePrice</h2>
<p>Estimativa do intervalo onde, com 95% de confiança, encontra-se a média populacional do preço de venda.</p>

{code_block(CODE['av2_ic'])}

{fig_block('av2_ic', 13, 'Intervalo de confiança 95% para o preço médio por grupo de reforma')}

<table>
  <tr><th>Grupo</th><th>n</th><th>Média</th><th>IC 95%</th></tr>
  <tr><td class="l">Geral (todos imóveis)</td><td class="c">{n_ic}</td><td class="r">${m_ic:,.2f}</td><td class="r">{ic_g_str}</td></tr>
  <tr><td class="l">Reformada = Sim</td><td class="c">{n_grupos_sim}</td><td class="r">{media_sim}</td><td class="r">{ic_sim_str}</td></tr>
  <tr><td class="l">Reformada = Não</td><td class="c">{n_grupos_nao}</td><td class="r">{media_nao}</td><td class="r">{ic_nao_str}</td></tr>
</table>

<p>Os ICs dos dois grupos de reforma se sobrepõem, corroborando a decisão do Teste t de não rejeitar H₀ para as médias. A média geral de SalePrice é estimada em US${m_ic:,.2f}, com IC 95% de {ic_g_str}.</p>

<h2>4.8 Resumo Geral dos Testes Estatísticos</h2>

<table>
  <tr><th>Hipótese</th><th>Teste</th><th>p-valor</th><th>Decisão</th><th>Conclusão</th></tr>
  <tr><td class="l">SalePrice é normal?</td><td class="c">Shapiro-Wilk</td><td class="c">{p_sw:.2e}</td><td class="c">Rejeita H₀</td><td class="l">NÃO é normal (assimetria positiva)</td></tr>
  <tr><td class="l">log(SalePrice) é normal?</td><td class="c">Shapiro-Wilk</td><td class="c">{p_sw_log:.2e}</td><td class="c">Rejeita H₀</td><td class="l">Reduz assimetria significativamente</td></tr>
  <tr><td class="l">Reformadas valem mais? (médias)</td><td class="c">Teste t Welch</td><td class="c">{p_valor_t:.4f}</td><td class="c">Não rejeita H₀</td><td class="l">Sem diferença nas médias (p &gt; 0,05)</td></tr>
  <tr><td class="l">Reformadas valem mais? (distribuições)</td><td class="c">Mann-Whitney</td><td class="c">{p_mw:.4f}</td><td class="c">Rejeita H₀</td><td class="l">Distribuições diferentes; Paradoxo de Simpson</td></tr>
  <tr><td class="l">Bairros têm preços diferentes?</td><td class="c">ANOVA + Tukey</td><td class="c">{p_anova:.2e}</td><td class="c">Rejeita H₀</td><td class="l">{n_sig_bairros} pares significativos</td></tr>
  <tr><td class="l">GrLivArea correlaciona com preço?</td><td class="c">Pearson+Spearman</td><td class="c">{p_pearson:.2e}</td><td class="c">Rejeita H₀</td><td class="l">Correlação forte (r = {r_pearson:.4f})</td></tr>
  <tr><td class="l">YearBuilt associado ao preço?</td><td class="c">Spearman</td><td class="c">{p_sp_year:.2e}</td><td class="c">Rejeita H₀</td><td class="l">Associação positiva (ρ = {r_sp_year:.4f})</td></tr>
</table>

<!-- ═══ 5 ENGENHARIA DE FEATURES ═══ -->
<h1>5 Engenharia de Features</h1>
<p>A etapa de engenharia de features, documentada no notebook <em>modelagem.ipynb</em>, visa preparar os dados para a modelagem por meio de: (i) tratamento criterioso de valores ausentes, (ii) criação de novas variáveis derivadas de conhecimento de domínio, e (iii) codificação de variáveis categóricas.</p>

<h2>5.1 Tratamento de Valores Ausentes</h2>
<p>A estratégia de imputação foi definida com base no domínio do problema, distinguindo dois tipos de valores ausentes:</p>
<ul>
  <li><strong>NA como "ausência do recurso":</strong> Para 15 colunas (PoolQC, Alley, FireplaceQu, GarageType, etc.), o NA indica que o imóvel não possui aquele recurso. Esses valores foram substituídos pela categoria literal <em>"None"</em>.</li>
  <li><strong>NA como dado faltante:</strong> Para variáveis numéricas restantes, utilizou-se a mediana (robusta a outliers). Para categóricas residuais, a moda.</li>
</ul>

{code_block(CODE['mod_missing'])}

<p>Após o tratamento, o conjunto de treinamento ficou com <strong>zero valores ausentes</strong>.</p>

<h2>5.2 Criação de Novas Features</h2>
<p>Dez novas variáveis foram criadas combinando informações existentes com base no conhecimento do negócio imobiliário:</p>

{code_block(CODE['mod_features'])}

<table>
  <tr><th>Feature</th><th>Composição</th><th>Justificativa</th></tr>
  <tr><td>TotalSF</td><td>TotalBsmtSF + 1stFlrSF + 2ndFlrSF</td><td>Área total real (incluindo porão)</td></tr>
  <tr><td>TotalBaths</td><td>Soma ponderada de banheiros</td><td>Consolida 4 indicadores de banheiros</td></tr>
  <tr><td>HouseAge</td><td>YrSold − YearBuilt</td><td>Idade no momento da venda</td></tr>
  <tr><td>YearsSinceRemod</td><td>YrSold − YearRemodAdd</td><td>Tempo desde última reforma</td></tr>
  <tr><td>Foi_Reformada</td><td>YearRemodAdd &gt; YearBuilt</td><td>Flag binária de reforma (confirmada na AV1)</td></tr>
  <tr><td>QualArea</td><td>OverallQual × GrLivArea</td><td>Interação qualidade-tamanho (feature mais importante)</td></tr>
  <tr><td>TotalPorchSF</td><td>Soma de todas as varandas</td><td>Área externa útil</td></tr>
  <tr><td>HasPool</td><td>PoolArea &gt; 0</td><td>Presença de piscina</td></tr>
  <tr><td>HasFireplace</td><td>Fireplaces &gt; 0</td><td>Presença de lareira</td></tr>
  <tr><td>HasGarage</td><td>GarageArea &gt; 0</td><td>Presença de garagem</td></tr>
</table>

{fig_block('mod_novas_features', 14, 'Relação das 6 principais novas features criadas com SalePrice')}

<p>A Figura 14 evidencia que <em>TotalSF</em> e <em>QualArea</em> possuem as correlações mais fortes e lineares com o preço — confirmando a utilidade da engenharia de features em relação às variáveis originais isoladas.</p>

<h2>5.3 Encoding de Variáveis Categóricas</h2>
<p>Duas estratégias de codificação foram aplicadas:</p>
<ul>
  <li><strong>Encoding Ordinal:</strong> Para 10 variáveis de qualidade (ExterQual, KitchenQual, BsmtQual, etc.), foi aplicado mapeamento numérico ordinal: None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5, preservando a ordem natural.</li>
  <li><strong>One-Hot Encoding (OHE):</strong> Para as demais variáveis categóricas, utilizou-se <code>pd.get_dummies(drop_first=True)</code>. Train e test foram concatenados antes do OHE para garantir colunas idênticas.</li>
</ul>

{code_block(CODE['mod_encoding'])}

<p>Após o encoding, o dataset expandiu para <strong>239 features</strong>.</p>

<h2>5.4 Transformação Logarítmica do Alvo</h2>
<p>Motivada pelos resultados do Teste de Normalidade (AV2, Seção 4.1), a variável alvo recebeu a transformação <code>log1p(SalePrice)</code>. Na predição, os valores são revertidos com <code>expm1()</code>.</p>

{code_block(CODE['mod_log'])}

{fig_block('mod_dist_saleprice', 15, 'Distribuição do SalePrice na escala original (esquerda) e após transformação log1p (direita)')}

<!-- ═══ 6 MODELAGEM ═══ -->
<h1>6 Modelagem</h1>

<h2>6.1 Treinamento dos Modelos</h2>
<p>Os dados foram divididos em 80% treino (1.168 amostras) e 20% validação (292 amostras), com semente aleatória fixada em 42. Seis modelos foram treinados:</p>

{code_block(CODE['mod_modelos'])}

<h2>6.2 Comparação dos Modelos</h2>

{fig_block('mod_comparacao', 16, 'Comparação de RMSE e R² entre os seis modelos testados no conjunto de validação')}

<table>
  <tr><th>Modelo</th><th>RMSE (USD)</th><th>R²</th></tr>
  {''.join(f"<tr><td class='l'>{'<strong>'+k+'</strong>' if k==met['melhor'] else k}</td><td class='r'>${v['RMSE']:,.0f}</td><td class='c'>{v['R2']:.4f}</td></tr>" for k,v in sorted(modelos_tabela.items(), key=lambda x: x[1]['RMSE']))}
</table>

<p>O modelo <strong>{met['melhor']}</strong> obteve o menor RMSE (US${met['modelos'][met['melhor']]['RMSE']:,.0f}) e o maior R² ({met['modelos'][met['melhor']]['R2']:.4f}) na validação. Com 239 features após o OHE, a regularização L2 do Ridge mostrou-se eficaz para controlar o sobreajuste em espaço de alta dimensionalidade. Os modelos de ensemble ficaram atrás, pois seus hiperparâmetros não foram otimizados com busca exaustiva.</p>

<h2>6.3 Impacto da Transformação Logarítmica</h2>
<p>Para quantificar o ganho da transformação, o XGBoost foi treinado com e sem log1p no alvo:</p>

{code_block(CODE['mod_log_compare'])}

{fig_block('mod_log_impact', 17, 'Comparação de RMSE do XGBoost com e sem transformação logarítmica do alvo')}

<p>A transformação logarítmica reduziu o RMSE de US${met['rmse_sem_log']:,.0f} para US${met['rmse_com_log']:,.0f} — melhora de {(met['rmse_sem_log']-met['rmse_com_log'])/met['rmse_sem_log']*100:.1f}%. O impacto é mais expressivo para modelos lineares (que assumem normalidade dos resíduos), mas também beneficia o XGBoost ao reduzir o efeito desproporcionado de outliers de alto valor.</p>

<h2>6.4 Impacto do Feature Scaling</h2>
<p>O efeito da normalização (StandardScaler) foi avaliado nos modelos lineares Ridge e Lasso:</p>

{code_block(CODE['mod_scaling'])}

{fig_block('mod_scaling', 18, 'Impacto do Feature Scaling (StandardScaler) no RMSE de Ridge e Lasso')}

<p>O scaling reduziu expressivamente o RMSE do Lasso e estabilizou o Ridge. Modelos regularizados são sensíveis à escala das features porque o parâmetro α penaliza os coeficientes proporcionalmente à magnitude das variáveis. Modelos baseados em árvore (Random Forest, Gradient Boosting, XGBoost) são invariantes ao scaling e não necessitam dessa etapa.</p>

<h2>6.5 Análise do Melhor Modelo</h2>

{fig_block('mod_melhor', 19, f'Análise do modelo {met["melhor"]}: diagrama predito vs. real (esq.) e distribuição dos resíduos (dir.)')}

<p>O diagrama predito vs. real (esquerda) mostra alinhamento com a reta de predição perfeita para imóveis na faixa de US$100.000 a US$300.000. Os maiores erros ocorrem em imóveis de alto valor (&gt;US$400.000), onde a variabilidade natural é maior. A distribuição dos resíduos (direita) é aproximadamente normal e centrada em zero — ausência de viés sistemático.</p>

<!-- ═══ 7 REGRAS DE NEGÓCIO ═══ -->
<h1>7 Regras de Negócio</h1>

<h2>7.1 Importância das Features</h2>
<p>A importância das features foi extraída do XGBoost via F-Score — número de vezes que cada feature foi usada como critério de divisão nas árvores:</p>

{code_block(CODE['mod_importance'])}

{fig_block('mod_importance', 20, 'Top 20 features mais importantes segundo o modelo XGBoost (F-Score de importância)')}

<p>As 10 features mais importantes identificadas foram:</p>
<ol>
  {''.join(f"<li><strong>{f}</strong></li>" for f in top10_feats)}
</ol>

<p>A feature <em>QualArea</em> (criada na engenharia, Seção 5.2) tornou-se a mais importante, superando individualmente OverallQual e GrLivArea. Isso demonstra que a combinação multiplicativa de qualidade e área captura melhor a variação de preço do que cada variável isolada.</p>

<h2>7.2 Modelo Simplificado</h2>
<p>Para avaliar a viabilidade de um modelo enxuto com apenas 10 features:</p>

{code_block(CODE['mod_simplificado'])}

{fig_block('mod_simplificado', 21, 'Comparação de RMSE e R² entre o modelo completo (239 features) e o simplificado (10 features)')}

<table>
  <tr><th>Modelo</th><th>Features</th><th>RMSE (USD)</th><th>R²</th></tr>
  <tr><td class="l">XGBoost Completo</td><td class="c">239</td><td class="r">US${met['rmse_completo']:,.0f}</td><td class="c">{met['modelos']['XGBoost']['R2']:.4f}</td></tr>
  <tr><td class="l">XGBoost Simplificado</td><td class="c">10</td><td class="r">US${met['rmse_simplificado']:,.0f}</td><td class="c">{met['r2_simplificado']:.4f}</td></tr>
</table>

<p>O modelo simplificado apresenta aumento de RMSE de {(met['rmse_simplificado']-met['rmse_completo'])/met['rmse_completo']*100:.1f}% em relação ao completo. Com R²&nbsp;=&nbsp;{met['r2_simplificado']:.4f}, ainda explica {met['r2_simplificado']*100:.1f}% da variância do preço. Para aplicações onde interpretabilidade e velocidade são prioritárias, o modelo simplificado representa uma alternativa viável.</p>

<h2>7.3 Características que Mais Agregam Valor Imobiliário</h2>

{fig_block('mod_negocio', 22, 'Correlação de Pearson das principais variáveis com SalePrice — perspectiva de negócio')}

<p>Da análise combinada de importância e correlação, emergem as seguintes regras de negócio:</p>
<ul>
  <li><strong>Qualidade construtiva</strong> é o principal vetor de valor: OverallQual, KitchenQual, BsmtQual figuram entre os top-10 features.</li>
  <li><strong>Área total</strong> (TotalSF, GrLivArea) tem peso determinante, especialmente quando combinada à qualidade (QualArea).</li>
  <li><strong>Infraestrutura de conforto</strong> (ar condicionado central, garagem, lareira) agrega valor mensurável.</li>
  <li><strong>Localização</strong> (zoneamento, bairro) impacta significativamente — confirmado pelo ANOVA com 174 pares de bairros significativamente diferentes.</li>
  <li><strong>Modernidade</strong> (HouseAge, YearBuilt) correlaciona positivamente com o preço.</li>
</ul>

<!-- ═══ 8 CROSS-VALIDATION ═══ -->
<h1>8 Avaliação Final – Cross-Validation</h1>
<p>Para uma estimativa imparcial da capacidade de generalização, foi realizada validação cruzada K-Fold (K&nbsp;=&nbsp;5, shuffle=True, random_state=42). A métrica utilizada foi RMSLE, que penaliza igualmente erros proporcionais em imóveis baratos e caros.</p>

{code_block(CODE['mod_cv'])}

{fig_block('mod_cv', 23, 'Resultados de Cross-Validation (5-Fold) com RMSLE — barras de erro indicam o desvio padrão entre os folds')}

<table>
  <tr><th>Modelo</th><th>RMSLE Médio (5-Fold)</th><th>Desvio Padrão</th><th>Interpretação</th></tr>
  {''.join(f"<tr><td class='l'>{k}</td><td class='c'>{v['mean']:.4f}</td><td class='c'>±{v['std']:.4f}</td><td class='l'>{'Melhor generalização' if k=='XGBoost' else ''}</td></tr>" for k,v in met['cv'].items())}
</table>

<p>O Cross-Validation revela uma inversão significativa: embora o Ridge tenha vencido na validação hold-out (RMSE menor), o <strong>XGBoost obteve o menor RMSLE médio no CV</strong> ({met['cv']['XGBoost']['mean']:.4f}) com menor variância entre folds ({met['cv']['XGBoost']['std']:.4f}), indicando melhor generalização. O Ridge apresentou a maior variância entre folds (σ&nbsp;=&nbsp;{met['cv']['Ridge']['std']:.4f}), sugerindo sensibilidade à partição dos dados.</p>

<div class="destaque"><strong>Recomendação:</strong> Para uso em produção, o XGBoost com 239 features é o modelo recomendado — melhor RMSLE médio no CV (0,1273) e menor variância. O Ridge pode ser preferido quando interpretabilidade e simplicidade são prioritárias.</div>

<!-- ═══ 9 CONCLUSÃO ═══ -->
<h1>9 Conclusão</h1>

{fig_block('mod_dashboard', 24, 'Dashboard final consolidado: RMSE por modelo, R², top 5 features e comparação completo vs. simplificado')}

<h2>9.1 Principais Aprendizados</h2>

<h4>a) A Análise Exploratória Guia a Modelagem</h4>
<p>As hipóteses levantadas na AV1 — correlação forte de GrLivArea, influência do bairro, valorização por recência — foram todas confirmadas estatisticamente na AV2 e operacionalizadas como features na modelagem. A sequência metodológica EDA → Testes → Modelagem evitou a criação de features sem embasamento.</p>

<h4>b) Engenharia de Features Supera Variáveis Brutas</h4>
<p>A feature <em>QualArea</em> (Qualidade × Área), criada a partir de raciocínio de domínio, tornou-se a variável mais importante do XGBoost — superando individualmente OverallQual e GrLivArea. Isso demonstra que combinar variáveis com base no contexto do problema pode criar preditores mais poderosos do que os dados brutos.</p>

<h4>c) Tratamento de NAs com Lógica de Domínio</h4>
<p>Substituir NA por "None" para recursos ausentes (sem piscina, sem lareira) em vez de imputar a mediana preservou a informação real: a ausência do recurso é um dado valioso, não um dado faltante. Imputar a mediana nessas colunas seria um erro de domínio que introduziria ruído no modelo.</p>

<h4>d) Transformação Logarítmica é Essencial</h4>
<p>A AV2 comprovou que SalePrice não é normal (Shapiro-Wilk p&nbsp;=&nbsp;{p_sw:.2e}). A transformação log1p reduziu a assimetria de {skewness_orig} para {skewness_log} e melhorou o RMSE do XGBoost em {(met['rmse_sem_log']-met['rmse_com_log'])/met['rmse_sem_log']*100:.1f}%. Para modelos lineares, o impacto é ainda mais expressivo.</p>

<h4>e) Feature Scaling é Crítico para Modelos Lineares</h4>
<p>O StandardScaler reduziu o RMSE do Lasso significativamente, confirmando que a regularização penaliza coeficientes proporcionalmente à escala das features. Modelos baseados em árvore são invariantes e não necessitam dessa etapa.</p>

<h4>f) Cross-Validation é Mais Confiável que Hold-Out</h4>
<p>O Ridge pareceu o melhor modelo na validação simples, mas o CV revelou que o XGBoost generaliza melhor (RMSLE CV = {met['cv']['XGBoost']['mean']:.4f} vs {met['cv']['Ridge']['mean']:.4f}). Decisões de seleção de modelo devem ser baseadas em CV, não em uma única partição.</p>

<h4>g) Paradoxo de Simpson Alerta para Confundimento</h4>
<p>A análise do grupo "casas reformadas" revelou um Paradoxo de Simpson: globalmente, imóveis não reformados tinham mediana maior, mas dentro de cada era de construção, reformados eram mais valorizados. Esse achado demonstra a importância de controlar variáveis de confundimento antes de concluir relações causais.</p>

<h4>h) Modelo Simplificado é Viável para Negócio</h4>
<p>Com apenas 10 features (4,2% das 239 disponíveis), o modelo simplificado atingiu R²&nbsp;=&nbsp;{met['r2_simplificado']:.4f} — suficiente para aplicações onde interpretabilidade, velocidade e custo de coleta de dados são prioritários.</p>

<h2>9.2 Casos de Uso do Modelo</h2>
<ul>
  <li><strong>Avaliação Automática de Imóveis (AVM):</strong> Estimativa instantânea de preço após preenchimento das características do imóvel pelo usuário em plataformas como Zap Imóveis ou QuintoAndar.</li>
  <li><strong>Análise de Portfólio Imobiliário:</strong> Identificação de imóveis sub ou superavaliados em relação ao preço predito, detectando oportunidades de compra ou venda para fundos de investimento (FIIs).</li>
  <li><strong>Garantia de Crédito Imobiliário:</strong> Suporte à avaliação do valor de garantia em financiamentos, reduzindo custo e tempo da perícia física em bancos e fintechs.</li>
  <li><strong>Impacto de Reforma:</strong> Com a feature <em>Foi_Reformada</em> e a análise por era de construção, é possível estimar o impacto financeiro esperado de uma reforma considerando a idade e localização do imóvel.</li>
</ul>

<h2>9.3 Limitações e Trabalhos Futuros</h2>
<ul>
  <li>Os hiperparâmetros dos modelos de ensemble não foram otimizados via GridSearchCV ou Optuna — potencial de melhora substancial no RMSLE.</li>
  <li>Técnicas de <em>ensemble stacking</em> (combinação de previsões de múltiplos modelos) tendem a superar modelos individuais neste tipo de problema.</li>
  <li>A análise de outliers (imóveis com área muito grande vendidos abaixo do esperado) pode ser aprofundada, com exclusão criteriosa baseada em regras de negócio.</li>
  <li>A validação do modelo em dados de outras cidades ou períodos testaria sua capacidade de generalização geográfica e temporal.</li>
</ul>

<hr>

<!-- ═══ REFERÊNCIAS ═══ -->
<h1>Referências</h1>

<p class="ref">DE COCK, Dean. <strong>Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project</strong>. Journal of Statistics Education, v. 19, n. 3, 2011. DOI: 10.1080/10691898.2011.11889627.</p>

<p class="ref">PEDREGOSA, F. et al. <strong>Scikit-learn: Machine Learning in Python</strong>. Journal of Machine Learning Research, v. 12, p. 2825–2830, 2011.</p>

<p class="ref">CHEN, Tianqi; GUESTRIN, Carlos. <strong>XGBoost: A Scalable Tree Boosting System</strong>. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York: ACM, 2016. p. 785–794.</p>

<p class="ref">HASTIE, Trevor; TIBSHIRANI, Robert; FRIEDMAN, Jerome. <strong>The Elements of Statistical Learning: Data Mining, Inference, and Prediction</strong>. 2. ed. New York: Springer, 2009.</p>

<p class="ref">SEABOLD, Skipper; PERKTOLD, Josef. <strong>Statsmodels: Econometric and Statistical Modeling with Python</strong>. In: Proceedings of the 9th Python in Science Conference. 2010. p. 57–61.</p>

<p class="ref">KAGGLE. <strong>House Prices – Advanced Regression Techniques</strong>. Disponível em: https://www.kaggle.com/c/house-prices-advanced-regression-techniques. Acesso em: 27 maio 2025.</p>

<p class="ref">ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. <strong>NBR 6023</strong>: informação e documentação – referências – elaboração. Rio de Janeiro: ABNT, 2018.</p>

<p class="ref">ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. <strong>NBR 14724</strong>: informação e documentação – trabalhos acadêmicos – apresentação. Rio de Janeiro: ABNT, 2011.</p>

</div>
</body>
</html>'''

with open('outputs/relatorio_abnt_completo.html', 'w', encoding='utf-8') as f:
    f.write(html)

size = os.path.getsize('outputs/relatorio_abnt_completo.html') / 1024 / 1024
print(f"\n✓ Relatório gerado: outputs/relatorio_abnt_completo.html ({size:.1f} MB)")
print(f"  Total de figuras embutidas: {len(all_figs)}")
