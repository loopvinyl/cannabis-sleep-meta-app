import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io

# ----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO INICIAL DA PÁGINA E IDIOMA
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Meta-Analysis App - Sleep & Cannabis", layout="wide")

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

# Dicionários de tradução
TEXTS = {
    "en": {
        "title": "🗂️ Interactive Meta-Analysis: THC/Cannabis & Sleep",
        "subtitle": "Based on the methodology of Tait et al. (2026) - Quality of Life Research",
        "lang_label": "Language / Idioma",
        "lang_options": ["English", "Português"],
        "select_studies": "📊 Select Studies to Include",
        "include": "Include",
        "study": "Study",
        "type": "Type",
        "n": "N",
        "d": "Cohen's d",
        "se": "Std. Error",
        "weight": "Weight (%)",
        "results": "📈 Meta-Analysis Results",
        "pooled_effect": "Pooled Effect (Cohen's d)",
        "ci": "95% Confidence Interval",
        "p_value": "P-value",
        "heterogeneity": "Heterogeneity",
        "i2": "I²",
        "tau2": "Tau²",
        "q_stat": "Q-statistic",
        "forest_plot": "🌲 Forest Plot",
        "funnel_plot": "🌀 Funnel Plot",
        "sensitivity_plot": "🔍 Sensitivity Analysis (Leave-One-Out)",
        "bar_plot": "📊 Effect Sizes Bar Plot",
        "info_sidebar": "ℹ️ About this App",
        "sidebar_text": "Select/deselect studies on the left to see how the pooled effect changes. The app uses a Random-Effects model (DerSimonian-Laird).\n\nFormatted for:",
        "download_hint": "Right-click on the plot to save it.",
        "method_title": "⚙️ Statistical Method",
        "method_text": "**DerSimonian-Laird (DL)** random-effects model.\n\nQ-statistic (Heterogeneity): **{q:.2f}**\n\nWeights are calculated based on within-study variance + between-studies variance (Tau²).",
        "interpretation_title": "📖 Interpretation Guide",
        "cohen_d_interpret": "**Cohen's d = {:.3f}** — This is a **{}** effect size ({}).",
        "i2_interpret": "**I² = {:.1f}%** — {} heterogeneity.",
        "p_value_interpret": "**p = {}** — The pooled effect is {} significant.",
        "hetero_low": "low",
        "hetero_moderate": "moderate",
        "hetero_high": "high",
        "significant": "statistically",
        "not_significant": "not statistically",
        "effect_small": "small",
        "effect_moderate": "moderate",
        "effect_large": "large",
        "effect_very_large": "very large",
        "effect_size_guide": "**Effect size classification:** d < 0.2 = negligible; 0.2–0.5 = small; 0.5–0.8 = moderate; > 0.8 = large.",
        "forest_explanation": "**Forest Plot:** Each square represents a study's effect size (Cohen's d), with horizontal lines indicating the 95% confidence interval. The size of the square reflects the study's weight in the meta-analysis. The diamond at the bottom shows the pooled effect and its confidence interval.",
        "funnel_explanation": "**Funnel Plot:** Used to detect publication bias. In the absence of bias, studies should be symmetrically distributed around the pooled effect. Asymmetry may suggest missing studies (e.g., due to publication bias).",
        "sensitivity_explanation": "**Sensitivity Analysis (Leave-One-Out):** This plot shows the pooled effect after removing each study one at a time. If the overall effect changes substantially when a particular study is removed, that study may have a disproportionate influence on the results.",
        "bar_explanation": "**Bar Plot of Effect Sizes:** Displays each study's Cohen's d with its 95% confidence interval. This helps visualize the variation in effects across studies.",
        "exclude_label": "Excluded Studies",
        "excluded_note": "No studies excluded. All selected.",
        "ci_label": "CI: [{:.3f}, {:.3f}]",
        "leave_one_out_label": "Leave-One-Out Pooled Effect",
        "studies_excluded": "Studies excluded"
    },
    "pt": {
        "title": "🗂️ Meta-Análise Interativa: THC/Cannabis & Sono",
        "subtitle": "Baseado na metodologia de Tait et al. (2026) - Quality of Life Research",
        "lang_label": "Idioma / Language",
        "lang_options": ["Português", "English"],
        "select_studies": "📊 Selecione os Estudos para Incluir",
        "include": "Incluir",
        "study": "Estudo",
        "type": "Tipo",
        "n": "N",
        "d": "d de Cohen",
        "se": "Erro Padrão",
        "weight": "Peso (%)",
        "results": "📈 Resultados da Meta-Análise",
        "pooled_effect": "Efeito Combinado (d de Cohen)",
        "ci": "Intervalo de Confiança 95%",
        "p_value": "Valor-p",
        "heterogeneity": "Heterogeneidade",
        "i2": "I²",
        "tau2": "Tau²",
        "q_stat": "Estatística Q",
        "forest_plot": "🌲 Forest Plot",
        "funnel_plot": "🌀 Funnel Plot",
        "sensitivity_plot": "🔍 Análise de Sensibilidade (Leave-One-Out)",
        "bar_plot": "📊 Gráfico de Barras dos Tamanhos de Efeito",
        "info_sidebar": "ℹ️ Sobre este App",
        "sidebar_text": "Selecione/desselecione os estudos à esquerda para ver como o efeito combinado muda. O app usa o modelo de Efeitos Aleatórios (DerSimonian-Laird).\n\nFormatado para:",
        "download_hint": "Clique com o botão direito no gráfico para salvá-lo.",
        "method_title": "⚙️ Método Estatístico",
        "method_text": "Modelo de efeitos aleatórios **DerSimonian-Laird (DL)**.\n\nEstatística Q (Heterogeneidade): **{q:.2f}**\n\nOs pesos são calculados com base na variância intra-estudo + variância entre estudos (Tau²).",
        "interpretation_title": "📖 Guia de Interpretação",
        "cohen_d_interpret": "**d de Cohen = {:.3f}** — Este é um tamanho de efeito **{}** ({}).",
        "i2_interpret": "**I² = {:.1f}%** — Heterogeneidade **{}**.",
        "p_value_interpret": "**p = {}** — O efeito combinado é {} significativo.",
        "hetero_low": "baixa",
        "hetero_moderate": "moderada",
        "hetero_high": "alta",
        "significant": "estatisticamente",
        "not_significant": "não estatisticamente",
        "effect_small": "pequeno",
        "effect_moderate": "moderado",
        "effect_large": "grande",
        "effect_very_large": "muito grande",
        "effect_size_guide": "**Classificação do tamanho de efeito:** d < 0,2 = desprezível; 0,2–0,5 = pequeno; 0,5–0,8 = moderado; > 0,8 = grande.",
        "forest_explanation": "**Forest Plot:** Cada quadrado representa o tamanho de efeito de um estudo (d de Cohen), com linhas horizontais indicando o intervalo de confiança de 95%. O tamanho do quadrado reflete o peso do estudo na meta-análise. O diamante na parte inferior mostra o efeito combinado e seu intervalo de confiança.",
        "funnel_explanation": "**Funnel Plot:** Usado para detectar viés de publicação. Na ausência de viés, os estudos devem estar simetricamente distribuídos em torno do efeito combinado. Assimetria pode sugerir estudos faltantes (ex: devido a viés de publicação).",
        "sensitivity_explanation": "**Análise de Sensibilidade (Leave-One-Out):** Este gráfico mostra o efeito combinado após remover cada estudo um de cada vez. Se o efeito geral mudar substancialmente quando um estudo particular é removido, esse estudo pode ter influência desproporcional nos resultados.",
        "bar_explanation": "**Gráfico de Barras dos Tamanhos de Efeito:** Exibe o d de Cohen de cada estudo com seu intervalo de confiança de 95%. Isso ajuda a visualizar a variação dos efeitos entre os estudos.",
        "exclude_label": "Estudos Excluídos",
        "excluded_note": "Nenhum estudo excluído. Todos selecionados.",
        "ci_label": "IC: [{:.3f}, {:.3f}]",
        "leave_one_out_label": "Efeito Combinado Leave-One-Out",
        "studies_excluded": "Estudos excluídos"
    }
}

def t(key, **kwargs):
    lang = st.session_state.lang
    text = TEXTS[lang].get(key, TEXTS["en"].get(key, key))
    if kwargs:
        return text.format(**kwargs)
    return text

def fmt_num(value, decimals=3):
    lang = st.session_state.lang
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        formatted = f"{value:.{decimals}f}"
        if lang == "pt":
            formatted = formatted.replace(".", ",")
        return formatted
    return str(value)

# ----------------------------------------------------------------------------
# 2. BASE DE DADOS DOS ESTUDOS (10 ESTUDOS)
# ----------------------------------------------------------------------------
STUDIES = [
    {"id": "Erridge et al. (2026)", "type": "Coorte", "n": 8945, "d": 0.84, "se": 0.009},
    {"id": "Pakdee, Sribunrieng e Poowanna (2026)", "type": "RCT", "n": 20, "d": 2.84, "se": 0.635},
    {"id": "Datta et al. (2025a)", "type": "Coorte", "n": 517, "d": 0.68, "se": 0.030},
    {"id": "Datta et al. (2025b)", "type": "Coorte", "n": 269, "d": 0.32, "se": 0.020},
    {"id": "Short et al. (2025)", "type": "Coorte", "n": 137, "d": 0.80, "se": 0.068},
    {"id": "Vivek et al. (2024)", "type": "Coorte", "n": 40, "d": 1.35, "se": 0.213},
    {"id": "Cooke et al. (2023)", "type": "Coorte", "n": 163, "d": 0.55, "se": 0.043},
    {"id": "Vaddiparti et al. (2023)", "type": "Coorte", "n": 15, "d": 1.44, "se": 0.372},
    {"id": "Ried et al. (2023)", "type": "RCT", "n": 29, "d": 0.69, "se": 0.128},
    {"id": "Montebello et al. (2022)", "type": "RCT", "n": 128, "d": 0.60, "se": 0.181},
]

# ----------------------------------------------------------------------------
# 3. FUNÇÃO DE META-ANÁLISE (DerSimonian-Laird)
# ----------------------------------------------------------------------------
def run_meta_analysis(df):
    if df.empty:
        return None
    d = df['d'].values
    se = df['se'].values
    k = len(d)
    w = 1 / (se ** 2)
    sum_w = np.sum(w)
    sum_wd = np.sum(w * d)
    sum_wd2 = np.sum(w * (d ** 2))
    Q = sum_wd2 - (sum_wd ** 2) / sum_w
    df_het = k - 1
    if df_het > 0:
        tau2 = max(0, (Q - df_het) / (sum_w - (np.sum(w ** 2) / sum_w)))
    else:
        tau2 = 0
    w_star = 1 / (se ** 2 + tau2)
    sum_w_star = np.sum(w_star)
    pooled_d = np.sum(w_star * d) / sum_w_star
    se_pooled = np.sqrt(1 / sum_w_star)
    ci_lb = pooled_d - 1.96 * se_pooled
    ci_ub = pooled_d + 1.96 * se_pooled
    z = pooled_d / se_pooled
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    i2 = max(0, ((Q - df_het) / Q) * 100) if Q > 0 else 0
    weights_percent = (w_star / sum_w_star) * 100
    return {
        "k": k, "pooled_d": pooled_d, "se_pooled": se_pooled,
        "ci_lb": ci_lb, "ci_ub": ci_ub, "p_val": p_val,
        "tau2": tau2, "i2": i2, "q": Q, "df": df_het,
        "weights": weights_percent, "w_star": w_star
    }

# ----------------------------------------------------------------------------
# 4. FUNÇÕES DE GRÁFICOS
# ----------------------------------------------------------------------------
def plot_forest(df, results, lang):
    if df.empty or results is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    df_sorted = df.copy()
    df_sorted['weight'] = results['weights']
    df_sorted = df_sorted.sort_values('d', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    d_values = df_sorted['d'].values
    se_values = df_sorted['se'].values
    ci_lower = d_values - 1.96 * se_values
    ci_upper = d_values + 1.96 * se_values
    ax.hlines(y=y_pos, xmin=ci_lower, xmax=ci_upper, color='gray', linewidth=1, alpha=0.7)
    sizes = (df_sorted['weight'] / df_sorted['weight'].max()) * 100 + 20
    ax.scatter(d_values, y_pos, s=sizes, color='#1f77b4', zorder=5, edgecolors='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['id'].tolist(), fontsize=9)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=results['pooled_d'], color='red', linestyle='-', linewidth=1.5, alpha=0.6)
    x_min = min(-0.5, ci_lower.min() - 0.2) if len(ci_lower) > 0 else -0.5
    x_max = max(2.5, ci_upper.max() + 0.2) if len(ci_upper) > 0 else 2.5
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Cohen's d" if lang == 'en' else "d de Cohen", fontsize=10)
    ax.set_title("Forest Plot", fontsize=12)
    diamond_y = -0.5
    diamond_x = results['pooled_d']
    ax.plot([results['ci_lb'], diamond_x, results['ci_ub'], diamond_x, results['ci_lb']],
            [diamond_y, diamond_y - 0.2, diamond_y, diamond_y + 0.2, diamond_y],
            color='red', linewidth=2)
    ax.text(diamond_x, diamond_y - 0.5, 
            f"Pooled: {fmt_num(results['pooled_d'], 3)} [{fmt_num(results['ci_lb'], 3)}, {fmt_num(results['ci_ub'], 3)}]",
            ha='center', fontsize=9, color='red', fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_funnel(df, results, lang):
    if df.empty or results is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    d = df['d'].values
    se = df['se'].values
    ax.scatter(d, se, color='#1f77b4', edgecolors='black', zorder=5)
    # Linha do efeito combinado
    ax.axvline(x=results['pooled_d'], color='red', linestyle='-', linewidth=1.5, alpha=0.6)
    # Limites do funil (assumindo distribuição normal)
    x_vals = np.linspace(results['ci_lb'] - 0.5, results['ci_ub'] + 0.5, 100)
    # Curvas do funil para 95% CI (1.96 * se)
    se_max = max(se) * 1.5
    y_vals = np.linspace(0.01, se_max, 100)
    x_upper = results['pooled_d'] + 1.96 * y_vals
    x_lower = results['pooled_d'] - 1.96 * y_vals
    ax.fill_betweenx(y_vals, x_lower, x_upper, color='lightgray', alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d" if lang == 'en' else "d de Cohen", fontsize=10)
    ax.set_ylabel("Standard Error" if lang == 'en' else "Erro Padrão", fontsize=10)
    ax.set_title("Funnel Plot", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_sensitivity(df, results, lang):
    if df.empty or results is None or len(df) < 3:
        return None
    # Leave-one-out
    studies = df['id'].tolist()
    pooled_effects = []
    for i in range(len(df)):
        df_loo = df.drop(i)
        res_loo = run_meta_analysis(df_loo)
        if res_loo:
            pooled_effects.append(res_loo['pooled_d'])
        else:
            pooled_effects.append(None)
    # Remover Nones
    valid = [(s, e) for s, e in zip(studies, pooled_effects) if e is not None]
    if not valid:
        return None
    studies_loo, effects_loo = zip(*valid)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(studies_loo))
    ax.scatter(effects_loo, y_pos, color='#ff7f0e', s=60, edgecolors='black', zorder=5)
    ax.axvline(x=results['pooled_d'], color='red', linestyle='--', linewidth=1.5, label='Overall effect')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s for s in studies_loo], fontsize=8)
    ax.set_xlabel("Pooled Cohen's d (without study)" if lang == 'en' else "d de Cohen combinado (sem estudo)", fontsize=10)
    ax.set_title("Sensitivity Analysis (Leave-One-Out)" if lang == 'en' else "Análise de Sensibilidade (Leave-One-Out)", fontsize=12)
    ax.legend()
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_bar_effects(df, lang):
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    df_sorted = df.sort_values('d', ascending=True)
    y_pos = np.arange(len(df_sorted))
    d_values = df_sorted['d'].values
    ci_lower = d_values - 1.96 * df_sorted['se'].values
    ci_upper = d_values + 1.96 * df_sorted['se'].values
    ax.barh(y_pos, d_values, xerr=(d_values - ci_lower, ci_upper - d_values), capsize=3, color='#2ca02c', edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['id'].tolist(), fontsize=8)
    ax.set_xlabel("Cohen's d" if lang == 'en' else "d de Cohen", fontsize=10)
    ax.set_title("Effect Sizes with 95% CI" if lang == 'en' else "Tamanhos de Efeito com IC 95%", fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

# ----------------------------------------------------------------------------
# 5. FUNÇÃO PARA INTERPRETAÇÃO
# ----------------------------------------------------------------------------
def interpret_results(results, lang):
    if results is None:
        return ""
    d = results['pooled_d']
    i2 = results['i2']
    p = results['p_val']
    # Classificação do d de Cohen
    if abs(d) < 0.2:
        effect_size_text = t('effect_small') + " (negligible)" if lang == 'en' else t('effect_small') + " (desprezível)"
        effect_desc = "small"
    elif abs(d) < 0.5:
        effect_size_text = t('effect_small')
        effect_desc = "small"
    elif abs(d) < 0.8:
        effect_size_text = t('effect_moderate')
        effect_desc = "moderate"
    elif abs(d) < 1.3:
        effect_size_text = t('effect_large')
        effect_desc = "large"
    else:
        effect_size_text = t('effect_very_large')
        effect_desc = "very large"
    # Heterogeneidade
    if i2 < 25:
        hetero_text = t('hetero_low')
    elif i2 < 50:
        hetero_text = t('hetero_moderate')
    else:
        hetero_text = t('hetero_high')
    # Significância
    if p < 0.05:
        sig_text = t('significant')
    else:
        sig_text = t('not_significant')
    
    cohen_line = t('cohen_d_interpret', d).format(d, effect_desc, effect_size_text)
    i2_line = t('i2_interpret', i2).format(i2, hetero_text)
    p_line = t('p_value_interpret', p).format(fmt_num(p, 3), sig_text)
    guide_line = t('effect_size_guide')
    
    return f"{cohen_line}\n\n{i2_line}\n\n{p_line}\n\n{guide_line}"

# ----------------------------------------------------------------------------
# 6. INTERFACE STREAMLIT
# ----------------------------------------------------------------------------
def main():
    lang = st.session_state.lang
    
    # Sidebar
    with st.sidebar:
        st.header(t('lang_label'))
        lang_choice = st.radio(
            "",
            options=t('lang_options'),
            index=0 if lang == 'en' else 1,
            label_visibility="collapsed"
        )
        if lang_choice == "Português" and lang != "pt":
            st.session_state.lang = "pt"
            st.rerun()
        elif lang_choice == "English" and lang != "en":
            st.session_state.lang = "en"
            st.rerun()
        
        st.divider()
        st.subheader(t('info_sidebar'))
        st.info(t('sidebar_text') + f" **{lang.upper()}**")
        st.caption("Developed for academic research purposes.")
        
        st.divider()
        st.subheader(t('method_title'))
        st.markdown(t('method_text', q=0.0))
    
    # Título principal
    st.title(t('title'))
    st.caption(t('subtitle'))
    
    # --------------------------------------------------------------------
    # SELEÇÃO DE ESTUDOS
    # --------------------------------------------------------------------
    df_studies = pd.DataFrame(STUDIES)
    
    # Inicializar estados (Pakdee desmarcado)
    for idx, row in df_studies.iterrows():
        key = f"include_{idx}"
        if key not in st.session_state:
            if row['id'] == "Pakdee, Sribunrieng e Poowanna (2026)":
                st.session_state[key] = False
            else:
                st.session_state[key] = True
    
    st.subheader(t('select_studies'))
    
    col1, col2, col3, col4, col5, col6 = st.columns([0.6, 3.5, 1.5, 1, 1.5, 1.5])
    col1.markdown("<div style='text-align: center; font-weight: bold;'>" + t('include') + "</div>", unsafe_allow_html=True)
    col2.markdown("<div style='text-align: left; font-weight: bold;'>" + t('study') + "</div>", unsafe_allow_html=True)
    col3.markdown("<div style='text-align: center; font-weight: bold;'>" + t('type') + "</div>", unsafe_allow_html=True)
    col4.markdown("<div style='text-align: center; font-weight: bold;'>" + t('n') + "</div>", unsafe_allow_html=True)
    col5.markdown("<div style='text-align: center; font-weight: bold;'>" + t('d') + "</div>", unsafe_allow_html=True)
    col6.markdown("<div style='text-align: center; font-weight: bold;'>" + t('se') + "</div>", unsafe_allow_html=True)
    
    selected_ids = []
    for idx, row in df_studies.iterrows():
        key = f"include_{idx}"
        col1, col2, col3, col4, col5, col6 = st.columns([0.6, 3.5, 1.5, 1, 1.5, 1.5])
        checked = col1.checkbox(label="", value=st.session_state[key], key=key, label_visibility="collapsed")
        if checked:
            selected_ids.append(idx)
        col2.write(row['id'])
        col3.write(row['type'])
        col4.write(f"{row['n']:,}")
        col5.write(fmt_num(row['d'], 2))
        col6.write(fmt_num(row['se'], 3))
    
    df_selected = df_studies.loc[selected_ids].copy()
    st.caption(f"**{len(df_selected)}** de **{len(df_studies)}** estudos selecionados.")
    
    # --------------------------------------------------------------------
    # EXECUTAR META-ANÁLISE
    # --------------------------------------------------------------------
    if len(df_selected) >= 2:
        results = run_meta_analysis(df_selected)
    else:
        results = None
        st.warning("Selecione pelo menos **2 estudos** para realizar a meta-análise.")
    
    if results:
        # Atualizar sidebar com resultados atuais
        with st.sidebar:
            st.subheader("📊 Current Results")
            st.metric(t('q_stat'), f"{fmt_num(results['q'], 2)}")
            st.metric(t('i2'), f"{fmt_num(results['i2'], 1)} %")
            st.metric(t('tau2'), f"{fmt_num(results['tau2'], 4)}")
        
        # --------------------------------------------------------------------
        # RESULTADOS PRINCIPAIS E INTERPRETAÇÃO
        # --------------------------------------------------------------------
        st.divider()
        st.subheader(t('results'))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label=t('pooled_effect'), value=fmt_num(results['pooled_d'], 3))
        with col2:
            ci_text = f"[{fmt_num(results['ci_lb'], 3)}, {fmt_num(results['ci_ub'], 3)}]"
            st.metric(label=t('ci'), value=ci_text)
        with col3:
            if results['p_val'] < 0.001:
                p_text = "< 0.001" if lang == 'en' else "< 0,001"
            else:
                p_text = fmt_num(results['p_val'], 3)
            st.metric(label=t('p_value'), value=p_text)
        with col4:
            st.metric(label=t('i2'), value=f"{fmt_num(results['i2'], 1)} %")
        
        # Interpretação
        st.divider()
        st.subheader(t('interpretation_title'))
        interpretation = interpret_results(results, lang)
        st.markdown(interpretation)
        
        # Tabela de pesos
        st.divider()
        st.subheader("📋 Study Weights")
        df_weights = df_selected.copy()
        df_weights['Weight (%)'] = [fmt_num(w, 2) for w in results['weights']]
        rename_map = {
            'id': t('study'), 
            'type': t('type'), 
            'n': t('n'), 
            'd': t('d'), 
            'se': t('se'),
            'Weight (%)': t('weight')
        }
        df_display = df_weights[['id', 'type', 'n', 'd', 'se', 'Weight (%)']].rename(columns=rename_map)
        # Formatação
        df_display[t('n')] = df_display[t('n')].apply(lambda x: f"{x:,}")
        df_display[t('d')] = df_display[t('d')].apply(lambda x: fmt_num(x, 2))
        df_display[t('se')] = df_display[t('se')].apply(lambda x: fmt_num(x, 3))
        
        st.dataframe(df_display, use_container_width=True)
        
        # --------------------------------------------------------------------
        # GRÁFICOS E EXPLICAÇÕES
        # --------------------------------------------------------------------
        # 1. Forest Plot
        st.divider()
        st.subheader(t('forest_plot'))
        fig_forest = plot_forest(df_selected, results, lang)
        if fig_forest:
            st.pyplot(fig_forest)
            st.caption(t('download_hint'))
        st.markdown(t('forest_explanation'))
        
        # 2. Funnel Plot
        st.divider()
        st.subheader(t('funnel_plot'))
        fig_funnel = plot_funnel(df_selected, results, lang)
        if fig_funnel:
            st.pyplot(fig_funnel)
            st.caption(t('download_hint'))
        st.markdown(t('funnel_explanation'))
        
        # 3. Bar Plot
        st.divider()
        st.subheader(t('bar_plot'))
        fig_bar = plot_bar_effects(df_selected, lang)
        if fig_bar:
            st.pyplot(fig_bar)
            st.caption(t('download_hint'))
        st.markdown(t('bar_explanation'))
        
        # 4. Sensitivity Plot (Leave-One-Out)
        st.divider()
        st.subheader(t('sensitivity_plot'))
        fig_sens = plot_sensitivity(df_selected, results, lang)
        if fig_sens:
            st.pyplot(fig_sens)
            st.caption(t('download_hint'))
            st.markdown(t('sensitivity_explanation'))
        else:
            st.info("Precisa de pelo menos 3 estudos para análise de sensibilidade." if lang == 'en' else "Precisa de pelo menos 3 estudos para análise de sensibilidade.")
    
    else:
        st.info("👈 Selecione mais estudos na tabela acima.")

if __name__ == "__main__":
    main()
