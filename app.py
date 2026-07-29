import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
import textwrap

# ----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO INICIAL DA PÁGINA E IDIOMA
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Meta-Analysis App - Sleep & Cannabis", layout="wide")

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

# ----------------------------------------------------------------------------
# 2. DICIONÁRIO DE TRADUÇÃO
# ----------------------------------------------------------------------------
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
        "funnel_plot": "🔍 Funnel Plot (Publication Bias)",
        "sensitivity_plot": "📉 Sensitivity Analysis (Leave-One-Out)",
        "info_sidebar": "ℹ️ About this App",
        "sidebar_text": "Select/deselect studies on the left to see how the pooled effect changes. The app uses a Random-Effects model (DerSimonian-Laird).\n\nFormatted for:",
        "download_hint": "Right-click on the plot to save it.",
        "method_title": "⚙️ Statistical Method",
        "method_text": "**DerSimonian-Laird (DL)** random-effects model.\n\nQ-statistic (Heterogeneity): **{q:.2f}**\n\nWeights are calculated based on within-study variance + between-studies variance (Tau²).",
        "interpretation_title": "📖 Interpretation Guide",
        "cohen_d_interpret": "**Cohen's d = {:.3f}** – This is a **{}** effect size. It indicates that the THC/cannabis intervention improved sleep quality by {:.1f} standard deviations compared to control/baseline.",
        "effect_size_small": "small",
        "effect_size_moderate": "moderate",
        "effect_size_large": "large",
        "i2_interpret": "**I² = {:.1f}%** – This indicates **{}** heterogeneity. {}",
        "heterogeneity_low": "low",
        "heterogeneity_moderate": "moderate",
        "heterogeneity_high": "high",
        "heterogeneity_low_text": "The studies are relatively consistent, suggesting that the pooled effect is reliable.",
        "heterogeneity_moderate_text": "There is some variability among studies, but the pooled effect remains informative.",
        "heterogeneity_high_text": "There is substantial variability among studies. This may reflect differences in populations, doses, or outcome measures. A random-effects model was used to account for this.",
        "p_value_interpret": "**p-value {}** – The result is {}.",
        "p_significant": "statistically significant (p < 0.05)",
        "p_not_significant": "not statistically significant (p ≥ 0.05)",
        "ci_interpret": "The 95% confidence interval [{:.3f}, {:.3f}] {} contain zero, {}.",
        "ci_does_not": "does not",
        "ci_does": "does",
        "ci_support": "supporting a significant effect",
        "ci_not_support": "suggesting no significant effect",
        "funnel_interpret": "The funnel plot shows the distribution of study effects against their standard errors. Asymmetry may indicate publication bias.",
        "sensitivity_interpret": "This plot shows the pooled effect after removing each study one at a time. If the effect remains stable, the result is robust.",
        "weight_table": "📋 Study Weights",
        "weight_info": "Larger squares indicate greater weight in the meta-analysis."
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
        "funnel_plot": "🔍 Gráfico de Funil (Viés de Publicação)",
        "sensitivity_plot": "📉 Análise de Sensibilidade (Leave-One-Out)",
        "info_sidebar": "ℹ️ Sobre este App",
        "sidebar_text": "Selecione/desselecione os estudos à esquerda para ver como o efeito combinado muda. O app usa o modelo de Efeitos Aleatórios (DerSimonian-Laird).\n\nFormatado para:",
        "download_hint": "Clique com o botão direito no gráfico para salvá-lo.",
        "method_title": "⚙️ Método Estatístico",
        "method_text": "Modelo de efeitos aleatórios **DerSimonian-Laird (DL)**.\n\nEstatística Q (Heterogeneidade): **{q:.2f}**\n\nOs pesos são calculados com base na variância intra-estudo + variância entre estudos (Tau²).",
        "interpretation_title": "📖 Guia de Interpretação",
        "cohen_d_interpret": "**d de Cohen = {:.3f}** – Este é um efeito **{}**. Indica que a intervenção com THC/cannabis melhorou a qualidade do sono em {:.1f} desvios padrão em comparação com o controle/baseline.",
        "effect_size_small": "pequeno",
        "effect_size_moderate": "moderado",
        "effect_size_large": "grande",
        "i2_interpret": "**I² = {:.1f}%** – Isso indica heterogeneidade **{}**. {}",
        "heterogeneity_low": "baixa",
        "heterogeneity_moderate": "moderada",
        "heterogeneity_high": "alta",
        "heterogeneity_low_text": "Os estudos são relativamente consistentes, sugerindo que o efeito combinado é confiável.",
        "heterogeneity_moderate_text": "Há alguma variabilidade entre os estudos, mas o efeito combinado permanece informativo.",
        "heterogeneity_high_text": "Há variabilidade substancial entre os estudos. Isso pode refletir diferenças nas populações, doses ou medidas de desfecho. Um modelo de efeitos aleatórios foi usado para considerar isso.",
        "p_value_interpret": "**Valor-p {}** – O resultado é {}.",
        "p_significant": "estatisticamente significativo (p < 0,05)",
        "p_not_significant": "não estatisticamente significativo (p ≥ 0,05)",
        "ci_interpret": "O intervalo de confiança 95% [{:.3f}, {:.3f}] {} contém zero, {}.",
        "ci_does_not": "não",
        "ci_does": "sim",
        "ci_support": "apoiando um efeito significativo",
        "ci_not_support": "sugerindo que não há efeito significativo",
        "funnel_interpret": "O gráfico de funil mostra a distribuição dos efeitos dos estudos contra seus erros padrão. Assimetria pode indicar viés de publicação.",
        "sensitivity_interpret": "Este gráfico mostra o efeito combinado após remover cada estudo um por vez. Se o efeito permanece estável, o resultado é robusto.",
        "weight_table": "📋 Pesos dos Estudos",
        "weight_info": "Quadrados maiores indicam maior peso na meta-análise."
    }
}

# ----------------------------------------------------------------------------
# 3. FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------------
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
# 4. BASE DE DADOS (10 ESTUDOS)
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
# 5. FUNÇÃO DE META-ANÁLISE (DerSimonian-Laird)
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
        "k": k,
        "pooled_d": pooled_d,
        "se_pooled": se_pooled,
        "ci_lb": ci_lb,
        "ci_ub": ci_ub,
        "p_val": p_val,
        "tau2": tau2,
        "i2": i2,
        "q": Q,
        "df": df_het,
        "weights": weights_percent,
        "w_star": w_star
    }

# ----------------------------------------------------------------------------
# 6. FUNÇÕES DE GRÁFICOS
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
    fig, ax = plt.subplots(figsize=(8, 6))
    d = df['d'].values
    se = df['se'].values
    ax.scatter(d, se, color='#1f77b4', zorder=5, edgecolors='black', linewidth=0.5)
    ax.set_xlabel("Cohen's d" if lang == 'en' else "d de Cohen", fontsize=10)
    ax.set_ylabel("Standard Error" if lang == 'en' else "Erro Padrão", fontsize=10)
    ax.set_title("Funnel Plot" if lang == 'en' else "Gráfico de Funil", fontsize=12)
    # Linha vertical no efeito combinado
    ax.axvline(x=results['pooled_d'], color='red', linestyle='-', linewidth=1.0, alpha=0.5)
    # Triângulo esperado (limites)
    x_limits = np.linspace(results['ci_lb'] - 0.5, results['ci_ub'] + 0.5, 100)
    y_limits = (np.max(se) / (results['ci_ub'] - results['ci_lb'] + 1)) * np.abs(x_limits - results['pooled_d'])
    ax.fill_between(x_limits, 0, y_limits, color='gray', alpha=0.1)
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_sensitivity(df, results, lang):
    if df.empty or results is None or len(df) < 2:
        return None
    d_orig = results['pooled_d']
    studies = df['id'].tolist()
    d_loo = []
    for i in range(len(df)):
        # CORREÇÃO: usar df.index[i] para pegar o índice real
        df_loo = df.drop(df.index[i])
        res_loo = run_meta_analysis(df_loo)
        if res_loo is not None:
            d_loo.append(res_loo['pooled_d'])
        else:
            d_loo.append(np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(studies))
    ax.scatter(d_loo, y_pos, color='#1f77b4', zorder=5, edgecolors='black', linewidth=0.5)
    ax.axvline(x=d_orig, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Original pooled d')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(studies, fontsize=8)
    ax.set_xlabel("Cohen's d (leave-one-out)" if lang == 'en' else "d de Cohen (leave-one-out)", fontsize=10)
    ax.set_title("Sensitivity Analysis (Leave-One-Out)" if lang == 'en' else "Análise de Sensibilidade (Leave-One-Out)", fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

# ----------------------------------------------------------------------------
# 7. FUNÇÃO DE INTERPRETAÇÃO
# ----------------------------------------------------------------------------
def interpret_results(results, lang):
    if results is None:
        return "No results to interpret."
    
    d = results['pooled_d']
    ci_lb = results['ci_lb']
    ci_ub = results['ci_ub']
    p = results['p_val']
    i2 = results['i2']
    
    if abs(d) < 0.2:
        effect_desc = t('effect_size_small')
    elif abs(d) < 0.5:
        effect_desc = t('effect_size_moderate')
    else:
        effect_desc = t('effect_size_large')
    
    cohen_line = t('cohen_d_interpret').format(d, effect_desc, abs(d))
    
    if i2 < 25:
        het_desc = t('heterogeneity_low')
        het_text = t('heterogeneity_low_text')
    elif i2 < 50:
        het_desc = t('heterogeneity_moderate')
        het_text = t('heterogeneity_moderate_text')
    else:
        het_desc = t('heterogeneity_high')
        het_text = t('heterogeneity_high_text')
    i2_line = t('i2_interpret').format(i2, het_desc, het_text)
    
    if p < 0.05:
        p_line = t('p_value_interpret').format(f"= {fmt_num(p, 3)}", t('p_significant'))
    else:
        p_line = t('p_value_interpret').format(f"= {fmt_num(p, 3)}", t('p_not_significant'))
    
    if ci_lb < 0 < ci_ub:
        ci_does = t('ci_does')
        ci_support_text = t('ci_not_support')
    else:
        ci_does = t('ci_does_not')
        ci_support_text = t('ci_support')
    ci_line = t('ci_interpret').format(ci_lb, ci_ub, ci_does, ci_support_text)
    
    return f"{cohen_line}\n\n{i2_line}\n\n{p_line}\n\n{ci_line}"

# ----------------------------------------------------------------------------
# 8. INTERFACE PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    lang = st.session_state.lang
    
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
    
    st.title(t('title'))
    st.caption(t('subtitle'))
    
    # --------------------------------------------------------------------
    # SELEÇÃO DE ESTUDOS
    # --------------------------------------------------------------------
    df_studies = pd.DataFrame(STUDIES)
    
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
        if lang == 'pt':
            col4.write(f"{row['n']:,}".replace(",", "."))
        else:
            col4.write(f"{row['n']:,}")
        col5.write(fmt_num(row['d'], 2))
        col6.write(fmt_num(row['se'], 3))
    
    df_selected = df_studies.loc[selected_ids].copy()
    st.caption(f"**{len(df_selected)}** de **{len(df_studies)}** estudos selecionados.")
    
    # --------------------------------------------------------------------
    # META-ANÁLISE
    # --------------------------------------------------------------------
    if len(df_selected) >= 2:
        results = run_meta_analysis(df_selected)
    else:
        results = None
        st.warning("Selecione pelo menos **2 estudos** para realizar a meta-análise.")
    
    if results:
        with st.sidebar:
            st.subheader("📊 Current Results")
            st.metric(t('q_stat'), f"{fmt_num(results['q'], 2)}")
            st.metric(t('i2'), f"{fmt_num(results['i2'], 1)} %")
            st.metric(t('tau2'), f"{fmt_num(results['tau2'], 4)}")
        
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
        
        # Tabela de Pesos
        st.divider()
        st.subheader(t('weight_table'))
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
        if lang == 'pt':
            df_display[t('n')] = df_display[t('n')].apply(lambda x: f"{x:,}".replace(",", "."))
        else:
            df_display[t('n')] = df_display[t('n')].apply(lambda x: f"{x:,}")
        df_display[t('d')] = df_display[t('d')].apply(lambda x: fmt_num(x, 2))
        df_display[t('se')] = df_display[t('se')].apply(lambda x: fmt_num(x, 3))
        st.dataframe(df_display, use_container_width=True)
        st.caption(t('weight_info'))
        
        # --------------------------------------------------------------------
        # GRÁFICOS
        # --------------------------------------------------------------------
        st.divider()
        st.subheader(t('forest_plot'))
        fig_forest = plot_forest(df_selected, results, lang)
        if fig_forest:
            st.pyplot(fig_forest)
            st.caption(t('download_hint'))
        
        st.divider()
        st.subheader(t('funnel_plot'))
        fig_funnel = plot_funnel(df_selected, results, lang)
        if fig_funnel:
            st.pyplot(fig_funnel)
            st.caption(t('funnel_interpret'))
        
        st.divider()
        st.subheader(t('sensitivity_plot'))
        fig_sens = plot_sensitivity(df_selected, results, lang)
        if fig_sens:
            st.pyplot(fig_sens)
            st.caption(t('sensitivity_interpret'))
        
        # --------------------------------------------------------------------
        # INTERPRETAÇÃO
        # --------------------------------------------------------------------
        st.divider()
        st.subheader(t('interpretation_title'))
        interpretation_text = interpret_results(results, lang)
        st.markdown(interpretation_text)
    
    else:
        st.info("👈 Selecione mais estudos na tabela acima.")

if __name__ == "__main__":
    main()
