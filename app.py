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

# Inicializa o estado do idioma
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'  # Padrão: Inglês

# Função para alternar o idioma
def set_lang(lang):
    st.session_state.lang = lang

# Dicionários de tradução (Strings do App)
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
        "info_sidebar": "ℹ️ About this App",
        "sidebar_text": "Select/deselect studies on the left to see how the pooled effect changes. The app uses a Random-Effects model (DerSimonian-Laird).\n\nFormatted for:",
        "download_hint": "Right-click on the plot to save it.",
        "method_title": "⚙️ Statistical Method",
        "method_text": "**DerSimonian-Laird (DL)** random-effects model.\n\nQ-statistic (Heterogeneity): **{q:.2f}**\n\nWeights are calculated based on within-study variance + between-studies variance (Tau²).",
        "exclude_label": "Excluded Studies",
        "excluded_note": "No studies excluded. All selected.",
        "ci_label": "CI: [{:.3f}, {:.3f}]"
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
        "info_sidebar": "ℹ️ Sobre este App",
        "sidebar_text": "Selecione/desselecione os estudos à esquerda para ver como o efeito combinado muda. O app usa o modelo de Efeitos Aleatórios (DerSimonian-Laird).\n\nFormatado para:",
        "download_hint": "Clique com o botão direito no gráfico para salvá-lo.",
        "method_title": "⚙️ Método Estatístico",
        "method_text": "Modelo de efeitos aleatórios **DerSimonian-Laird (DL)**.\n\nEstatística Q (Heterogeneidade): **{q:.2f}**\n\nOs pesos são calculados com base na variância intra-estudo + variância entre estudos (Tau²).",
        "exclude_label": "Estudos Excluídos",
        "excluded_note": "Nenhum estudo excluído. Todos selecionados.",
        "ci_label": "IC: [{:.3f}, {:.3f}]"
    }
}

# Função para obter o texto atual
def t(key, **kwargs):
    lang = st.session_state.lang
    text = TEXTS[lang].get(key, TEXTS["en"].get(key, key))
    if kwargs:
        return text.format(**kwargs)
    return text

# Função para formatar números (Localização)
def fmt_num(value, decimals=3):
    lang = st.session_state.lang
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        formatted = f"{value:.{decimals}f}"
        if lang == "pt":
            # Troca ponto por vírgula para decimais (Brasil)
            formatted = formatted.replace(".", ",")
        return formatted
    return str(value)

# ----------------------------------------------------------------------------
# 2. BASE DE DADOS DOS ESTUDOS (FINALIZADA E CORRIGIDA)
# ----------------------------------------------------------------------------
STUDIES = [
    {"id": "Erridge et al. (2026)", "type": "Coorte", "n": 8945, "d": 0.84, "se": 0.009},
    {"id": "Datta et al. (2025a)", "type": "Coorte", "n": 517, "d": 0.68, "se": 0.030},
    {"id": "Datta et al. (2025b)", "type": "Coorte", "n": 269, "d": 0.32, "se": 0.020},
    {"id": "Short et al. (2025)", "type": "Coorte", "n": 137, "d": 0.80, "se": 0.068},
    {"id": "Vivek et al. (2024)", "type": "Coorte", "n": 40, "d": 1.35, "se": 0.213},
    {"id": "Cooke et al. (2023)", "type": "Coorte", "n": 163, "d": 0.55, "se": 0.043},
    {"id": "Vaddiparti et al. (2023)", "type": "Coorte", "n": 15, "d": 1.44, "se": 0.372},
    {"id": "Ried et al. (2023)", "type": "RCT", "n": 29, "d": 0.69, "se": 0.128},
    {"id": "Montebello et al. (2022)", "type": "RCT", "n": 128, "d": 0.60, "se": 0.181},
    # OBS: O outlier (Pakdee) está comentado/removido da base principal, mas se quiser manter desabilitado, descomente a linha abaixo:
    # {"id": "Pakdee, Sribunrieng e Poowanna (2026)", "type": "RCT", "n": 20, "d": 2.84, "se": 0.635},
]

# ----------------------------------------------------------------------------
# 3. FUNÇÃO PRINCIPAL DE META-ANÁLISE (Modelo de Efeitos Aleatórios - DL)
# ----------------------------------------------------------------------------
def run_meta_analysis(df):
    """
    Executa o modelo de efeitos aleatórios de DerSimonian-Laird.
    Retorna: dicionário com resultados.
    """
    if df.empty:
        return None
    
    d = df['d'].values
    se = df['se'].values
    k = len(d)
    
    # Pesos fixos (inverso da variância)
    w = 1 / (se ** 2)
    
    # Estatística Q (heterogeneidade)
    sum_w = np.sum(w)
    sum_wd = np.sum(w * d)
    sum_wd2 = np.sum(w * (d ** 2))
    Q = sum_wd2 - (sum_wd ** 2) / sum_w
    
    # Tau² (DerSimonian-Laird)
    df_het = k - 1
    if df_het > 0:
        tau2 = max(0, (Q - df_het) / (sum_w - (np.sum(w ** 2) / sum_w)))
    else:
        tau2 = 0
    
    # Pesos ajustados (efeitos aleatórios)
    w_star = 1 / (se ** 2 + tau2)
    sum_w_star = np.sum(w_star)
    
    # Efeito combinado (Cohen's d)
    pooled_d = np.sum(w_star * d) / sum_w_star
    
    # Erro padrão do efeito combinado
    se_pooled = np.sqrt(1 / sum_w_star)
    
    # Intervalo de Confiança (95%)
    ci_lb = pooled_d - 1.96 * se_pooled
    ci_ub = pooled_d + 1.96 * se_pooled
    
    # Valor-p
    z = pooled_d / se_pooled
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # I² (porcentagem da variabilidade devido à heterogeneidade)
    i2 = max(0, ((Q - df_het) / Q) * 100) if Q > 0 else 0
    
    # Calcular pesos percentuais
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
# 4. FUNÇÃO PARA GERAR O FOREST PLOT (MATPLOTLIB)
# ----------------------------------------------------------------------------
def plot_forest(df, results, lang):
    if df.empty or results is None:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ordenar pelo tamanho do efeito (para visualização)
    df_sorted = df.copy()
    df_sorted['weight'] = results['weights']
    df_sorted = df_sorted.sort_values('d', ascending=False).reset_index(drop=True)
    
    y_pos = np.arange(len(df_sorted))
    d_values = df_sorted['d'].values
    se_values = df_sorted['se'].values
    ci_lower = d_values - 1.96 * se_values
    ci_upper = d_values + 1.96 * se_values
    
    # Plotar linhas de CI (Intervalo de Confiança)
    ax.hlines(y=y_pos, xmin=ci_lower, xmax=ci_upper, color='gray', linewidth=1, alpha=0.7)
    
    # Plotar quadrados (tamanho proporcional ao peso)
    sizes = (df_sorted['weight'] / df_sorted['weight'].max()) * 100 + 20
    ax.scatter(d_values, y_pos, s=sizes, color='#1f77b4', zorder=5, edgecolors='black', linewidth=0.5)
    
    # Nomes dos estudos
    ax.set_yticks(y_pos)
    
    # Formatar nomes (cortar se muito longos)
    labels = df_sorted['id'].tolist()
    ax.set_yticklabels(labels, fontsize=9)
    
    # Linha vertical no zero (referência)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Linha vertical no efeito combinado
    ax.axvline(x=results['pooled_d'], color='red', linestyle='-', linewidth=1.5, alpha=0.6)
    
    # Configurar eixos
    x_min = min(-0.5, ci_lower.min() - 0.2) if len(ci_lower) > 0 else -0.5
    x_max = max(2.5, ci_upper.max() + 0.2) if len(ci_upper) > 0 else 2.5
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(t('d') + " (Cohen's d)" if lang == 'en' else "d de Cohen", fontsize=10)
    ax.set_title(t('forest_plot'), fontsize=12)
    
    # Adicionar diamante do efeito combinado (na parte inferior)
    diamond_y = -0.5
    diamond_x = results['pooled_d']
    diamond_width = results['ci_ub'] - results['ci_lb']
    # Desenhar diamante
    ax.plot([results['ci_lb'], diamond_x, results['ci_ub'], diamond_x, results['ci_lb']],
            [diamond_y, diamond_y - 0.2, diamond_y, diamond_y + 0.2, diamond_y],
            color='red', linewidth=2)
    ax.text(diamond_x, diamond_y - 0.5, 
            f"Pooled: {fmt_num(results['pooled_d'], 3)} [{fmt_num(results['ci_lb'], 3)}, {fmt_num(results['ci_ub'], 3)}]",
            ha='center', fontsize=9, color='red', fontweight='bold')
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

# ----------------------------------------------------------------------------
# 5. INTERFACE DO STREAMLIT
# ----------------------------------------------------------------------------
def main():
    lang = st.session_state.lang
    
    # Sidebar: Seleção de Idioma
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
        
        # Informações
        st.subheader(t('info_sidebar'))
        st.info(t('sidebar_text') + f" **{lang.upper()}**")
        st.caption("Developed for academic research purposes.")
        
        st.divider()
        st.subheader(t('method_title'))
        st.markdown(t('method_text', q=0.0))  # O valor Q será atualizado na seção de resultados
    
    # Título Principal
    st.title(t('title'))
    st.caption(t('subtitle'))
    
    # ----------------------------------------------------------------
    # CARREGAR E SELECIONAR ESTUDOS
    # ----------------------------------------------------------------
    df_studies = pd.DataFrame(STUDIES)
    
    # Adicionar coluna de seleção (checkbox)
    st.subheader(t('select_studies'))
    
    # Criar colunas para layout da tabela
    cols = st.columns([0.5, 3, 1.5, 1, 1.5, 1.5])
    cols[0].write("**" + t('include') + "**")
    cols[1].write("**" + t('study') + "**")
    cols[2].write("**" + t('type') + "**")
    cols[3].write("**" + t('n') + "**")
    cols[4].write("**" + t('d') + "**")
    cols[5].write("**" + t('se') + "**")
    
    selected_ids = []
    for idx, row in df_studies.iterrows():
        # Usar session state para manter o estado dos checkboxes
        key = f"include_{idx}"
        if key not in st.session_state:
            st.session_state[key] = True  # Todos selecionados por padrão (exceto outliers se quiser)
        
        # Define o estado inicial (desmarcar outliers automaticamente? Não, vamos deixar todos marcados por padrão)
        # Mas se quisermos, podemos desmarcar o Pakdee, mas ele está comentado na lista.
        
        cols = st.columns([0.5, 3, 1.5, 1, 1.5, 1.5])
        checked = cols[0].checkbox("", value=st.session_state[key], key=key, label_visibility="collapsed")
        st.session_state[key] = checked
        if checked:
            selected_ids.append(idx)
        
        cols[1].write(row['id'])
        cols[2].write(row['type'])
        cols[3].write(f"{row['n']:,}".replace(",", ".") if lang == 'pt' else f"{row['n']:,}")
        cols[4].write(fmt_num(row['d'], 2))
        cols[5].write(fmt_num(row['se'], 3))
    
    # Filtrar dados selecionados
    df_selected = df_studies.loc[selected_ids].copy()
    
    # Mostrar quantos estudos foram selecionados
    st.caption(f"**{len(df_selected)}** de **{len(df_studies)}** estudos selecionados.")
    
    # ----------------------------------------------------------------
    # RODAR META-ANÁLISE
    # ----------------------------------------------------------------
    if len(df_selected) >= 2:
        results = run_meta_analysis(df_selected)
    else:
        results = None
        st.warning("Selecione pelo menos **2 estudos** para realizar a meta-análise." if lang == 'en' else "Selecione pelo menos **2 estudos** para realizar a meta-análise.")
    
    if results:
        # Atualizar a sidebar com os resultados de Q
        # (A sidebar não atualiza dinamicamente via markdown com .format, então vamos recriar o markdown)
        with st.sidebar:
            st.subheader("📊 Current Results")
            st.metric(t('q_stat'), f"{fmt_num(results['q'], 2)}")
            st.metric(t('i2'), f"{fmt_num(results['i2'], 1)} %")
            st.metric(t('tau2'), f"{fmt_num(results['tau2'], 4)}")
        
        # ----------------------------------------------------------------
        # EXIBIR RESULTADOS
        # ----------------------------------------------------------------
        st.divider()
        st.subheader(t('results'))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label=t('pooled_effect'),
                value=fmt_num(results['pooled_d'], 3),
                delta=None
            )
        with col2:
            ci_text = f"[{fmt_num(results['ci_lb'], 3)}, {fmt_num(results['ci_ub'], 3)}]"
            st.metric(label=t('ci'), value=ci_text)
        with col3:
            # Formata o valor-p (se muito pequeno, mostra "< 0.001")
            if results['p_val'] < 0.001:
                p_text = "< 0.001" if lang == 'en' else "< 0,001"
            else:
                p_text = fmt_num(results['p_val'], 3)
            st.metric(label=t('p_value'), value=p_text)
        with col4:
            st.metric(label=t('i2'), value=f"{fmt_num(results['i2'], 1)} %")
        
        # Tabela de pesos
        st.divider()
        st.subheader("📋 Study Weights")
        df_weights = df_selected.copy()
        df_weights['Weight (%)'] = [fmt_num(w, 2) for w in results['weights']]
        # Reordenar colunas
        display_cols = ['id', 'type', 'n', 'd', 'se', 'Weight (%)']
        # Renomear para o idioma
        rename_map = {
            'id': t('study'), 
            'type': t('type'), 
            'n': t('n'), 
            'd': t('d'), 
            'se': t('se'),
            'Weight (%)': t('weight')
        }
        df_display = df_weights[['id', 'type', 'n', 'd', 'se', 'Weight (%)']].rename(columns=rename_map)
        # Formatar números na tabela
        if lang == 'pt':
            df_display[t('n')] = df_display[t('n')].apply(lambda x: f"{x:,}".replace(",", "."))
        else:
            df_display[t('n')] = df_display[t('n')].apply(lambda x: f"{x:,}")
        df_display[t('d')] = df_display[t('d')].apply(lambda x: fmt_num(x, 2))
        df_display[t('se')] = df_display[t('se')].apply(lambda x: fmt_num(x, 3))
        
        st.dataframe(df_display, use_container_width=True)
        
        # ----------------------------------------------------------------
        # FOREST PLOT
        # ----------------------------------------------------------------
        st.divider()
        st.subheader(t('forest_plot'))
        
        fig = plot_forest(df_selected, results, lang)
        if fig:
            st.pyplot(fig)
            st.caption(t('download_hint'))
    
    else:
        # Mensagem quando menos de 2 estudos estão selecionados
        st.info("👈 Selecione mais estudos na tabela acima." if lang == 'en' else "👈 Selecione mais estudos na tabela acima.")

if __name__ == "__main__":
    main()
