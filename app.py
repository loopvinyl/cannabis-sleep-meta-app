import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO INICIAL
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Meta-Analysis App - Sleep & Cannabis", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "en"  # Padrão: inglês

# ----------------------------------------------------------------------------
# 2. FUNÇÕES COMPARTILHADAS (meta-análise, gráficos, formatação)
# ----------------------------------------------------------------------------
def fmt_num(value, decimals=3):
    """Formata números com vírgula decimal no PT e ponto no EN."""
    lang = st.session_state.lang
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        formatted = f"{value:.{decimals}f}"
        if lang == "pt":
            formatted = formatted.replace(".", ",")
        return formatted
    return str(value)

def run_meta_analysis(df):
    if df.empty:
        return None
    d = df["d"].values
    se = df["se"].values
    k = len(d)
    w = 1 / (se**2)
    sum_w = np.sum(w)
    sum_wd = np.sum(w * d)
    sum_wd2 = np.sum(w * (d**2))
    Q = sum_wd2 - (sum_wd**2) / sum_w
    df_het = k - 1
    if df_het > 0:
        tau2 = max(0, (Q - df_het) / (sum_w - (np.sum(w**2) / sum_w)))
    else:
        tau2 = 0
    w_star = 1 / (se**2 + tau2)
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
        "w_star": w_star,
    }

def plot_forest(df, results):
    if df.empty or results is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    df_sorted = df.copy()
    df_sorted["weight"] = results["weights"]
    df_sorted = df_sorted.sort_values("d", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    d_values = df_sorted["d"].values
    se_values = df_sorted["se"].values
    ci_lower = d_values - 1.96 * se_values
    ci_upper = d_values + 1.96 * se_values
    ax.hlines(y=y_pos, xmin=ci_lower, xmax=ci_upper, color="gray", linewidth=1, alpha=0.7)
    sizes = (df_sorted["weight"] / df_sorted["weight"].max()) * 100 + 20
    ax.scatter(d_values, y_pos, s=sizes, color="#1f77b4", zorder=5, edgecolors="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["id"].tolist(), fontsize=9)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=results["pooled_d"], color="red", linestyle="-", linewidth=1.5, alpha=0.6)
    x_min = min(-0.5, ci_lower.min() - 0.2) if len(ci_lower) > 0 else -0.5
    x_max = max(2.5, ci_upper.max() + 0.2) if len(ci_upper) > 0 else 2.5
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Cohen's d" if st.session_state.lang == "en" else "d de Cohen", fontsize=10)
    ax.set_title("Forest Plot", fontsize=12)
    diamond_y = -0.5
    diamond_x = results["pooled_d"]
    ax.plot(
        [results["ci_lb"], diamond_x, results["ci_ub"], diamond_x, results["ci_lb"]],
        [diamond_y, diamond_y - 0.2, diamond_y, diamond_y + 0.2, diamond_y],
        color="red",
        linewidth=2,
    )
    ax.text(
        diamond_x,
        diamond_y - 0.5,
        f"Pooled: {fmt_num(results['pooled_d'], 3)} [{fmt_num(results['ci_lb'], 3)}, {fmt_num(results['ci_ub'], 3)}]",
        ha="center",
        fontsize=9,
        color="red",
        fontweight="bold",
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig

def plot_funnel(df, results):
    if df.empty or results is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    d = df["d"].values
    se = df["se"].values
    ax.scatter(d, se, color="#1f77b4", zorder=5, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Cohen's d" if st.session_state.lang == "en" else "d de Cohen", fontsize=10)
    ax.set_ylabel("Standard Error" if st.session_state.lang == "en" else "Erro Padrão", fontsize=10)
    ax.set_title("Funnel Plot", fontsize=12)
    ax.axvline(x=results["pooled_d"], color="red", linestyle="-", linewidth=1.0, alpha=0.5)
    x_limits = np.linspace(results["ci_lb"] - 0.5, results["ci_ub"] + 0.5, 100)
    y_limits = (np.max(se) / (results["ci_ub"] - results["ci_lb"] + 1)) * np.abs(x_limits - results["pooled_d"])
    ax.fill_between(x_limits, 0, y_limits, color="gray", alpha=0.1)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig

def plot_sensitivity(df, results):
    if df.empty or results is None or len(df) < 2:
        return None
    d_orig = results["pooled_d"]
    studies = df["id"].tolist()
    d_loo = []
    for i in range(len(df)):
        df_loo = df.drop(df.index[i])
        res_loo = run_meta_analysis(df_loo)
        if res_loo is not None:
            d_loo.append(res_loo["pooled_d"])
        else:
            d_loo.append(np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(studies))
    ax.scatter(d_loo, y_pos, color="#1f77b4", zorder=5, edgecolors="black", linewidth=0.5)
    ax.axvline(x=d_orig, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Original pooled d")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(studies, fontsize=8)
    ax.set_xlabel("Cohen's d (leave-one-out)" if st.session_state.lang == "en" else "d de Cohen (leave-one-out)", fontsize=10)
    ax.set_title("Sensitivity Analysis (Leave-One-Out)" if st.session_state.lang == "en" else "Análise de Sensibilidade (Leave-One-Out)", fontsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

# ----------------------------------------------------------------------------
# 3. BASE DE DADOS (10 ESTUDOS)
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
# 4. INTERFACE PRINCIPAL (UI DUPLICADA)
# ----------------------------------------------------------------------------
def main():
    lang = st.session_state.lang

    # ======================================================================
    # SIDEBAR (comum a ambos os idiomas – apenas o seletor de idioma)
    # ======================================================================
    with st.sidebar:
        st.header("Language / Idioma")
        lang_choice = st.radio(
            "",
            options=["English", "Português"],
            index=0 if lang == "en" else 1,
            label_visibility="collapsed",
        )
        if lang_choice == "Português" and lang != "pt":
            st.session_state.lang = "pt"
            st.rerun()
        elif lang_choice == "English" and lang != "en":
            st.session_state.lang = "en"
            st.rerun()

        st.divider()

        # Informações comuns
        if lang == "en":
            st.subheader("ℹ️ About this App")
            st.info(
                "Select/deselect studies on the left to see how the pooled effect changes. "
                "The app uses a Random-Effects model (DerSimonian-Laird).\n\n"
                "Formatted for: **EN**"
            )
            st.caption("Developed for academic research purposes.")
            st.divider()
            st.subheader("⚙️ Statistical Method")
            st.markdown(
                "**DerSimonian-Laird (DL)** random-effects model.\n\n"
                "Q-statistic (Heterogeneity): **{q:.2f}**\n\n"
                "Weights are calculated based on within-study variance + between-studies variance (Tau²)."
            )
        else:
            st.subheader("ℹ️ Sobre este App")
            st.info(
                "Selecione/desselecione os estudos à esquerda para ver como o efeito combinado muda. "
                "O app usa o modelo de Efeitos Aleatórios (DerSimonian-Laird).\n\n"
                "Formatado para: **PT-BR**"
            )
            st.caption("Desenvolvido para fins acadêmicos.")
            st.divider()
            st.subheader("⚙️ Método Estatístico")
            st.markdown(
                "Modelo de efeitos aleatórios **DerSimonian-Laird (DL)**.\n\n"
                "Estatística Q (Heterogeneidade): **{q:.2f}**\n\n"
                "Os pesos são calculados com base na variância intra-estudo + variância entre estudos (Tau²)."
            )

    # ======================================================================
    # SE O IDIOMA FOR INGLÊS → RENDERIZA A UI EM INGLÊS
    # ======================================================================
    if lang == "en":
        # ---- TÍTULO ----
        st.title("🗂️ Interactive Meta-Analysis: THC/Cannabis & Sleep")
        st.caption("Based on the methodology of Tait et al. (2026) - Quality of Life Research")

        # ---- SELECIONAR ESTUDOS ----
        df_studies = pd.DataFrame(STUDIES)

        for idx, row in df_studies.iterrows():
            key = f"include_{idx}"
            if key not in st.session_state:
                if row["id"] == "Pakdee, Sribunrieng e Poowanna (2026)":
                    st.session_state[key] = False
                else:
                    st.session_state[key] = True

        st.subheader("📊 Select Studies to Include")

        col1, col2, col3, col4, col5, col6 = st.columns([0.6, 3.5, 1.5, 1, 1.5, 1.5])
        col1.markdown("<div style='text-align: center; font-weight: bold;'>Include</div>", unsafe_allow_html=True)
        col2.markdown("<div style='text-align: left; font-weight: bold;'>Study</div>", unsafe_allow_html=True)
        col3.markdown("<div style='text-align: center; font-weight: bold;'>Type</div>", unsafe_allow_html=True)
        col4.markdown("<div style='text-align: center; font-weight: bold;'>N</div>", unsafe_allow_html=True)
        col5.markdown("<div style='text-align: center; font-weight: bold;'>Cohen's d</div>", unsafe_allow_html=True)
        col6.markdown("<div style='text-align: center; font-weight: bold;'>Std. Error</div>", unsafe_allow_html=True)

        selected_ids = []
        for idx, row in df_studies.iterrows():
            key = f"include_{idx}"
            c1, c2, c3, c4, c5, c6 = st.columns([0.6, 3.5, 1.5, 1, 1.5, 1.5])
            checked = c1.checkbox("", value=st.session_state[key], key=key, label_visibility="collapsed")
            if checked:
                selected_ids.append(idx)
            c2.write(row["id"])
            c3.write(row["type"])
            c4.write(f"{row['n']:,}")
            c5.write(f"{row['d']:.2f}")
            c6.write(f"{row['se']:.3f}")

        df_selected = df_studies.loc[selected_ids].copy()
        st.caption(f"**{len(df_selected)}** out of **{len(df_studies)}** studies selected.")

        # ---- EXECUTAR META-ANÁLISE ----
        if len(df_selected) >= 2:
            results = run_meta_analysis(df_selected)
        else:
            results = None
            st.warning("Select at least **2 studies** to run the meta-analysis.")

        if results:
            # Sidebar com resultados atuais
            with st.sidebar:
                st.subheader("📊 Current Results")
                st.metric("Q-statistic", f"{fmt_num(results['q'], 2)}")
                st.metric("I²", f"{fmt_num(results['i2'], 1)} %")
                st.metric("Tau²", f"{fmt_num(results['tau2'], 4)}")

            st.divider()
            st.subheader("📈 Meta-Analysis Results")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Pooled Effect (Cohen's d)", fmt_num(results["pooled_d"], 3))
            with c2:
                ci_text = f"[{fmt_num(results['ci_lb'], 3)}, {fmt_num(results['ci_ub'], 3)}]"
                st.metric("95% Confidence Interval", ci_text)
            with c3:
                if results["p_val"] < 0.001:
                    p_text = "< 0.001"
                else:
                    p_text = fmt_num(results["p_val"], 3)
                st.metric("P-value", p_text)
            with c4:
                st.metric("I²", f"{fmt_num(results['i2'], 1)} %")

            # Tabela de pesos
            st.divider()
            st.subheader("📋 Study Weights")
            df_weights = df_selected.copy()
            df_weights["Weight (%)"] = [fmt_num(w, 2) for w in results["weights"]]
            df_display = df_weights[["id", "type", "n", "d", "se", "Weight (%)"]]
            df_display["n"] = df_display["n"].apply(lambda x: f"{x:,}")
            df_display["d"] = df_display["d"].apply(lambda x: f"{x:.2f}")
            df_display["se"] = df_display["se"].apply(lambda x: f"{x:.3f}")
            st.dataframe(df_display, use_container_width=True)
            st.caption("Larger squares indicate greater weight in the meta-analysis.")

            # Gráficos
            st.divider()
            st.subheader("🌲 Forest Plot")
            fig_forest = plot_forest(df_selected, results)
            if fig_forest:
                st.pyplot(fig_forest)
                st.caption("Right-click on the plot to save it.")

            st.divider()
            st.subheader("🔍 Funnel Plot (Publication Bias)")
            fig_funnel = plot_funnel(df_selected, results)
            if fig_funnel:
                st.pyplot(fig_funnel)
                st.caption(
                    "The funnel plot shows the distribution of study effects against their standard errors. "
                    "Asymmetry may indicate publication bias."
                )

            st.divider()
            st.subheader("📉 Sensitivity Analysis (Leave-One-Out)")
            fig_sens = plot_sensitivity(df_selected, results)
            if fig_sens:
                st.pyplot(fig_sens)
                st.caption(
                    "This plot shows the pooled effect after removing each study one at a time. "
                    "If the effect remains stable, the result is robust."
                )

            # Interpretação em inglês
            st.divider()
            st.subheader("📖 Interpretation Guide")

            d = results["pooled_d"]
            ci_lb = results["ci_lb"]
            ci_ub = results["ci_ub"]
            p = results["p_val"]
            i2 = results["i2"]

            if abs(d) < 0.2:
                effect_desc = "small"
            elif abs(d) < 0.5:
                effect_desc = "moderate"
            else:
                effect_desc = "large"

            st.markdown(
                f"**Cohen's d = {d:.3f}** – This is a **{effect_desc}** effect size. "
                f"It indicates that the THC/cannabis intervention improved sleep quality by "
                f"{abs(d):.1f} standard deviations compared to control/baseline."
            )

            if i2 < 25:
                het_desc = "low"
                het_text = "The studies are relatively consistent, suggesting that the pooled effect is reliable."
            elif i2 < 50:
                het_desc = "moderate"
                het_text = "There is some variability among studies, but the pooled effect remains informative."
            else:
                het_desc = "high"
                het_text = (
                    "There is substantial variability among studies. This may reflect differences in populations, "
                    "doses, or outcome measures. A random-effects model was used to account for this."
                )

            st.markdown(f"**I² = {i2:.1f}%** – This indicates **{het_desc}** heterogeneity. {het_text}")

            if p < 0.05:
                p_text = "statistically significant (p < 0.05)"
            else:
                p_text = "not statistically significant (p ≥ 0.05)"
            st.markdown(f"**p-value = {fmt_num(p, 3)}** – The result is **{p_text}**.")

            if ci_lb < 0 < ci_ub:
                ci_does = "does"
                ci_support = "suggesting no significant effect"
            else:
                ci_does = "does not"
                ci_support = "supporting a significant effect"
            st.markdown(
                f"The 95% confidence interval [{fmt_num(ci_lb, 3)}, {fmt_num(ci_ub, 3)}] "
                f"{ci_does} contain zero, {ci_support}."
            )

        else:
            st.info("👈 Select more studies in the table above.")

    # ======================================================================
    # SE O IDIOMA FOR PORTUGUÊS → RENDERIZA A UI EM PORTUGUÊS
    # ======================================================================
    else:
        # ---- TÍTULO ----
        st.title("🗂️ Meta-Análise Interativa: THC/Cannabis & Sono")
        st.caption("Baseado na metodologia de Tait et al. (2026) - Quality of Life Research")

        # ---- SELECIONAR ESTUDOS ----
        df_studies = pd.DataFrame(STUDIES)

        for idx, row in df_studies.iterrows():
            key = f"include_{idx}"
            if key not in st.session_state:
                if row["id"] == "Pakdee, Sribunrieng e Poowanna (2026)":
                    st.session_state[key] = False
                else:
                    st.session_state[key] = True

        st.subheader("📊 Selecione os Estudos para Incluir")

        col1, col2, col3, col4, col5, col6 = st.columns([0.6, 3.5, 1.5, 1, 1.5, 1.5])
        col1.markdown("<div style='text-align: center; font-weight: bold;'>Incluir</div>", unsafe_allow_html=True)
        col2.markdown("<div style='text-align: left; font-weight: bold;'>Estudo</div>", unsafe_allow_html=True)
        col3.markdown("<div style='text-align: center; font-weight: bold;'>Tipo</div>", unsafe_allow_html=True)
        col4.markdown("<div style='text-align: center; font-weight: bold;'>N</div>", unsafe_allow_html=True)
        col5.markdown("<div style='text-align: center; font-weight: bold;'>d de Cohen</div>", unsafe_allow_html=True)
        col6.markdown("<div style='text-align: center; font-weight: bold;'>Erro Padrão</div>", unsafe_allow_html=True)

        selected_ids = []
        for idx, row in df_studies.iterrows():
            key = f"include_{idx}"
            c1, c2, c3, c4, c5, c6 = st.columns([0.6, 3.5, 1.5, 1, 1.5, 1.5])
            checked = c1.checkbox("", value=st.session_state[key], key=key, label_visibility="collapsed")
            if checked:
                selected_ids.append(idx)

            tipo_traduzido = "Coorte" if row["type"] == "Coorte" else "ECR"
            c2.write(row["id"])
            c3.write(tipo_traduzido)
            c4.write(f"{row['n']:,}".replace(",", "."))
            c5.write(fmt_num(row["d"], 2))
            c6.write(fmt_num(row["se"], 3))

        df_selected = df_studies.loc[selected_ids].copy()
        st.caption(f"**{len(df_selected)}** de **{len(df_studies)}** estudos selecionados.")

        # ---- EXECUTAR META-ANÁLISE ----
        if len(df_selected) >= 2:
            results = run_meta_analysis(df_selected)
        else:
            results = None
            st.warning("Selecione pelo menos **2 estudos** para realizar a meta-análise.")

        if results:
            # Sidebar com resultados atuais
            with st.sidebar:
                st.subheader("📊 Resultados Atuais")
                st.metric("Estatística Q", fmt_num(results["q"], 2))
                st.metric("I²", f"{fmt_num(results['i2'], 1)} %")
                st.metric("Tau²", fmt_num(results["tau2"], 4))

            st.divider()
            st.subheader("📈 Resultados da Meta-Análise")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Efeito Combinado (d de Cohen)", fmt_num(results["pooled_d"], 3))
            with c2:
                ci_text = f"[{fmt_num(results['ci_lb'], 3)}, {fmt_num(results['ci_ub'], 3)}]"
                st.metric("Intervalo de Confiança 95%", ci_text)
            with c3:
                if results["p_val"] < 0.001:
                    p_text = "< 0,001"
                else:
                    p_text = fmt_num(results["p_val"], 3)
                st.metric("Valor-p", p_text)
            with c4:
                st.metric("I²", f"{fmt_num(results['i2'], 1)} %")

            # Tabela de pesos
            st.divider()
            st.subheader("📋 Pesos dos Estudos")
            df_weights = df_selected.copy()
            df_weights["Peso (%)"] = [fmt_num(w, 2) for w in results["weights"]]
            df_display = df_weights[["id", "type", "n", "d", "se", "Peso (%)"]]
            df_display["n"] = df_display["n"].apply(lambda x: f"{x:,}".replace(",", "."))
            df_display["d"] = df_display["d"].apply(lambda x: fmt_num(x, 2))
            df_display["se"] = df_display["se"].apply(lambda x: fmt_num(x, 3))
            df_display = df_display.rename(columns={
                "id": "Estudo",
                "type": "Tipo",
                "n": "N",
                "d": "d de Cohen",
                "se": "Erro Padrão",
                "Peso (%)": "Peso (%)"
            })
            st.dataframe(df_display, use_container_width=True)
            st.caption("Quadrados maiores indicam maior peso na meta-análise.")

            # Gráficos
            st.divider()
            st.subheader("🌲 Forest Plot")
            fig_forest = plot_forest(df_selected, results)
            if fig_forest:
                st.pyplot(fig_forest)
                st.caption("Clique com o botão direito no gráfico para salvá-lo.")

            st.divider()
            st.subheader("🔍 Gráfico de Funil (Viés de Publicação)")
            fig_funnel = plot_funnel(df_selected, results)
            if fig_funnel:
                st.pyplot(fig_funnel)
                st.caption(
                    "O gráfico de funil mostra a distribuição dos efeitos dos estudos contra seus erros padrão. "
                    "Assimetria pode indicar viés de publicação."
                )

            st.divider()
            st.subheader("📉 Análise de Sensibilidade (Leave-One-Out)")
            fig_sens = plot_sensitivity(df_selected, results)
            if fig_sens:
                st.pyplot(fig_sens)
                st.caption(
                    "Este gráfico mostra o efeito combinado após remover cada estudo um por vez. "
                    "Se o efeito permanece estável, o resultado é robusto."
                )

            # Interpretação em português
            st.divider()
            st.subheader("📖 Guia de Interpretação")

            d = results["pooled_d"]
            ci_lb = results["ci_lb"]
            ci_ub = results["ci_ub"]
            p = results["p_val"]
            i2 = results["i2"]

            if abs(d) < 0.2:
                effect_desc = "pequeno"
            elif abs(d) < 0.5:
                effect_desc = "moderado"
            else:
                effect_desc = "grande"

            st.markdown(
                f"**d de Cohen = {d:.3f}** – Este é um efeito **{effect_desc}**. "
                f"Indica que a intervenção com THC/cannabis melhorou a qualidade do sono em "
                f"{abs(d):.1f} desvios padrão em comparação com o controle/baseline."
            )

            if i2 < 25:
                het_desc = "baixa"
                het_text = "Os estudos são relativamente consistentes, sugerindo que o efeito combinado é confiável."
            elif i2 < 50:
                het_desc = "moderada"
                het_text = "Há alguma variabilidade entre os estudos, mas o efeito combinado permanece informativo."
            else:
                het_desc = "alta"
                het_text = (
                    "Há variabilidade substancial entre os estudos. Isso pode refletir diferenças nas populações, "
                    "doses ou medidas de desfecho. Um modelo de efeitos aleatórios foi usado para considerar isso."
                )

            st.markdown(f"**I² = {i2:.1f}%** – Isso indica heterogeneidade **{het_desc}**. {het_text}")

            if p < 0.05:
                p_text = "estatisticamente significativo (p < 0,05)"
            else:
                p_text = "não estatisticamente significativo (p ≥ 0,05)"
            st.markdown(f"**Valor-p = {fmt_num(p, 3)}** – O resultado é **{p_text}**.")

            if ci_lb < 0 < ci_ub:
                ci_does = "sim"
                ci_support = "sugerindo que não há efeito significativo"
            else:
                ci_does = "não"
                ci_support = "apoiando um efeito significativo"
            st.markdown(
                f"O intervalo de confiança 95% [{fmt_num(ci_lb, 3)}, {fmt_num(ci_ub, 3)}] "
                f"{ci_does} contém zero, {ci_support}."
            )

        else:
            st.info("👈 Selecione mais estudos na tabela acima.")

if __name__ == "__main__":
    main()
